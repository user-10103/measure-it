#!/usr/bin/env python3
"""Find a GIS aerial ImageServer for every Florida county.

WHY: county_imagery.COUNTY_ENDPOINTS holds 5 of Florida's 67 counties, and
FCDOP — the statewide fallback — answers 499 Token Required to everything. So
62 of 67 counties fall through to NAIP, which is 0.6 m/px 4-band and OUT OF
DOMAIN for a facet model fine-tuned on 7-15 cm orthophotos. The report does not
look different when that happens; it is just wrong by more.

There is also a SECOND registry at adresses/pipeline/utils/county_imagery.py
holding Lee (7.62 cm, verified) which the pipeline copy does not have, while the
pipeline copy has Pasco/Sarasota/Broward which that one lacks. Neither is the
union. This engine's output is intended to end that split.

METHOD (from the 2026-08-24 survey — guessing hostnames does not work,
gis.leegov.com / maps.leepa.org / maps.stlucieco.gov do not even resolve):
  1. AGOL free-text search for the county's imagery services.
  2. Pull the owning org id off any hit, then enumerate that whole org — the
     org is the unit that publishes, so this finds services whose titles never
     mention "aerial".
  3. Probe each ImageServer's metadata for native pixel size, honouring the
     spatial reference's UNIT (Florida State Plane is in FEET; reading
     pixelSizeX as metres understates GSD by 3.28x and makes 0.25 ft look like
     world-class 25 cm imagery).
  4. Record capabilities. A TilesOnly MapServer cannot serve exportImage and
     needs a tile-mosaic code path that county_imagery.py does not have, so it
     is reported as FOUND-BUT-UNUSABLE rather than as coverage.

WHAT THIS DOES NOT DO: verify with a real pixel fetch. A service can advertise
15 cm and 404 every tile, or geoblock non-Florida egress (Indian River does).
`--verify LAT LON` fetches one chip per candidate; without it a county is
reported as ADVERTISED, never as verified. The distinction is the whole point —
an unverified endpoint promoted to "coverage" is how 62 counties came to look
handled.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

UA = {"User-Agent": "Mozilla/5.0 (measure-it county-imagery discovery)"}
AGOL = "https://www.arcgis.com/sharing/rest"

FL_COUNTIES = [
    "Alachua", "Baker", "Bay", "Bradford", "Brevard", "Broward", "Calhoun",
    "Charlotte", "Citrus", "Clay", "Collier", "Columbia", "DeSoto", "Dixie",
    "Duval", "Escambia", "Flagler", "Franklin", "Gadsden", "Gilchrist",
    "Glades", "Gulf", "Hamilton", "Hardee", "Hendry", "Hernando", "Highlands",
    "Hillsborough", "Holmes", "Indian River", "Jackson", "Jefferson",
    "Lafayette", "Lake", "Lee", "Leon", "Levy", "Liberty", "Madison",
    "Manatee", "Marion", "Martin", "Miami-Dade", "Monroe", "Nassau",
    "Okaloosa", "Okeechobee", "Orange", "Osceola", "Palm Beach", "Pasco",
    "Pinellas", "Polk", "Putnam", "St. Johns", "St. Lucie", "Santa Rosa",
    "Sarasota", "Seminole", "Sumter", "Suwannee", "Taylor", "Union",
    "Volusia", "Wakulla", "Walton", "Washington",
]
assert len(FL_COUNTIES) == 67, len(FL_COUNTIES)

# Metres per unit, by esri unit name. Florida State Plane is in US survey feet
# and several county servers publish in it; treating those as metres reports
# 0.25 ft imagery as 0.25 m and inverts the whole quality ranking.
UNIT_M = {"esriFeet": 0.3048, "esriSurveyFoot": 0.3048006096,
          "esriFoot_US": 0.3048006096, "esriMeters": 1.0, "esriMeter": 1.0}

USABLE_CAP = "Image"          # exportImage; TilesOnly services lack this

# Imagery older than this cannot measure a roof that exists today. Counties
# publish deep historic archives (Lee goes back to 1953), and those scans are
# often the FINEST-grained services in the org — a 1979 frame scanned at 1.4 cm
# outranks 2025 orthophotos at 7.6 cm on resolution alone. Ranking on GSD with
# no age term picked the 1979 scan for Lee on the first run of this engine: the
# same one-sided criterion as `cover` in building_select and `max(iou)` in the
# facet gate. Resolution is only meaningful among CURRENT imagery.
MIN_YEAR = 2018
_YEAR_RE = __import__("re").compile(r"(19|20)\d{2}")

# Titles that advertise partial coverage. "Beaches Imagery" covers the coast,
# not the county, and a county marked covered by it silently drops every inland
# address to NAIP.
PARTIAL_HINTS = ("beach", "coastal", "shoreline", "corridor", "downtown",
                 "park", "hurricane", "storm", "post-", "damage", "swipe")


def parse_year(*texts) -> Optional[int]:
    """Latest plausible year mentioned, or None. '2006-2008' -> 2008."""
    years = []
    for t in texts:
        if not t:
            continue
        years += [int(m.group()) for m in _YEAR_RE.finditer(str(t))]
    years = [y for y in years if 1930 <= y <= 2030]
    return max(years) if years else None


def looks_partial(title: str) -> bool:
    t = (title or "").lower()
    return any(h in t for h in PARTIAL_HINTS)


def _get(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def agol_search(q: str, num: int = 20) -> List[dict]:
    u = (f"{AGOL}/search?f=json&num={num}&sortField=modified&sortOrder=desc"
         f"&q={urllib.parse.quote(q)}")
    try:
        return _get(u).get("results", [])
    except Exception:  # noqa: BLE001 — a failed search is "found nothing"
        return []


_ORG_CACHE: Dict[str, Optional[str]] = {}
_ORG_NAME: Dict[str, str] = {}


_ORG_URL_RE = __import__("re").compile(
    r"services\d*\.arcgis\.com/([A-Za-z0-9]{10,})/", __import__("re").I)


def org_ids_from(item: dict) -> List[str]:
    """Org ids for a search hit, from its URL first and its orgId field second.

    ``community/users/<owner>`` needs authentication and returns nothing
    anonymously — it produced 0 orgs for every county tried. The org id is
    already in an AGOL-hosted service URL
    (``services2.arcgis.com/<ORGID>/arcgis/rest/...``), which is how the
    2026-08-24 survey did it and needs no credentials.
    """
    out = []
    m = _ORG_URL_RE.search(item.get("url") or "")
    if m:
        out.append(m.group(1))
    if item.get("orgId"):
        out.append(item["orgId"])
    return out


def org_for_owner(owner: str) -> Optional[str]:
    """Resolve an AGOL username to its org id.

    Search results carry ``orgId`` only sometimes — for "Lee County aerial
    imagery" every hit had ``orgId: None`` — so enumerating orgs from that
    field alone walked to whichever org DID populate it (FDEP, which publishes
    a statewide historic archive mentioning every county) and never reached the
    county's own portal. The owner username is always present.
    """
    if owner in _ORG_CACHE:
        return _ORG_CACHE[owner]
    org = None
    try:
        d = _get(f"{AGOL}/community/users/{urllib.parse.quote(owner)}?f=json", 12)
        org = d.get("orgId")
    except Exception:  # noqa: BLE001
        pass
    _ORG_CACHE[owner] = org
    return org


def org_name(org: str) -> str:
    if org not in _ORG_NAME:
        try:
            _ORG_NAME[org] = _get(f"{AGOL}/portals/{org}?f=json", 12).get("name") or ""
        except Exception:  # noqa: BLE001
            _ORG_NAME[org] = ""
    return _ORG_NAME[org]


def _county_tokens(county: str) -> List[str]:
    c = county.lower().replace("st. ", "st ").replace("-", " ")
    return [t for t in c.split() if len(t) > 2]


def candidates_for(county: str) -> List[dict]:
    """Image/Map services plausibly carrying this county's CURRENT imagery.

    Pass 1 searches for imagery by name. Pass 2 finds the COUNTY'S OWN AGOL org
    and enumerates it — that is the step that matters. Lee's 7.62 cm service
    lives on a standalone server (gisimageserver.leegov.com) that no
    imagery-titled search returns, but AGOL indexes 84 items for the Lee County
    Florida GIS org and the 2025 service is among them. Searching only for
    "<county> aerial" walks straight to FDEP's statewide historic archive,
    which mentions every county and is current for none.
    """
    seen: Dict[str, dict] = {}
    org_votes: Dict[str, int] = {}

    def note(it, how):
        url = it.get("url")
        if url and url not in seen:
            seen[url] = {"title": it.get("title"), "url": url,
                         "type": it.get("type"), "owner": it.get("owner"),
                         "orgid": it.get("orgId"), "found_by": how}

    for q in (f'"{county} County" Florida (aerial OR ortho OR imagery) '
              f'AND (type:"Image Service" OR type:"Map Service")',
              f'{county} Florida aerial AND type:"Image Service"'):
        for it in agol_search(q):
            note(it, "search")
            for o in org_ids_from(it):
                org_votes[o] = org_votes.get(o, 0) + 1

    # Widen the net for the county's own portal using NON-imagery layers every
    # county publishes; the org id rides along in the hosted service URL.
    for q in (f'"{county} County" Florida AND (parcels OR zoning OR boundary)',
              f'{county} County Florida GIS'):
        for it in agol_search(q):
            for o in org_ids_from(it):
                org_votes[o] = org_votes.get(o, 0) + 1

    toks = _county_tokens(county)
    scored = []
    for o in list(org_votes)[:14]:
        nm = org_name(o)
        scored.append((any(t in nm.lower() for t in toks), org_votes[o], o, nm))
    scored.sort(reverse=True)
    for matched, _votes, o, nm in scored[:3]:
        if not matched and scored and scored[0][0]:
            break            # a county-named org exists; do not dilute with others
        for it in agol_search(f'orgid:{o} AND (aerial OR ortho OR imagery)', 40):
            note(it, f"org:{nm}" + ("" if matched else " (NOT the county's own)"))
    # ORDER BEFORE THE CAP. --max-probes truncates this list, and insertion
    # order puts every free-text search hit ahead of the org enumeration -- so
    # Lee probed 14 candidates, all of them FDEP historic frames, and reported
    # HISTORIC-ONLY while "2025 Aerial Imagery" on the county's own server sat
    # at position 15 unprobed. The run printed a complete-looking result for a
    # search that never reached the answer. Sort newest first, and prefer the
    # county's own org when years tie.
    cands = list(seen.values())
    for c in cands:
        c["_year"] = parse_year(c.get("title")) or 0
        c["_own"] = 1 if str(c.get("found_by", "")).startswith("org:") \
            and "NOT the county" not in c["found_by"] else 0
    cands.sort(key=lambda c: (c["_year"], c["_own"]), reverse=True)
    return cands


def probe(url: str, timeout: int = 15) -> dict:
    """Read a service's declared native resolution and capabilities."""
    out = {"reachable": False, "gsd_m": None, "unit": None, "bands": None,
           "capabilities": None, "exportImage": False, "error": None}
    try:
        d = _get(url.rstrip("/") + "?f=json", timeout=timeout)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["reachable"] = True
    if isinstance(d, dict) and d.get("error"):
        out["error"] = str(d["error"].get("message") or d["error"])
        return out
    caps = d.get("capabilities") or ""
    out["capabilities"] = caps
    out["exportImage"] = USABLE_CAP in caps
    out["bands"] = d.get("bandCount")
    px = d.get("pixelSizeX")
    sr = (d.get("spatialReference") or {})
    wkid = sr.get("latestWkid") or sr.get("wkid")
    # The unit is not always on the service doc; fall back to the wkid range.
    unit = d.get("serviceDataType") and None
    unit = sr.get("unit") or unit
    if px:
        # Florida State Plane ftUS wkids: 2881-2883 (NAD83 HARN), 6440-6443.
        ft = unit in ("esriFeet", "esriSurveyFoot", "esriFoot_US") or (
            wkid in (2881, 2882, 2883, 6440, 6441, 6442, 6443))
        factor = UNIT_M.get(unit, 0.3048006096 if ft else 1.0)
        out["unit"] = unit or ("ftUS(by wkid)" if ft else "m(assumed)")
        out["gsd_m"] = round(float(px) * factor, 4)
    out["wkid"] = wkid
    out["name"] = d.get("name") or d.get("mapName")
    return out


def attributable(county: str, p: dict) -> bool:
    """Is this service actually THIS county's?

    Free-text search returns neighbours. Bradford — a rural county with no
    imagery portal of its own — was credited with "St. Andrews and St. Joseph
    2024 Imagery", which is Bay County's coastal survey, purely because it was
    the newest thing the search returned. That endpoint answers an exportImage
    for a Bradford bbox with a BLANK image rather than an error, so the county
    reads as covered and every chip comes back empty.

    A candidate counts only with provenance: it came from the county's own AGOL
    org, or its title names the county. Anything else is found, unattributed.
    """
    toks = _county_tokens(county)
    if not toks:
        return False
    title = (p.get("title") or "").lower()
    if any(t in title for t in toks):
        return True
    found_by = str(p.get("found_by") or "")
    return found_by.startswith("org:") and "NOT the county" not in found_by


def classify(county: str, probes: List[dict]) -> dict:
    """Best CURRENT usable endpoint, and an honest status when there is none.

    Candidates are filtered by age BEFORE being ranked by resolution. Ranking
    first and filtering later would still surface the 1979 scan as "best" with
    a footnote, and a footnote is not a filter.
    """
    for p in probes:
        p["year"] = parse_year(p.get("title"), p.get("name"))
        p["partial"] = looks_partial(p.get("title"))

    for p in probes:
        p["attributable"] = attributable(county, p)

    usable = [p for p in probes
              if p.get("exportImage") and p.get("gsd_m") and p["attributable"]]
    current = [p for p in usable
               if p.get("year") and p["year"] >= MIN_YEAR and not p["partial"]]
    if current:
        best = min(current, key=lambda p: (p["gsd_m"], -p["year"]))
        status = "ADVERTISED" if best["gsd_m"] <= 0.30 else "ADVERTISED-COARSE"
        return {"county": county, "status": status, "best": best,
                "n_candidates": len(probes), "n_current": len(current)}
    if usable:
        newest = max(usable, key=lambda p: (p.get("year") or 0))
        return {"county": county, "status": "HISTORIC-ONLY", "best": None,
                "n_candidates": len(probes),
                "note": f"usable services exist but none are current and "
                        f"county-wide (newest {newest.get('year')}, "
                        f"{'partial coverage' if newest.get('partial') else 'undated'})"}
    unattributed = [p for p in probes if p.get("exportImage") and p.get("gsd_m")
                    and not p.get("attributable")]
    if unattributed:
        return {"county": county, "status": "UNATTRIBUTED", "best": None,
                "n_candidates": len(probes),
                "note": f"{len(unattributed)} usable service(s) found but none "
                        f"traceable to this county (a neighbour's imagery in "
                        f"the search results) — would answer with a blank chip"}
    reachable = [p for p in probes if p.get("reachable") and not p.get("error")]
    if reachable:
        return {"county": county, "status": "FOUND-BUT-UNUSABLE", "best": None,
                "n_candidates": len(probes),
                "note": "services reachable but none expose exportImage "
                        "(TilesOnly needs a tile-mosaic path we do not have)"}
    return {"county": county, "status": "NONE", "best": None,
            "n_candidates": len(probes),
            "note": "no reachable imagery service found — this county falls "
                    "through to NAIP today"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counties", nargs="*", default=None,
                    help="subset to probe (default: all 67)")
    ap.add_argument("--out", default="fl_imagery_discovery.json")
    ap.add_argument("--max-probes", type=int, default=8,
                    help="services to probe per county")
    ap.add_argument("--sleep", type=float, default=0.3)
    a = ap.parse_args(argv)

    counties = a.counties or FL_COUNTIES
    results = []
    for i, c in enumerate(counties, 1):
        cands = candidates_for(c)
        probes = []
        for cand in cands[:a.max_probes]:
            p = probe(cand["url"])
            p.update({k: cand.get(k) for k in ("title", "url", "type", "owner",
                                               "orgid", "found_by", "org_name")})
            probes.append(p)
            time.sleep(a.sleep)
        res = classify(c, probes)
        res["probes"] = probes
        results.append(res)
        b = res.get("best")
        print(f"[{i:>2}/{len(counties)}] {c:<14} {res['status']:<18} "
              f"{('%.4f m/px' % b['gsd_m']) if b else '':<12} "
              f"{(str(b.get('year')) + '  ' + b['title'][:34]) if b else res.get('note', '')}",
              flush=True)
        json.dump(results, open(a.out, "w"), indent=2)

    by = {}
    for r in results:
        by.setdefault(r["status"], []).append(r["county"])
    print("\n=== COVERAGE ===")
    for k in ("ADVERTISED", "ADVERTISED-COARSE", "HISTORIC-ONLY",
              "UNATTRIBUTED", "FOUND-BUT-UNUSABLE", "NONE"):
        if k in by:
            print(f"{k:<20} {len(by[k]):>3}  {', '.join(by[k])}")
    adv = len(by.get("ADVERTISED", []))
    print(f"\n{adv}/{len(counties)} counties have an advertised <=30 cm "
          f"CURRENT (>={MIN_YEAR}) county-wide "
          f"exportImage endpoint. ADVERTISED is NOT verified — no pixels were "
          f"fetched. Run the chip verification before trusting any of these.")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
