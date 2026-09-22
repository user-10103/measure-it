"""
Audit facet-labeling convention & granularity across two COCO sets.

Diagnoses whether a train/eval facet-count gap (ep6: train 8.55 vs eval 4.69 =
1.82x, which reads downstream as ~1.5x "over-segmentation") is really the model's
fault or a LABEL problem — and which kind:

  (a) LABELING CONVENTION — the same roofs are simply cut finer in one set
      (facets are substantial, just more of them). For an EagleView-grade target
      the finer set is likely the *correct* one; the coarser set is under-labeled.
  (b) SELECTION BIAS — a readiness `--keep` filter dropped simple/low-facet roofs
      from train, biasing it toward complex roofs, so the model learned to always
      cut finely. The gap is then a sampling artifact, not a convention.
  (c) GENUINE OVER-LABELING — one set is full of sliver facets (tiny fragments of
      one plane). That's a real labeling defect to clean up.

The discriminators:
  * category NAMES/ids per set (a schema mismatch = different provenance),
  * facets/roof distribution (mean/median/histogram),
  * per-facet AREA vs its roof (sliver fraction — the (c) tell),
  * address_id OVERLAP: if the same roof appears in both sets, compare its facet
    count head-to-head — that separates (a) from (b) cleanly.

Pure geometry from COCO `area`/`bbox`; no GPU, no shapely.

Usage:
  python -m training.label_convention_audit TRAIN.coco.json EVAL.coco.json \
      [--labels train eval] [--sliver-frac 0.02]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

FACET_NAMES = {"facet", "roof_facet", "roof facet", "roof face"}
ROOF_NAMES = {"roof_polygon", "roof_outline", "roof", "outline"}


def _pcts(vals: List[float], qs=(5, 10, 50, 90, 95)) -> Dict[str, float]:
    if not vals:
        return {f"p{q}": 0.0 for q in qs}
    a = np.asarray(vals, dtype=float)
    return {f"p{q}": round(float(np.percentile(a, q)), 2) for q in qs}


def _resolve(coco: dict):
    facet_ids, roof_ids, names = set(), set(), {}
    for c in coco.get("categories", []):
        nm = str(c.get("name", "")).lower()
        names[c["id"]] = c.get("name", "")
        if nm in FACET_NAMES:
            facet_ids.add(c["id"])
        elif nm in ROOF_NAMES:
            roof_ids.add(c["id"])
    return facet_ids, roof_ids, names


def audit_one(coco: dict, sliver_frac: float) -> dict:
    facet_ids, roof_ids, names = _resolve(coco)
    if not facet_ids:                                  # fall back: everything is a facet
        facet_ids = {c["id"] for c in coco.get("categories", [])}
    fac_by_img = defaultdict(list)                     # image_id -> [facet area px^2]
    roof_area = {}                                     # image_id -> roof_polygon area
    for a in coco.get("annotations", []):
        if a["category_id"] in facet_ids:
            fac_by_img[a["image_id"]].append(float(a.get("area") or 0.0))
        elif a["category_id"] in roof_ids:
            roof_area[a["image_id"]] = roof_area.get(a["image_id"], 0.0) + float(a.get("area") or 0.0)

    imgs = coco.get("images", [])
    counts, facet_frac, sliver_hits = [], [], 0
    n_facets_total = 0
    for im in imgs:
        fac = fac_by_img.get(im["id"], [])
        counts.append(len(fac))
        n_facets_total += len(fac)
        denom = roof_area.get(im["id"]) or sum(fac) or 1.0
        for ar in fac:
            fr = ar / denom
            facet_frac.append(fr)
            if fr < sliver_frac:
                sliver_hits += 1

    counts_nonzero = [c for c in counts if c > 0]
    hist = Counter(min(c, 20) for c in counts_nonzero)   # cap bucket at 20+
    return {
        "categories": {k: names[k] for k in sorted(names)},
        "images": len(imgs),
        "facet_bearing_images": len(counts_nonzero),
        "facets_total": n_facets_total,
        "facets_per_roof_mean": round(float(np.mean(counts_nonzero)), 2) if counts_nonzero else 0.0,
        "facets_per_roof_median": round(float(np.median(counts_nonzero)), 1) if counts_nonzero else 0.0,
        "facets_per_roof_pcts": _pcts(counts_nonzero, (10, 50, 90)),
        "facets_per_roof_max": max(counts_nonzero) if counts_nonzero else 0,
        "facet_frac_of_roof_pcts": _pcts([100 * f for f in facet_frac]),   # % of roof
        "sliver_facets_pct": round(100 * sliver_hits / max(n_facets_total, 1), 1),
        "count_hist": dict(sorted(hist.items())),
        "_counts_by_addr": None,   # filled by caller when address_id present
        "_addr_counts": None,
    }


def _addr_counts(coco: dict) -> Dict[str, int]:
    facet_ids, _, _ = _resolve(coco)
    if not facet_ids:
        facet_ids = {c["id"] for c in coco.get("categories", [])}
    id2addr = {im["id"]: (im.get("address_id") or im.get("file_name", "").rsplit("/", 1)[-1].split(".")[0])
               for im in coco.get("images", [])}
    per = defaultdict(int)
    seen = {id2addr[i] for i in id2addr}
    for a in coco.get("annotations", []):
        if a["category_id"] in facet_ids:
            per[id2addr.get(a["image_id"])] += 1
    for s in seen:
        per.setdefault(s, 0)
    return dict(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coco", nargs=2, help="TRAIN and EVAL coco json paths")
    ap.add_argument("--labels", nargs=2, default=["train", "eval"])
    ap.add_argument("--sliver-frac", type=float, default=0.02,
                    help="a facet below this fraction of its roof is a sliver (default 2%%)")
    ns = ap.parse_args()

    cocos = [json.load(open(p)) for p in ns.coco]
    reports = [audit_one(c, ns.sliver_frac) for c in cocos]

    for lab, rep in zip(ns.labels, reports):
        print(f"\n===== {lab} =====")
        print(f"categories        : {rep['categories']}")
        print(f"images            : {rep['images']}  (facet-bearing {rep['facet_bearing_images']})")
        print(f"facets total      : {rep['facets_total']}")
        print(f"facets/roof       : mean {rep['facets_per_roof_mean']}  median "
              f"{rep['facets_per_roof_median']}  p10/p90 "
              f"{rep['facets_per_roof_pcts']['p10']}/{rep['facets_per_roof_pcts']['p90']}  "
              f"max {rep['facets_per_roof_max']}")
        print(f"facet % of roof   : p5 {rep['facet_frac_of_roof_pcts']['p5']}%  "
              f"p50 {rep['facet_frac_of_roof_pcts']['p50']}%  "
              f"p95 {rep['facet_frac_of_roof_pcts']['p95']}%")
        print(f"SLIVER facets     : {rep['sliver_facets_pct']}%  (<{ns.sliver_frac:.0%} of roof)")
        print(f"count histogram   : {rep['count_hist']}")

    # ---- cross-comparison + verdict --------------------------------------
    tr, ev = reports
    ratio = tr["facets_per_roof_mean"] / max(ev["facets_per_roof_mean"], 1e-9)
    tr_names = set(tr["categories"].values())
    ev_names = set(ev["categories"].values())
    print("\n===== CROSS-COMPARISON =====")
    print(f"category names    : {ns.labels[0]}={tr_names}  {ns.labels[1]}={ev_names}"
          f"{'   <-- SCHEMA MISMATCH' if tr_names != ev_names else ''}")
    print(f"granularity ratio : {ratio:.2f}x  ({ns.labels[0]} finer)")

    a_tr, a_ev = _addr_counts(cocos[0]), _addr_counts(cocos[1])
    shared = set(a_tr) & set(a_ev)
    print(f"roof overlap      : {len(shared)} roof(s) in BOTH sets")
    if shared:
        diffs = [a_tr[s] - a_ev[s] for s in shared]
        same = sum(1 for d in diffs if d == 0)
        print(f"  same roof, facet-count delta: mean {np.mean(diffs):+.2f}  "
              f"median {np.median(diffs):+.0f}  identical on {same}/{len(shared)}")
        print("  -> non-zero deltas on shared roofs = a real CONVENTION difference (a)")
    else:
        print("  -> disjoint sets: can't do head-to-head; rely on the distributions above.")
        print("     If train's extra facets are SUBSTANTIAL (low sliver%), the gap is")
        print("     convention/selection, not garbage — the finer set may be the correct one.")

    # ---- verdict: report what these two sets actually show -----------------
    print("\nVERDICT:")
    schema_ok = tr_names == ev_names
    leak = len(shared) > 0
    gap = abs(ratio - 1.0) > 0.15
    if not schema_ok:
        print("  ! SCHEMA MISMATCH — the sets use different category names/ids.")
    if leak:
        print(f"  ! LEAKAGE — {len(shared)} roof(s) in both sets; the eval is not held out.")
    if gap:
        print(f"  ! GRANULARITY GAP {ratio:.2f}x — one set is labelled finer than the other.")
        print("      low sliver% -> (a) convention or (b) selection bias, NOT model over-seg;"
              "\n      high sliver% -> (c) genuine over-labelling to clean up.")
    if not (schema_ok and not leak and not gap):
        print("  => an F1 measured across these two sets is NOT a clean model score.")
    else:
        print(f"  OK — same schema, 0 leakage, granularity {ratio:.2f}x. "
              "This pair is a valid benchmark.")


if __name__ == "__main__":
    main()
