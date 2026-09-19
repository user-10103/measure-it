"""High-resolution county GIS aerial fetch (3-inch / 7.6 cm) by address or lat/lon.

Florida county orthoimagery is a PUBLIC RECORD — free, and commercial use is
permitted (counties ask only for attribution "as a professional courtesy").
This is the per-address imagery source for the address -> Roof Report pipeline,
replacing 0.6 m NAIP with ~7.6 cm county imagery (8x finer linear resolution).

Verified 2026-06-08: Pinellas `Aerials2024` ImageServer reports
pixelSize = 0.0762 m (exactly 3 inches); a live `exportImage` fetch by lat/lon
returned a real 660x660 tile with cars/lane-stripes resolved, no auth, no key.

Reachability note: Pinellas (egis.pinellas.gov) responds fine. Hillsborough
(maps.hillsboroughcounty.org) and Pasco (pascogis.pascocountyfl.net) refused
connections (HTTP 000) from our test network — those servers appear to block
some datacenter IP ranges. Run those two from a different egress (e.g. Colab,
which sits on Google's network) or a residential IP. Endpoints below are
correct and documented regardless of our network's reach.
"""
import io
import json
import math
import urllib.parse
import urllib.request

_CENSUS = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"
_UA = {"User-Agent": "Mozilla/5.0 (measure-it county-imagery fetch)"}

# county BASENAME -> {url: ImageServer, gsd_m: native ground sample distance,
#                     year: capture year, reachable: from a generic datacenter IP}
COUNTY_ENDPOINTS = {
    "Pinellas": {
        "url": "https://egis.pinellas.gov/gis/rest/services/Aerials/Aerials2024/ImageServer",
        "gsd_m": 0.0762, "year": 2024, "reachable": True,
    },
    "Hillsborough": {
        "url": "https://maps.hillsboroughcounty.org/arcgis/rest/services/"
               "AerialsNew/Aerials2025_3_inch_MrSid/ImageServer",
        "gsd_m": 0.0762, "year": 2025, "reachable": False,  # blocks some IPs
    },
    "Pasco": {
        "url": "https://pascogis.pascocountyfl.net/giswebi/rest/services/"
               "Aerials/Aerials2023/ImageServer",
        "gsd_m": 0.0762, "year": 2023, "reachable": False,  # blocks some IPs
    },
    # Found via the ArcGIS Online public search API and verified live
    # 2026-09-14: ?f=json returns the service (pixelSizeX below) and an
    # anonymous exportImage returns real pixels. No token, no account.
    "Sarasota": {
        "url": "https://ags3.scgov.net/agsimg/rest/services/Imagery2020s/"
               "SC2026WM/ImageServer",
        "gsd_m": 0.0762, "year": 2026, "reachable": True,
    },
    "Broward": {
        "url": "https://bcgishub.broward.org/image/rest/services/Imagery/"
               "aerialscurrent/ImageServer",
        "gsd_m": 0.1524, "year": 2025, "reachable": True,
    },
    # Lee lived only in adresses/pipeline/utils/county_imagery.py — a SECOND
    # copy of this registry that the pipeline does not import. So the finest
    # imagery found in Florida (0.25 ft = 7.62 cm, hand-verified 2026-08-24 on
    # a 1050x1050 Cape Coral fetch: solar panels countable, vents visible) was
    # unreachable from the code that makes reports, and every Lee address was
    # measured on 0.6 m NAIP. Re-verified live 2026-09-19: pixelSizeX 0.25,
    # wkid 6443 (NAD83(2011) FL West ftUS), caps Image,Metadata,Catalog.
    # NOTE the unit: 0.25 read as metres would call this 25 cm imagery.
    "Lee": {
        "url": "https://gisimageserver.leegov.com/imageserver/rest/services/"
               "Aerials/Aerials2025/ImageServer",
        "urls": [
            "https://gisimageserver.leegov.com/imageserver/rest/services/"
            "Aerials/Aerials2025/ImageServer",
            "https://gisimageserver.leegov.com/imageserver/rest/services/"
            "Aerials/Aerials2024/ImageServer",
        ],
        "gsd_m": 0.0762, "year": 2025, "reachable": True,
        "maxImageWidth": 15000, "mosaic": False,
    },
}

# HOW TO GROW THIS REGISTRY (no token, no account):
#
#   curl -sG --data-urlencode 'q=<county> imagery type:"Image Service" access:public' \
#        --data 'f=json&num=20' https://www.arcgis.com/sharing/rest/search
#
# then for each candidate url, GET "<url>?f=json" and read pixelSizeX for the
# native GSD, and confirm an anonymous exportImage returns image bytes. Counties
# publish these themselves, so coverage grows one verified endpoint at a time.
#
# Verified failures are worth recording too, so they are not re-probed:
#   ca.dep.state.fl.us (FCDOP statewide) -- 499 Token Required on everything
#   gis.brevardfl.gov                    -- does not resolve
#   maps.brevardfl.gov                   -- Cloudflare challenge (403)
#
# NOTE these servers commonly answer with JPEG, not PNG. gis_chip._export_image
# must keep accepting jpgpng: a PNG-only magic-number check silently discards
# working 3-inch imagery.

# Statewide fallback: FL County Digital Orthoimagery Program (FCDOP), 6-inch
# (0.15 m), 3-year cycle. It was recorded here as reachable and public. It is
# NOT: the service answers every request, including a bare ?f=json, with
#     {"error":{"code":499,"message":"Token Required"}}
# Verified 2026-09-14 against both the ImageServer and its parent folder. That
# is why 1250 Pineapple Ave kept measuring off 30 cm NAIP -- the resolver was
# correctly falling through to a statewide set that has never been able to
# answer. Needs an ArcGIS token; skipped entirely unless one is configured,
# rather than spending a request and a timeout on a guaranteed 499.
FCDOP_FALLBACK = {
    "url": "https://ca.dep.state.fl.us/arcgis/rest/services/Imagery/"
           "Aerial_Imagery_2019/ImageServer",
    "gsd_m": 0.15, "year": 2019, "reachable": False,
    "requires_token": True, "token_env": "FDEP_ARCGIS_TOKEN",
}


def geocode(address, timeout=30):
    """Address -> (lon, lat, county_basename) via the free US Census geocoder."""
    q = urllib.parse.urlencode({
        "address": address, "benchmark": "Public_AR_Current",
        "vintage": "Current_Current", "format": "json"})
    req = urllib.request.Request(f"{_CENSUS}?{q}", headers=_UA)
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    matches = d["result"]["addressMatches"]
    if not matches:
        raise ValueError(f"Census geocoder: no match for {address!r}")
    a = matches[0]
    lon, lat = a["coordinates"]["x"], a["coordinates"]["y"]
    counties = a["geographies"].get("Counties", [])
    county = counties[0]["BASENAME"] if counties else None
    return lon, lat, county


def endpoint_for(county):
    """Pick the best ImageServer for a county (3-inch if known, else FCDOP)."""
    return COUNTY_ENDPOINTS.get(county, FCDOP_FALLBACK)


def fetch_chip(lon, lat, county=None, size_m=60.0, out_gsd_m=None, timeout=60):
    """Fetch a square aerial chip centred on (lon, lat) from the county ImageServer.

    Returns (png_bytes, pixel_to_world, meta). `pixel_to_world(x, y)` maps chip
    pixels -> ground metres (origin top-left, +x right, +y down) for measure-it's
    process_chip_rgb. `out_gsd_m` defaults to the source's native GSD.
    """
    ep = endpoint_for(county)
    gsd = out_gsd_m or ep["gsd_m"]
    half = size_m / 2.0
    dlat = half / 111320.0
    dlon = half / (111320.0 * math.cos(math.radians(lat)))
    bbox = f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}"
    w = h = max(8, round(size_m / gsd))
    q = urllib.parse.urlencode({
        "bbox": bbox, "bboxSR": 4326, "size": f"{w},{h}",
        "imageSR": 3857, "format": "png", "f": "image"})
    url = f"{ep['url']}/exportImage?{q}"
    req = urllib.request.Request(url, headers=_UA)
    png = urllib.request.urlopen(req, timeout=timeout).read()
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        # ArcGIS returns a JSON error body (not a PNG) on bad requests
        raise RuntimeError(f"non-PNG response from {ep['url']}: {png[:200]!r}")

    def pixel_to_world(x, y):
        return (x * gsd, y * gsd)

    meta = {"gsd_m": gsd, "width": w, "height": h, "county": county,
            "source": ep["url"], "year": ep["year"], "bbox_4326": bbox}
    return png, pixel_to_world, meta


def fetch_for_address(address, **kw):
    """End-to-end: address -> (png_bytes, pixel_to_world, meta) at 3-inch."""
    lon, lat, county = geocode(address)
    png, p2w, meta = fetch_chip(lon, lat, county=county, **kw)
    meta.update({"address": address, "lon": lon, "lat": lat})
    return png, p2w, meta
