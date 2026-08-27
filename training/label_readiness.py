"""
Training-data readiness gate — the label-side analogue of report_qc.

"Train SAM3 on our annotations and get a world-class report" is only true if the
annotations are actually train-ready. This scores a COCO export per-roof and emits:
  - coverage: what fraction of roofs carry facet / edge / FO labels at all,
  - quality:  facet-label geometry pass rate (via label_quality.score_labels),
  - a KEEP manifest of the image_ids that should feed SAM3 facet training
    (clean facets only) — so the retrain never trains on empty or garbage labels.

Pure geometry, no GPU. Run:  python -m training.label_readiness <coco.json> [keep_out.json]
"""
from __future__ import annotations
import json, sys
from collections import defaultdict, Counter
from shapely.geometry import Polygon
from shapely.ops import unary_union
from training.label_quality import score_labels

# Resolve categories by NAME, not hardcoded id — a fresh Label Studio export
# renumbers ids and names the roof category `roof_outline` (not `roof_polygon`).
FACET_NAMES = {"facet", "roof_facet", "roof facet"}
ROOF_NAMES = {"roof_polygon", "roof_outline", "roof", "outline"}
EDGE_NAMES = {"edge_eave": "eave", "eave": "eave", "edge_rake": "rake", "rake": "rake",
              "edge_ridge": "ridge", "ridge": "ridge", "edge_valley": "valley",
              "valley": "valley", "edge_hip": "hip", "hip": "hip"}
FO_NAMES = {"chimney": "chimney", "skylight": "skylight", "solar_panel": "solar_panel",
            "ac_unit": "ac_unit", "vent": "vent", "satellite_dish": "satellite_dish",
            "tree_overhang": "tree_overhang", "other_object": "other_object"}


def _resolve_cats(coco: dict):
    """Map category ids -> role using the export's own names (id-agnostic)."""
    facet_ids, roof_ids, edge_map, fo_map = set(), set(), {}, {}
    for c in coco.get("categories", []):
        nm = str(c.get("name", "")).lower()
        cid = c["id"]
        if nm in FACET_NAMES:
            facet_ids.add(cid)
        elif nm in ROOF_NAMES:
            roof_ids.add(cid)
        elif nm in EDGE_NAMES:
            edge_map[cid] = EDGE_NAMES[nm]
        elif nm in FO_NAMES:
            fo_map[cid] = FO_NAMES[nm]
    return facet_ids, roof_ids, edge_map, fo_map


def _polys(seg):
    out = []
    if isinstance(seg, list):
        for ring in seg:
            if isinstance(ring, list) and len(ring) >= 6:
                try:
                    p = Polygon(list(zip(ring[0::2], ring[1::2])))
                    if not p.is_valid:
                        p = p.buffer(0)
                    if not p.is_empty:
                        out.append(p)
                except Exception:
                    pass
    return out


def assess(coco: dict) -> dict:
    facet_ids, roof_ids, edge_map, fo_map = _resolve_cats(coco)
    fname = {im["id"]: im.get("file_name", str(im["id"])) for im in coco["images"]}
    roofs = {im["id"]: {"roof": None, "facets": [], "edges": Counter(), "fo": Counter()}
             for im in coco["images"]}
    for a in coco["annotations"]:
        r = roofs.get(a["image_id"])
        if r is None:
            continue
        c = a["category_id"]
        if c in roof_ids:
            ps = _polys(a.get("segmentation"))
            if ps:
                r["roof"] = unary_union([r["roof"]] + ps) if r["roof"] else unary_union(ps)
        elif c in facet_ids:
            r["facets"].extend(_polys(a.get("segmentation")))
        elif c in edge_map:
            r["edges"][edge_map[c]] += 1
        elif c in fo_map:
            r["fo"][fo_map[c]] += 1

    total = with_facets = facet_pass = with_edges = with_ridge_or_hip = with_fo = 0
    keep, reasons = [], Counter()
    for iid, r in roofs.items():
        if r["roof"] is None:
            continue
        total += 1
        if r["facets"]:
            with_facets += 1
            q = score_labels(r["facets"], r["roof"])
            if q["passed"]:
                facet_pass += 1
                keep.append(fname[iid])
            else:
                for rs in q["reasons"]:
                    reasons[rs.split("=")[0]] += 1
        else:
            reasons["no_facets"] += 1
        if r["edges"]:
            with_edges += 1
            if r["edges"].get("ridge", 0) + r["edges"].get("hip", 0) > 0:
                with_ridge_or_hip += 1
        if r["fo"]:
            with_fo += 1

    pct = lambda x: round(100 * x / total, 1) if total else 0.0
    return {
        "roofs": total,
        "facet_coverage_pct": pct(with_facets),
        "facet_quality_pass_pct_of_labeled": round(100 * facet_pass / with_facets, 1) if with_facets else 0.0,
        "train_ready_facets": facet_pass,
        "edge_coverage_pct": pct(with_edges),
        "ridge_or_hip_coverage_pct": pct(with_ridge_or_hip),
        "fo_coverage_pct": pct(with_fo),
        "failure_reasons": dict(reasons.most_common()),
        "keep_ids": keep,
    }


if __name__ == "__main__":
    coco = json.load(open(sys.argv[1]))
    a = assess(coco)
    keep = a.pop("keep_ids")
    print(json.dumps(a, indent=2))
    print(f"\nTRAIN-READY (facet) roofs: {len(keep)} / {a['roofs']}")
    if len(sys.argv) > 2:
        json.dump({"file_names": keep}, open(sys.argv[2], "w"))
        print(f"wrote keep manifest -> {sys.argv[2]} ({len(keep)} file_names)")
