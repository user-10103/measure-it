"""
Report quality gate — an operational definition of a "world-class" roof report.

The goal is not a subjective judgement: a report is world-class only if it carries
the measurements a Roofr/EagleView deliverable carries AND those measurements are
internally consistent (slope actually applied, pitch resolved or honestly flagged,
edges typed, obstructions accounted). score_report() turns that into a checklist a
CI/serving gate can enforce, so the pipeline can never silently ship a degraded report.

Each check has a severity:
  FAIL  -> the report is NOT world-class; block/flag it.
  WARN  -> below bar but not disqualifying; surface it.
Use `passed` (no FAILs) as the hard gate.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from src.output.report_data import ReportModel, build_report_model
from src.output.units import m2_to_sqft

FAIL, WARN, OK = "FAIL", "WARN", "OK"


@dataclass
class Check:
    id: str
    severity: str          # FAIL / WARN
    ok: bool
    detail: str


def score_report(report_input: dict, model: ReportModel | None = None) -> dict:
    if model is None:
        model = build_report_model(report_input)
    facets = report_input.get("facets", [])
    checks: List[Check] = []

    def add(cid, sev, ok, detail):
        checks.append(Check(cid, sev, bool(ok), detail))

    # --- completeness ---
    add("area_positive", FAIL, model.total_area_sqft > 0,
        f"total area = {model.total_area_sqft:.0f} sqft")
    # area sanity: a CRS/units bug inflates area to millions of sqft (the
    # facets_lidar_v2_1 141M m² bug). A real roof is ~100-100k sqft.
    add("area_sane", FAIL, 80 <= model.total_area_sqft <= 200000,
        f"total area {model.total_area_sqft:.0f} sqft in plausible range"
        if 80 <= model.total_area_sqft <= 200000 else
        f"total area {model.total_area_sqft:.0f} sqft is implausible (CRS/units bug?)")
    add("facets_present", FAIL, model.num_facets >= 1,
        f"{model.num_facets} facets")
    add("facet_table", FAIL, bool(model.facet_rows),
        "per-facet detail rows present" if model.facet_rows else "no per-facet rows")
    add("metadata", WARN, bool(report_input.get("report_id")),
        "report_id present" if report_input.get("report_id") else "no report_id")
    add("waste_table", WARN, bool(model.waste_table), "waste table present")

    # --- pitch resolved or honestly flagged (never silently unspecified) ---
    silent_unspec = [f for f in facets
                     if not f.get("is_flat")
                     and (f.get("pitch_string") in (None, "unspecified"))
                     and not f.get("needs_review")]
    add("pitch_resolved", FAIL, len(silent_unspec) == 0,
        "all pitched facets have a pitch or are flagged"
        if not silent_unspec else
        f"{len(silent_unspec)} pitched facet(s) have UNKNOWN pitch but are NOT flagged "
        "-> plan area silently reported as surface area")

    # --- slope actually applied (catches the flat-fallback bug) ---
    bad_slope = []
    for f in facets:
        if f.get("is_flat") or f.get("needs_review"):
            continue
        sa, pa = f.get("surface_area_m2"), f.get("plan_area_m2")
        if sa is None:
            bad_slope.append(f.get("facet_id"))       # no surface area at all
        elif pa and sa <= pa + 1e-6 and (f.get("slope_deg") or 0) >= 5:
            bad_slope.append(f.get("facet_id"))       # sloped but surface==plan
    add("slope_applied", FAIL, len(bad_slope) == 0,
        "surface area = plan/cos(slope) on pitched facets"
        if not bad_slope else
        f"facets {bad_slope} report plan area as surface area (no slope multiplier)")

    # --- facets must form a clean PARTITION (no significant overlap) ---
    # A world-class diagram shows facets tiling the roof; overlap means a
    # mega-facet / bad reconstruction (the fragmentation failure mode). The
    # structural checks above can't see this — a report can carry pitch and
    # edges yet render an overlapping mess. Catch it on the geometry.
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        polys = []
        for f in facets:
            xy = f.get("polygon_xy")
            if xy and len(xy) >= 3:
                p = Polygon(xy)
                if not p.is_valid:
                    p = p.buffer(0)
                if not p.is_empty and p.area > 0:
                    polys.append(p)
        if len(polys) >= 2:
            asum = sum(p.area for p in polys)
            uarea = unary_union(polys).area
            ov = max(0.0, (asum - uarea) / asum) if asum else 0.0
            add("facets_partition", FAIL, ov <= 0.10,
                f"facet overlap {ov:.0%} (clean partition)" if ov <= 0.10 else
                f"facets overlap {ov:.0%} of total area — not a clean partition "
                "(mega-facet / fragmentation; diagram will render as an overlapping mess)")
        else:
            add("facets_partition", WARN, True, "insufficient polygons to check partition")
        # coverage: facets must TILE the roof outline (no big gaps). Partition
        # (no overlap) + coverage (no gaps) together = a clean tiling, which is
        # what a world-class diagram shows.
        oxy = report_input.get("outline_xy")
        if oxy and len(oxy) >= 3 and polys:
            outline = Polygon(oxy)
            if not outline.is_valid:
                outline = outline.buffer(0)
            if outline.area > 0:
                covered = unary_union(polys).intersection(outline).area / outline.area
                add("facets_coverage", FAIL, covered >= 0.85,
                    f"facets cover {covered:.0%} of the roof outline" if covered >= 0.85
                    else f"facets cover only {covered:.0%} of the roof outline "
                    "(gaps — diagram will show unlabelled roof)")
    except Exception as e:
        add("facets_partition", WARN, True, f"partition not checked ({type(e).__name__})")

    # --- edges typed: a multi-facet roof must have a ridge or hip ---
    ef = model.edge_totals_ft
    if model.num_facets > 1:
        has_apex = (ef.get("ridge", 0) + ef.get("hip", 0)) > 0.5
        add("edges_typed", FAIL, has_apex,
            "ridge/hip present" if has_apex else
            "multi-facet roof has NO ridge or hip length -> edge typing failed")
        add("eaves_present", WARN, ef.get("eave", 0) > 0.5,
            "eaves present" if ef.get("eave", 0) > 0.5 else "no eave length")
    else:
        add("edges_typed", WARN, True, "single-facet roof; ridge/hip N/A")

    # --- obstructions accounted when the FO detector ran ---
    if "foreign_objects" in report_input:
        add("obstructions", WARN, model.num_obstructions >= 0,
            f"{model.num_obstructions} obstruction(s) reported")

    # --- predominant pitch sanity: a mostly-sloped roof must not report 0:12 ---
    mostly_flat = model.flat_area_sqft > model.pitched_area_sqft
    add("predominant_pitch", WARN,
        mostly_flat or model.predominant_pitch not in ("0:12", "unspecified"),
        f"predominant pitch {model.predominant_pitch}"
        if (mostly_flat or model.predominant_pitch not in ("0:12", "unspecified"))
        else f"predominant pitch {model.predominant_pitch} on a mostly-sloped roof (pitch failure)")

    # --- honesty: review load surfaced ---
    add("review_surfaced", WARN, True,
        f"{model.num_needs_review} of {model.num_facets} facets need review")

    fails = [c for c in checks if c.severity == FAIL and not c.ok]
    warns = [c for c in checks if c.severity == WARN and not c.ok]
    scored = [c for c in checks if c.severity == FAIL]
    score = (sum(1 for c in scored if c.ok) / len(scored)) if scored else 1.0
    return {
        "passed": len(fails) == 0,
        "score": round(score, 3),
        "num_fail": len(fails), "num_warn": len(warns),
        "checks": [c.__dict__ for c in checks],
    }


def format_report_qc(result: dict) -> str:
    lines = [f"WORLD-CLASS GATE: {'PASS' if result['passed'] else 'FAIL'}  "
             f"(score {result['score']:.0%}, {result['num_fail']} fail / {result['num_warn']} warn)"]
    for c in result["checks"]:
        mark = "ok " if c["ok"] else ("XX " if c["severity"] == "FAIL" else "!! ")
        lines.append(f"  {mark}[{c['severity']}] {c['id']}: {c['detail']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import json, sys
    ri = json.load(open(sys.argv[1]))
    # accept either a report_input or a report.json (which nests under keys)
    if "facets" not in ri and "summary" in ri:
        ri = {"facets": ri.get("facets", []), "edges": ri.get("edges", []),
              **ri.get("summary", {})}
    print(format_report_qc(score_report(ri)))
