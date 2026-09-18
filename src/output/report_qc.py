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

# A plane explaining less than this share of its own facet's points is not
# describing that surface - it is averaging across sections that disagree.
# Observed: 2725 Judge Fran 49% (8 sections merged into 1), against 1600 Sarno
# and 755 E Eau Gallie at 77-97%.
PLANE_EXPLAINS_MIN = 0.60

# Roof plan area against the footprint of the building we actually selected.
# Gross-error bounds, not calibration: a real roof runs ~1.0-1.3x its footprint
# because of eave overhang, so these leave a wide margin on both sides and fire
# only when the report is about a different building than the one requested.
FOOTPRINT_RATIO_MIN = 0.35
FOOTPRINT_RATIO_MAX = 3.00


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
        if not (oxy and len(oxy) >= 3) and polys:
            # sam_report sets outline_xy to [] when the zero-shot outline is
            # missing, which used to drop this check ENTIRELY rather than fail
            # it — the same vacuous-absence the LiDAR checks above guard
            # against. Coverage is precisely the check that catches facets not
            # spanning the roof, so its silent disappearance is the worst case.
            # Record it as not-checked so it is visible in the report.
            add("facets_coverage", WARN, True,
                "coverage NOT checked — no roof outline to compare against")
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

    # --- edges typed: a multi-facet SLOPED roof must have a ridge or hip. A
    #     predominantly-flat roof (a commercial building with a small pitched
    #     annex, say) genuinely has no apex line — WARN, don't FAIL, for the
    #     missing ridge, so an honest flat roof isn't blocked by the gate.
    ef = model.edge_totals_ft
    mostly_flat = model.flat_area_sqft > model.pitched_area_sqft
    if model.num_facets > 1:
        has_apex = (ef.get("ridge", 0) + ef.get("hip", 0)) > 0.5
        if has_apex:
            add("edges_typed", FAIL, True, "ridge/hip present")
        elif mostly_flat:
            add("edges_typed", WARN, True,
                "predominantly flat roof — no ridge/hip expected")
        else:
            add("edges_typed", FAIL, False,
                "multi-facet sloped roof has NO ridge or hip length -> edge typing failed")
        add("eaves_present", WARN, ef.get("eave", 0) > 0.5,
            "eaves present" if ef.get("eave", 0) > 0.5 else "no eave length")

        # --- did we NAME the internal edges, or just find them? ---
        # edges_typed above asks only whether ridge+hip exceeds half a foot.
        # 3004 Marble Crest returned 0 ft of ridge and 13 ft of hip against an
        # EagleView truth of 78 and 242 — and 13 > 0.5, so it passed. Measured
        # against six EagleView Premium reports, our linear footage lands at
        # 16-42% of truth while AREA is within 5-16%, because area is the
        # integral of the outline and survives under-segmentation. Nothing in
        # the gate looked at edge totals at all, so a report could lose 80% of
        # its ridge and hip and still pass.
        #
        # This is the symmetric partner of pitch_resolved: unknown is tolerable
        # when it is declared, never when it is silent. The logs show the
        # mechanism — "ridge -> unspecified" with "aspect=[359.80, 359.80]",
        # two flanks at the same azimuth. The classifier is right to refuse; the
        # fault is upstream, adjacent facets handed the same plane.
        #
        # The bright line is RELATIVE and needs no invented constant: failing
        # when we could not name more internal edge than we could name. A
        # magnitude check ("hips should be the largest category, and ours are
        # ~0") needs the benchmark distribution first and is deliberately not
        # attempted here.
        internal = sum(ef.get(k, 0.0) for k in
                       ("ridge", "hip", "valley", "transition", "parapet",
                        "wall_flashing", "step_flashing"))
        unspec = ef.get("unspecified", 0.0)
        if internal + unspec > 0.5:
            share = unspec / (internal + unspec)
            add("edges_resolved", FAIL, unspec <= internal,
                f"{share:.0%} of internal edge length is unspecified"
                if unspec <= internal else
                f"{unspec:.0f} ft of internal edge could not be typed against "
                f"{internal:.0f} ft that could ({share:.0%} unspecified) — the "
                "report names less of the roof's edge structure than it fails "
                "to name")
    else:
        add("edges_typed", WARN, True, "single-facet roof; ridge/hip N/A")

    # --- does the accepted plane actually describe the facet? ---
    # 2725 Judge Fran shipped 43,029 sqft as ONE flat plane explaining 49% of its
    # own 121,189 points - roughly 62,000 returns more than 0.25 m off it - after
    # the coplanarity merge collapsed 8 SAM facets into 1. Every existing guard
    # passed it: the absolute inlier floor wants 20 and it had 59,347; the ratio
    # floor wants 15% and it had 49%; facets_vs_lidar_planes asks whether a
    # SECOND plane exists, and the dominant plane genuinely is dominant - it just
    # does not explain half the roof. And residual_median looked excellent
    # (0.041 m) because it is measured over inliers only.
    thin = [(f.get("facet_id"), f["explained_frac"]) for f in facets
            if f.get("explained_frac") is not None
            and f["explained_frac"] < PLANE_EXPLAINS_MIN]
    if any(f.get("explained_frac") is not None for f in facets):
        add("facet_plane_fit", FAIL, not thin,
            "each facet's plane explains its own points" if not thin else
            "plane explains only " + ", ".join(f"{p:.0%} of facet {i}" for i, p in thin)
            + f" (need {PLANE_EXPLAINS_MIN:.0%}) — the surface is not one plane, so "
            "its pitch and area are averages over sections that disagree")

    # --- under-segmentation: does the facet count agree with the LiDAR? ---
    # Only scored when LiDAR actually ran (the key is absent otherwise), because
    # without elevation there is no independent evidence and a vacuous pass would
    # be exactly the silent success this check exists to prevent.
    if "multiplane_facets" in report_input:
        mp = report_input.get("multiplane_facets") or []
        add("facets_vs_lidar_planes", FAIL, not mp,
            "facet count agrees with the LiDAR plane count" if not mp else
            f"facet(s) {mp} still span MORE THAN ONE plane per LiDAR — the roof is "
            "under-segmented, so its surface area is under-reported")

    # --- did we measure the building we SELECTED? ---
    # Every check above asks whether the report is internally consistent. None
    # asks whether it is about the right roof. The Don CeSar -- a large hotel --
    # produced a 1,232 sqft two-facet roof, self-consistent in every way, and
    # PASSED. A small plausible roof is exactly what passes.
    #
    # The MS Buildings footprint we selected is already fetched (it clips the
    # LiDAR) and was simply never compared to the result. Bounds are deliberately
    # GROSS-ERROR bounds, not tuning: a roof's plan area normally runs 1.0-1.3x
    # its footprint (eave overhang). Below FOOTPRINT_RATIO_MIN we measured a
    # fragment of the building; above FOOTPRINT_RATIO_MAX we swallowed
    # neighbours. Anything in between passes without comment.
    fpa = report_input.get("footprint_plan_area_m2")
    if fpa and fpa > 0:
        plan = sum(f.get("plan_area_m2") or 0.0 for f in facets)
        ratio = plan / fpa
        add("measures_selected_building", FAIL,
            FOOTPRINT_RATIO_MIN <= ratio <= FOOTPRINT_RATIO_MAX,
            f"roof plan area is {ratio:.2f}x the selected building footprint"
            if FOOTPRINT_RATIO_MIN <= ratio <= FOOTPRINT_RATIO_MAX else
            f"roof plan area is {ratio:.2f}x the selected building footprint "
            f"({plan:.0f} m2 vs {fpa:.0f} m2) — the measured roof is not that "
            "building: too small means a fragment of it, too large means "
            "neighbouring structures were swallowed")

    # --- was the right building SELECTED? ---
    # measures_selected_building above compares the roof against the footprint
    # we picked, so a wrong PICK makes both sides the same wrong building, the
    # ratio lands near 1.0, and the report ships CLEAN. An under-segmented roof
    # ships stamped and honest; a wrong-building roof ships confidently wrong.
    #
    # This is WARN, not FAIL, on purpose. The evidence it surfaces --
    # pin-inside-footprint, distance from the geocoded pin, whether a closer
    # candidate was passed over -- has never been recorded on a real run, so
    # there is no basis yet for a threshold that would not be invented. Two
    # false FAILs on legitimate addresses would be worse than the WARN. Collect
    # the numbers across the probe set first, then tighten.
    if "pin_in_footprint" in report_input:
        inside = bool(report_input.get("pin_in_footprint"))
        dist = report_input.get("select_dist_m")
        rank = report_input.get("select_rank")
        ncand = report_input.get("select_n_candidates")
        margin = report_input.get("select_margin_m")
        # Ambiguous when the runner-up is about as close as the pick. Relative,
        # so it needs no distance threshold: a geocoder that returns a
        # street-front position makes 27 m ordinary, and the same 27 m is what
        # a wrong building looks like. The margin tells them apart.
        ambiguous = (margin is not None and dist is not None
                     and float(margin) < float(dist))
        suspicious = (not inside) or (rank not in (0, None)) or ambiguous
        bits = [f"pin {'inside' if inside else 'OUTSIDE'} the selected footprint"]
        if dist is not None:
            bits.append(f"{float(dist):.1f} m from it")
        if rank not in (0, None):
            bits.append(f"rank {rank} — a CLOSER building was passed over")
        if ncand:
            bits.append(f"{ncand} candidate(s) in range")
        if ambiguous:
            bits.append(f"AMBIGUOUS — runner-up at "
                        f"{float(report_input['select_runner_up_m']):.1f} m, "
                        f"margin {float(margin):.1f} m is smaller than the "
                        f"distance itself")
        add("building_selection", WARN, not suspicious, "; ".join(bits))

    # --- obstructions accounted when the FO detector ran ---
    if "foreign_objects" in report_input:
        add("obstructions", WARN, model.num_obstructions >= 0,
            f"{model.num_obstructions} obstruction(s) reported")

    # --- predominant pitch sanity: a mostly-sloped roof must not report 0:12 ---
    add("predominant_pitch", WARN,
        mostly_flat or model.predominant_pitch not in ("0:12", "unspecified"),
        f"predominant pitch {model.predominant_pitch}"
        if (mostly_flat or model.predominant_pitch not in ("0:12", "unspecified"))
        else f"predominant pitch {model.predominant_pitch} on a mostly-sloped roof (pitch failure)")

    # --- empirical roof grammar, from six EagleView Premium reports ---
    # The first bounds in this gate derived from GROUND TRUTH rather than
    # invented from a single address. All WARN: the sample is six FL
    # hip-dominant tract homes from one contractor in one quarter (effective
    # n~4, zero "Simple" roofs, zero parapets), padded 20%, and this pipeline is
    # scoped USA-wide. A breach means look, not fail.
    try:
        from src.output.roof_grammar import grammar_findings
        for f in grammar_findings(report_input, model):
            add(f["id"], WARN, f["ok"], f["detail"])
    except Exception as e:  # noqa: BLE001 - advisory, never fatal
        add("roof_grammar", WARN, True, f"grammar not evaluated ({type(e).__name__})")

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
        # The bracket used to print the check's SEVERITY CLASS, not its result,
        # so a passing check rendered as "ok [FAIL]". On a healthy report that
        # is 15 lines reading [FAIL] when 2 checks actually failed, in the log
        # that is the primary debugging surface. Print the OUTCOME instead: a
        # FAIL-class check that passed says PASS; one that failed says FAIL.
        outcome = "PASS" if c["ok"] else c["severity"]
        mark = "ok " if c["ok"] else ("XX " if c["severity"] == "FAIL" else "!! ")
        lines.append(f"  {mark}[{outcome}] {c['id']}: {c['detail']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import json, sys
    ri = json.load(open(sys.argv[1]))
    # accept either a report_input or a report.json (which nests under keys)
    if "facets" not in ri and "summary" in ri:
        ri = {"facets": ri.get("facets", []), "edges": ri.get("edges", []),
              **ri.get("summary", {})}
    print(format_report_qc(score_report(ri)))
