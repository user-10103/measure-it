"""World-class report gate + honesty-flag behaviour."""
from src.output.report_qc import score_report
from src.output.report_data import build_report_model


def _good():
    return {"address": "A", "report_id": "MI-1", "report_date": "2026-08-25",
            "facets": [
                # non-overlapping halves = a clean partition
                {"facet_id": 0, "surface_area_m2": 80, "plan_area_m2": 74.5,
                 "pitch_string": "6:12", "slope_deg": 26.6, "aspect_bin": "N",
                 "is_flat": False, "needs_review": False,
                 "polygon_xy": [[0, 0], [10, 0], [10, 3], [0, 3]]},
                {"facet_id": 1, "surface_area_m2": 80, "plan_area_m2": 74.5,
                 "pitch_string": "6:12", "slope_deg": 26.6, "aspect_bin": "S",
                 "is_flat": False, "needs_review": False,
                 "polygon_xy": [[0, 3], [10, 3], [10, 6], [0, 6]]}],
            "outline_xy": [[0, 0], [10, 0], [10, 6], [0, 6]],
            "edges": [{"edge_type": "ridge", "length_m": 10, "geometry_xy": [[0, 5], [10, 5]]},
                      {"edge_type": "eave", "length_m": 16, "geometry_xy": [[0, 0], [16, 0]]}]}


def _degraded():
    return {"address": "B",
            "facets": [
                {"facet_id": 0, "surface_area_m2": None, "plan_area_m2": 74.5,
                 "pitch_string": None, "slope_deg": None, "aspect_bin": None,
                 "is_flat": False, "needs_review": False,
                 "polygon_xy": [[0, 0], [10, 0], [10, 6], [0, 6]]}],
            "edges": [{"edge_type": "eave", "length_m": 16, "geometry_xy": [[0, 0], [16, 0]]}]}


def test_flat_roof_edges_typed_warns_not_fails():
    # a predominantly-flat commercial roof (two big flat facets, no ridge/hip):
    # edges_typed must WARN, not FAIL — an honest flat roof isn't blocked by the
    # gate for a ridge line that physically isn't there.
    ri = {"address": "C", "report_id": "MI-2",
          "facets": [
              {"facet_id": 0, "surface_area_m2": 300, "plan_area_m2": 300,
               "pitch_string": "0:12", "slope_deg": 0.0, "aspect_bin": None,
               "is_flat": True, "needs_review": False,
               "polygon_xy": [[0, 0], [30, 0], [30, 15], [0, 15]]},
              {"facet_id": 1, "surface_area_m2": 300, "plan_area_m2": 300,
               "pitch_string": "0:12", "slope_deg": 0.0, "aspect_bin": None,
               "is_flat": True, "needs_review": False,
               "polygon_xy": [[0, 15], [30, 15], [30, 30], [0, 30]]}],
          "outline_xy": [[0, 0], [30, 0], [30, 30], [0, 30]],
          "edges": [{"edge_type": "eave", "length_m": 40, "geometry_xy": [[0, 0], [40, 0]]}]}
    checks = {c["id"]: c for c in score_report(ri)["checks"]}
    assert checks["edges_typed"]["severity"] == "WARN"
    assert checks["edges_typed"]["ok"] is True


def test_world_class_passes():
    r = score_report(_good())
    assert r["passed"], r
    assert r["score"] == 1.0


def test_degraded_fails_on_slope_and_pitch():
    r = score_report(_degraded())
    assert not r["passed"]
    failed = {c["id"] for c in r["checks"] if c["severity"] == "FAIL" and not c["ok"]}
    assert "slope_applied" in failed


def test_honesty_flag_marks_unmeasured_pitch():
    # build_report_model must flag a non-flat, unmeasured-pitch facet for review
    m = build_report_model(_degraded())
    assert m.num_needs_review == 1
    assert m.facet_rows[0]["needs_review"] is True


def test_flagged_unmeasured_passes_pitch_resolved():
    # once flagged, the honesty check no longer treats it as a silent number
    ri = _degraded()
    ri["facets"][0]["needs_review"] = True
    r = score_report(ri)
    pitch = next(c for c in r["checks"] if c["id"] == "pitch_resolved")
    assert pitch["ok"], pitch


def test_overlapping_facets_fail_partition():
    # two facets covering the same square = 100% overlap -> not world-class
    sq = [[0, 0], [4, 0], [4, 4], [0, 4]]
    ri = {"report_id": "X", "facets": [
        {"facet_id": 0, "surface_area_m2": 16, "plan_area_m2": 14, "pitch_string": "6:12",
         "slope_deg": 26.6, "aspect_bin": "N", "is_flat": False, "needs_review": False, "polygon_xy": sq},
        {"facet_id": 1, "surface_area_m2": 16, "plan_area_m2": 14, "pitch_string": "6:12",
         "slope_deg": 26.6, "aspect_bin": "S", "is_flat": False, "needs_review": False, "polygon_xy": sq}],
        "edges": [{"edge_type": "ridge", "length_m": 4, "geometry_xy": [[0, 2], [4, 2]]}]}
    r = score_report(ri)
    failed = {c["id"] for c in r["checks"] if c["severity"] == "FAIL" and not c["ok"]}
    assert "facets_partition" in failed
    assert not r["passed"]


def test_area_sane_rejects_crs_bug():
    ri = _good()
    ri["facets"][0]["surface_area_m2"] = 5_000_000     # CRS/units bug
    r = score_report(ri)
    failed = {c["id"] for c in r["checks"] if c["severity"] == "FAIL" and not c["ok"]}
    assert "area_sane" in failed


def test_coverage_fails_on_gappy_facets():
    ri = _good()
    # shrink facets so they cover far less than the outline -> gap
    ri["facets"] = [ri["facets"][0]]
    ri["facets"][0]["polygon_xy"] = [[0, 0], [2, 0], [2, 1], [0, 1]]
    r = score_report(ri)
    cov = next(c for c in r["checks"] if c["id"] == "facets_coverage")
    assert not cov["ok"]


def test_failing_gate_yields_plain_english_incomplete_reason():
    """A report that fails the gate must be STAMPED, not shipped looking finished.
    report_service scores the gate BEFORE writing the PDF and sets this reason,
    which pdf_report renders as the red INCOMPLETE banner on the cover."""
    from src.serve.report_service import incomplete_reason
    # 1250 Pineapple Ave signature: many facets, zero ridges/hips (impossible)
    ri = {"address": "bad", "report_id": "MI-BAD",
          "outline_xy": [[0, 0], [10, 0], [10, 10], [0, 10]],
          "facets": [{"facet_id": i + 1,
                      "polygon_xy": [[i, 0], [i + 1, 0], [i + 1, 10], [i, 10]],
                      "plan_area_m2": 10.0, "surface_area_m2": 10.8,
                      "slope_deg": 26.6, "pitch_string": "6:12",
                      "aspect_bin": "N", "is_flat": False, "needs_review": False}
                     for i in range(10)],
          "edges": [{"edge_type": "eave", "length_m": 10,
                     "geometry_xy": [[0, 0], [10, 0]]}]}
    qc = score_report(ri)
    assert not qc["passed"]
    reason = incomplete_reason(qc)
    assert reason and "edges_typed" not in reason      # jargon must not reach a client
    assert "roof edge structure not resolved" in reason

    # a passing report is never stamped
    assert incomplete_reason(score_report(_good())) is None


def test_undersegmented_roof_fails_the_gate():
    """One big facet passes partition/coverage/edges_typed trivially. When LiDAR
    says that facet spans more than one plane, the gate must FAIL — an
    under-segmented roof under-reports surface area, and material is ordered off
    that number."""
    ri = _good()
    ri["multiplane_facets"] = [0]
    r = score_report(ri)
    failed = {c["id"] for c in r["checks"] if c["severity"] == "FAIL" and not c["ok"]}
    assert "facets_vs_lidar_planes" in failed
    assert not r["passed"]


def test_lidar_agreement_passes_and_is_absent_without_lidar():
    ri = _good()
    ri["multiplane_facets"] = []            # LiDAR ran, agrees
    assert score_report(ri)["passed"]
    ids = {c["id"] for c in score_report(_good())["checks"]}
    assert "facets_vs_lidar_planes" not in ids   # no LiDAR -> not vacuously passed


def test_facet_whose_plane_explains_half_its_points_fails():
    """2725 Judge Fran: 8 SAM facets merged into 1, the resulting plane explaining
    49% of 121,189 points — ~62,000 returns more than 0.25 m off it — and every
    existing guard passed it. residual_median even looked excellent, because it is
    measured over inliers only: the worse the fit, the better that number reads."""
    ri = _good()
    ri["facets"][0]["explained_frac"] = 0.49
    ri["facets"][1]["explained_frac"] = 0.95
    r = score_report(ri)
    failed = {c["id"] for c in r["checks"] if c["severity"] == "FAIL" and not c["ok"]}
    assert "facet_plane_fit" in failed
    assert not r["passed"]
    detail = next(c["detail"] for c in r["checks"] if c["id"] == "facet_plane_fit")
    assert "49%" in detail


def test_well_fitted_facets_pass_and_check_is_absent_without_lidar():
    ri = _good()
    for f, frac in zip(ri["facets"], (0.77, 0.97)):   # Sarno / Eau Gallie range
        f["explained_frac"] = frac
    assert score_report(ri)["passed"]
    ids = {c["id"] for c in score_report(_good())["checks"]}
    assert "facet_plane_fit" not in ids      # no LiDAR -> no vacuous pass
