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
