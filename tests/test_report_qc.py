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


def test_coverage_is_never_silently_absent_when_the_outline_is_missing():
    """sam_report sets outline_xy=[] when the zero-shot outline fails, which used
    to drop facets_coverage from the checklist entirely instead of failing it.
    A check that vanishes reads exactly like a check that passed — the pattern
    the LiDAR checks already guard against, and coverage is the one that catches
    facets not spanning the roof."""
    ri = _good()
    for missing in ([], None):
        r = dict(ri)
        if missing is None:
            r.pop("outline_xy", None)
        else:
            r["outline_xy"] = missing
        cov = [c for c in score_report(r)["checks"] if c["id"] == "facets_coverage"]
        assert cov, f"facets_coverage vanished for outline_xy={missing!r}"
        assert "NOT checked" in cov[0]["detail"]
        assert cov[0]["severity"] == "WARN"     # visible, but not a hard block


def test_coverage_still_fails_hard_when_an_outline_is_present():
    ri = _good()
    ri["facets"] = [ri["facets"][0]]
    ri["facets"][0]["polygon_xy"] = [[0, 0], [2, 0], [2, 1], [0, 1]]
    cov = next(c for c in score_report(ri)["checks"] if c["id"] == "facets_coverage")
    assert cov["severity"] == "FAIL" and not cov["ok"]


def test_flat_slope_threshold_has_exactly_one_definition():
    """It was declared in BOTH metrics.py and pitch_policy.py. Two copies of a
    classification threshold can drift, and then a roof is flat in one stage and
    pitched in the next. Identity, not equality: equality would still pass if
    someone re-declared the same number."""
    from src.roofs import metrics, pitch_policy
    assert pitch_policy.FLAT_SLOPE_DEG is metrics.FLAT_SLOPE_DEG
    src = open(pitch_policy.__file__).read()
    assert "FLAT_SLOPE_DEG = " not in src, "pitch_policy re-declares the threshold"


def test_report_states_which_imagery_it_was_measured_from(tmp_path):
    """A roof measured off a 30 cm NAIP tile and one measured off a 15 cm county
    ortho produced identical-looking reports. The resolver recorded the source
    on meta and the PDF printed nothing, so the reader could not tell which
    reliability they were holding."""
    import subprocess

    from src.output.pdf_report import _imagery_label, generate_report

    assert _imagery_label({}) == "not recorded"
    assert _imagery_label({"imagery_source": "naip", "imagery_gsd_m": 0.3}) \
        == "NAIP (0.30 m/px)"
    assert _imagery_label({"imagery_source": "county-3in", "imagery_gsd_m": 0.0762,
                           "imagery_year": 2024}) == "County 3-inch (0.08 m/px, 2024)"

    ri = _good()
    ri.update(imagery_source="naip", imagery_gsd_m=0.3)
    out = tmp_path / "r.pdf"
    generate_report(ri, str(out))
    txt = subprocess.run(["pdftotext", str(out), "-"],
                         capture_output=True, text=True).stdout
    assert "Imagery source" in txt
    assert "NAIP" in txt


def test_a_self_consistent_report_for_the_wrong_building_fails():
    """The Don CeSar -- a large hotel -- produced a 1,232 sqft two-facet roof and
    PASSED the gate. Every check the gate ran was about internal geometric
    self-consistency, and a small plausible roof is internally consistent. None
    of them asked whether it was the right building.

    The MS Buildings footprint is already fetched (it clips the LiDAR) and was
    never compared against the result."""
    ri = _good()                      # two facets, 149 m2 of plan area
    ri["footprint_plan_area_m2"] = 6000.0        # a hotel
    r = score_report(ri)
    failed = {c["id"] for c in r["checks"] if c["severity"] == "FAIL" and not c["ok"]}
    assert "measures_selected_building" in failed
    assert not r["passed"]
    detail = next(c["detail"] for c in r["checks"]
                  if c["id"] == "measures_selected_building")
    assert "not that building" in detail


def test_swallowing_the_neighbours_fails_too():
    """The other direction: a roof far LARGER than its footprint means the
    segmentation merged adjacent structures."""
    ri = _good()
    ri["footprint_plan_area_m2"] = 20.0
    failed = {c["id"] for c in score_report(ri)["checks"]
              if c["severity"] == "FAIL" and not c["ok"]}
    assert "measures_selected_building" in failed


def test_normal_overhang_passes_without_comment():
    """A roof runs ~1.0-1.3x its footprint because of eave overhang. These are
    gross-error bounds, not another Brevard-calibrated tripwire."""
    ri = _good()
    plan = sum(f["plan_area_m2"] for f in ri["facets"])
    for mult in (0.8, 1.0, 1.15, 1.3):
        ri["footprint_plan_area_m2"] = plan / mult
        assert score_report(ri)["passed"], mult


def test_check_is_absent_when_no_footprint_was_recorded():
    """Never a vacuous pass: without a footprint there is no evidence, so the
    check does not appear at all."""
    ids = {c["id"] for c in score_report(_good())["checks"]}
    assert "measures_selected_building" not in ids


def test_gate_printer_shows_the_outcome_not_the_severity_class():
    """format_report_qc printed the check's SEVERITY CLASS in brackets, so a
    passing check rendered as "ok [FAIL]". On a healthy report that is 15 lines
    reading [FAIL] when 2 checks actually failed -- in the log that is the
    primary debugging surface."""
    from src.output.report_qc import format_report_qc

    txt = format_report_qc(score_report(_good()))
    assert "ok [PASS] area_positive" in txt, txt
    assert "ok [FAIL]" not in txt, txt          # the bug: pass marked FAIL

    ri = _good()
    ri["facets"][0]["surface_area_m2"] = 5_000_000        # CRS/units bug
    bad = format_report_qc(score_report(ri))
    assert "XX [FAIL] area_sane" in bad, bad               # a real failure still reads FAIL


def test_selection_evidence_is_surfaced_because_the_area_check_cannot_see_it():
    """measures_selected_building compares the roof against the footprint we
    PICKED. If the pick is wrong, both sides are the same wrong building, the
    ratio is ~1.0 and the report ships CLEAN — not stamped. That is the failure
    that ends a client relationship, and the area check is structurally blind to
    it. The selection's own evidence is the only independent signal."""
    ri = _good()
    ri.update(footprint_plan_area_m2=sum(f["plan_area_m2"] for f in ri["facets"]),
              pin_in_footprint=False, select_dist_m=4.68, select_rank=1,
              select_n_candidates=3)
    r = score_report(ri)
    sel = next(c for c in r["checks"] if c["id"] == "building_selection")
    assert not sel["ok"]
    assert "OUTSIDE" in sel["detail"] and "CLOSER building was passed over" in sel["detail"]
    # the area check passes on the very same report — that is the whole point
    area = next(c for c in r["checks"] if c["id"] == "measures_selected_building")
    assert area["ok"]


def test_a_clean_selection_does_not_warn():
    ri = _good()
    ri.update(pin_in_footprint=True, select_dist_m=0.0, select_rank=0,
              select_n_candidates=1)
    sel = next(c for c in score_report(ri)["checks"] if c["id"] == "building_selection")
    assert sel["ok"], sel


def test_selection_check_is_absent_without_the_evidence():
    ids = {c["id"] for c in score_report(_good())["checks"]}
    assert "building_selection" not in ids
