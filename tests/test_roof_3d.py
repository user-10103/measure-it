"""3D from LiDAR planes, not from 45-degree obliques.

A gable: two 6x4 facets sharing a ridge at y=4, each rising 4:12 toward it.
"""
import math

import pytest
from shapely.geometry import Polygon

from src.output.roof_3d import CRACK_WARN_M, Roof3D, build_roof_3d, to_obj

SLOPE = math.tan(math.radians(18.435))      # 4:12


def gable(dz_ridge=0.0):
    """Two facets meeting at y=4. dz_ridge offsets the second plane's intercept."""
    south = Polygon([(0, 0), (6, 0), (6, 4), (0, 4)])
    north = Polygon([(0, 4), (6, 4), (6, 8), (0, 8)])
    return [
        {"polygon": south, "plane_abc": (0.0, SLOPE, 3.0)},
        {"polygon": north, "plane_abc": (0.0, -SLOPE, 3.0 + 8 * SLOPE + dz_ridge)},
    ]


def test_the_ridge_welds_into_one_shared_edge():
    m = build_roof_3d(gable())
    assert m.n_facets == 2 and len(m.faces) == 2
    # 8 distinct plan corners, not 4+4=8 with the ridge doubled -> 6 unique.
    assert len(m.vertices) == 6, "ridge vertices were not welded"
    ridge = [v for v in m.vertices if abs(v[1] - 4) < 1e-6]
    assert len(ridge) == 2
    peak = 3.0 + 4 * SLOPE
    for _x, _y, z in ridge:
        assert z == pytest.approx(peak, abs=1e-6)


def test_eaves_sit_lower_than_the_ridge():
    m = build_roof_3d(gable())
    zs = [v[2] for v in m.vertices]
    assert max(zs) - min(zs) == pytest.approx(4 * SLOPE, abs=1e-6)


def test_a_seam_that_does_not_close_is_reported_not_smoothed():
    """Averaging closes the mesh visually. The gap must still be in the manifest.

    Two planes that disagree at their shared seam mean the seam is in the wrong
    place, or a plane was fitted through canopy. Either way the model renders
    solid, so the number is the only way anyone finds out.
    """
    m = build_roof_3d(gable(dz_ridge=0.60))
    assert m.max_crack_m == pytest.approx(0.60, abs=1e-6)
    assert m.n_cracks_over_tol == 2
    assert m.manifest()["seams_over_tol"] == 2
    assert any("seam" in n for n in m.notes)
    # and it still welds, so the viewer is not shown a hole
    assert len(m.vertices) == 6


def test_a_small_disagreement_is_welded_and_not_flagged():
    """Real independent fits disagree ~1-2 cm at the ridge. That is not a defect."""
    m = build_roof_3d(gable(dz_ridge=0.015))
    assert m.n_cracks_over_tol == 0
    assert m.max_crack_m < CRACK_WARN_M


def test_a_declined_facet_never_gets_an_invented_pitch():
    """Laying it flat at mean roof height makes a model that is wrong invisibly."""
    facets = gable() + [{"polygon": Polygon([(6, 0), (9, 0), (9, 4), (6, 4)])}]
    m = build_roof_3d(facets)
    assert m.n_facets == 2 and m.n_declined == 1
    assert m.manifest()["declined_facets"] == 1
    groups = {g for _i, g in m.faces}
    assert "declined" in groups
    assert any("NO pitch" in n for n in m.notes)


def test_obj_is_one_based_and_keeps_the_groups():
    m = build_roof_3d(gable() + [{"polygon": Polygon([(6, 0), (9, 0), (9, 4), (6, 4)])}])
    obj = to_obj(m)
    assert obj.count("\nv ") + obj.startswith("v ") >= 6
    assert "g roof" in obj and "g declined" in obj
    faces = [l for l in obj.splitlines() if l.startswith("f ")]
    assert faces and all(int(i) >= 1 for l in faces for i in l.split()[1:])
    assert max(int(i) for l in faces for i in l.split()[1:]) <= len(m.vertices)


def test_flat_facets_are_their_own_group():
    f = [{"polygon": Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
          "plane_abc": (0.0, 0.0, 5.0), "is_flat": True}]
    m = build_roof_3d(f)
    assert {g for _i, g in m.faces} == {"flat"}
    assert all(v[2] == pytest.approx(5.0) for v in m.vertices)


def test_empty_input_produces_an_empty_model_not_a_crash():
    m = build_roof_3d([])
    assert m.manifest()["facets"] == 0 and m.vertices == []
    assert to_obj(m).startswith("# roof")


def test_report_dialect_polygon_xy_builds_the_same_model():
    """The report's facets are JSON: `polygon_xy`, never a shapely object.

    Reading only f["polygon"] made the first wiring of this module a silent
    no-op — every facet skipped, empty model returned, caller wrote no file,
    nothing logged.
    """
    shapely_form = gable()
    json_form = [{"polygon_xy": [list(p) for p in f["polygon"].exterior.coords],
                  "plane_abc": f["plane_abc"]} for f in shapely_form]
    a = build_roof_3d(shapely_form)
    b = build_roof_3d(json_form)
    assert len(a.vertices) == len(b.vertices) == 6
    assert sorted(round(v[2], 6) for v in a.vertices) == \
           sorted(round(v[2], 6) for v in b.vertices)


def test_facets_with_no_usable_geometry_raise_instead_of_returning_nothing():
    with pytest.raises(ValueError, match="none produced geometry"):
        build_roof_3d([{"plan_area_m2": 12.0}, {"plan_area_m2": 9.0}])


def test_report_facets_without_pitch_are_all_declined_and_say_so():
    """Before LiDAR fusion every facet has polygon_xy and no plane."""
    facets = [{"polygon_xy": [[0, 0], [4, 0], [4, 4], [0, 4]]},
              {"polygon_xy": [[4, 0], [8, 0], [8, 4], [4, 4]]}]
    m = build_roof_3d(facets)
    assert m.n_facets == 0 and m.n_declined == 2
    assert {g for _i, g in m.faces} == {"declined"}
    assert any("NO pitch" in n for n in m.notes)
