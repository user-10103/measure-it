"""Fusion is read-only annotation: known synthetic slopes -> exact pitch."""
import logging
import math

import numpy as np
import pytest
from shapely.geometry import box

from src.roofs.fuse_sam_lidar import (
    annotate_facets_with_lidar,
    fuse_into_report_input,
    split_multiplane_facets,
)
from src.roofs.segment import Facet


def _grid_points(poly, z_fn, step=0.5):
    minx, miny, maxx, maxy = poly.bounds
    xs, ys = np.meshgrid(np.arange(minx + 0.25, maxx, step),
                         np.arange(miny + 0.25, maxy, step))
    x, y = xs.ravel(), ys.ravel()
    return np.column_stack([x, y, z_fn(x, y)])


def test_six_twelve_pitch_recovered_exactly():
    # z = 0.5*x  -> gradient 0.5 -> rise 6 per 12 run -> "6:12", slope 26.57 deg
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = _grid_points(f.polygon, lambda x, y: 0.5 * x)
    ann = annotate_facets_with_lidar([f], pts)
    a = ann[1]
    assert a["pitch_string"] == "6:12"
    assert abs(a["slope_deg"] - math.degrees(math.atan(0.5))) < 0.5
    # sloped area = plan / cos(theta): 100 / cos(26.57deg) ~ 111.8
    assert abs(a["surface_area_m2"] - 100.0 / math.cos(math.atan(0.5))) < 1.0
    assert not a["is_flat"]


def test_flat_roof_detected():
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = _grid_points(f.polygon, lambda x, y: np.full_like(x, 5.0))
    a = annotate_facets_with_lidar([f], pts)[1]
    assert a["is_flat"]
    assert abs(a["surface_area_m2"] - 100.0) < 0.5      # flat: sloped == plan


def test_two_facets_annotated_independently():
    f1 = Facet(facet_id=1, polygon=box(0, 0, 10, 10))     # 6:12
    f2 = Facet(facet_id=2, polygon=box(20, 0, 30, 10))    # flat
    pts = np.vstack([
        _grid_points(f1.polygon, lambda x, y: 0.5 * x),
        _grid_points(f2.polygon, lambda x, y: np.full_like(x, 3.0)),
    ])
    ann = annotate_facets_with_lidar([f1, f2], pts)
    assert ann[1]["pitch_string"] == "6:12" and not ann[1]["is_flat"]
    assert ann[2]["is_flat"]


def test_too_few_points_stays_unspecified():
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 1.0], [3.0, 1.0, 0.5]])
    assert annotate_facets_with_lidar([f], pts) == {}    # absent, not wrong


def test_fusion_never_touches_geometry():
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = _grid_points(f.polygon, lambda x, y: 0.5 * x)
    before = list(f.polygon.exterior.coords)
    annotate_facets_with_lidar([f], pts)
    assert list(f.polygon.exterior.coords) == before     # frozen shapes


def test_split_multiplane_facet_at_the_ridge():
    # one facet the model returned as a blob, but the points form a ridge at x=5:
    # left half z=0.5*x, right half z=0.5*(10-x) -> two planes ~53deg apart
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = _grid_points(f.polygon, lambda x, y: np.where(x < 5, 0.5 * x, 0.5 * (10 - x)),
                       step=0.3)
    out, changed = split_multiplane_facets([f], pts)
    assert changed and len(out) == 2
    assert abs(sum(g.polygon.area for g in out) - 100.0) < 1.0   # area conserved
    assert all(g.polygon.area > 20 for g in out)                 # two real halves


def test_single_plane_facet_not_split():
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = _grid_points(f.polygon, lambda x, y: 0.4 * x + 0.1 * y + 2.0, step=0.3)
    out, changed = split_multiplane_facets([f], pts)
    assert not changed and len(out) == 1


def test_flat_facet_not_split_on_clutter():
    # a flat roof with a tilted rooftop-clutter blob must NOT split — a flat roof
    # is one plane (the Tampa regression: a big flat facet was split into slivers).
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    flat = _grid_points(f.polygon, lambda x, y: np.full_like(x, 5.0), step=0.3)
    rng = np.random.RandomState(7)
    cx, cy = rng.uniform(0, 3, 300), rng.uniform(0, 3, 300)
    clutter = np.column_stack([cx, cy, 5.0 + 1.2 * cx])          # tilted HVAC blob
    out, changed = split_multiplane_facets([f], np.vstack([flat, clutter]))
    assert not changed and len(out) == 1


def test_two_plane_split_rejected_when_a_piece_is_point_starved():
    # a genuine two-plane facet, but the crease is off-centre and sampling coarse,
    # so one piece would fall under MIN_FACET_POINTS -> don't split (would just
    # manufacture a pitch-less sliver).
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    # planes meet at x=3.75; left piece is the small one, coarse grid keeps it <30 pts
    pts = _grid_points(f.polygon,
                       lambda x, y: np.where(x < 3.75, 0.5 * x, 1.5 + 0.1 * x),
                       step=1.1)
    out, changed = split_multiplane_facets([f], pts)
    assert not changed and len(out) == 1


def test_noisy_flat_roof_accepted_as_flat():
    # a flat roof so noisy the RANSAC fit is below the inlier floor, but the best
    # plane is LEVEL -> recorded as flat (0:12), not dropped and left unspecified.
    f = Facet(facet_id=1, polygon=box(0, 0, 12, 12))
    rng = np.random.RandomState(6)
    xs, ys = rng.uniform(0, 12, 500), rng.uniform(0, 12, 500)
    z = 6.0 + rng.normal(0, 1.2, 500)                           # ~16% within 0.25 m
    ann = annotate_facets_with_lidar([f], np.column_stack([xs, ys, z]))
    # the facet is recorded (not dropped) and treated as flat -> surface == plan
    assert 1 in ann and ann[1]["is_flat"] is True
    assert ann[1]["surface_area_m2"] == pytest.approx(f.polygon.area, rel=1e-6)


def test_ground_returns_excluded_from_fit_and_eave():
    # a facet with roof points ~5-9 m up, plus driveway returns at z~0.2 bleeding in
    f = Facet(facet_id=1, polygon=box(0, 0, 8, 8))
    roof = _grid_points(f.polygon, lambda x, y: 0.4 * x + 5.0, step=0.4)
    ground = np.column_stack([np.linspace(0, 8, 60), np.linspace(0, 8, 60),
                              np.full(60, 0.2)])
    pts = np.vstack([roof, ground])
    ann = annotate_facets_with_lidar([f], pts, ground_z=0.0)
    assert 1 in ann
    # eave computed on roof points only -> well above ground, not dragged to ~0.2
    assert ann[1]["eave_height_m"] > 3.0


def test_fuse_into_report_input_fills_only_annotated():
    ri = {"facets": [
        {"facet_id": 1, "polygon_xy": [[0, 0]], "plan_area_m2": 100.0,
         "pitch_string": None, "slope_deg": None, "aspect_bin": None,
         "is_flat": False, "surface_area_m2": None},
        {"facet_id": 2, "polygon_xy": [[5, 5]], "plan_area_m2": 50.0,
         "pitch_string": None, "slope_deg": None, "aspect_bin": None,
         "is_flat": False, "surface_area_m2": None},
    ]}
    ann = {1: {"pitch_string": "6:12", "slope_deg": 26.57, "aspect_bin": "E",
               "is_flat": False, "surface_area_m2": 111.8}}
    out = fuse_into_report_input(ri, ann)
    assert out["facets"][0]["pitch_string"] == "6:12"
    assert out["facets"][0]["polygon_xy"] == [[0, 0]]     # geometry untouched
    assert out["facets"][1]["pitch_string"] is None       # stays unspecified
    # the unresolved facet must be FLAGGED, not left silent (gate honesty guard)
    assert out["facets"][1]["needs_review"] is True
    assert out["facets"][0].get("needs_review") is not True  # resolved -> not flagged


def test_unresolved_pitch_facet_is_flagged_so_gate_passes():
    """A facet whose plane fit failed (no annotation) is flagged needs_review, so
    report_qc.pitch_resolved / slope_applied treat it as honestly disclosed rather
    than a silent plan-area-as-surface-area error. This is the recurring gate FAIL."""
    from src.output.report_qc import score_report
    ri = {
        "address": "x", "report_id": "MI-TEST",
        "outline_xy": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "facets": [
            {"facet_id": 1, "polygon_xy": [[0, 0], [10, 0], [10, 5], [0, 5]],
             "plan_area_m2": 50.0, "pitch_string": None, "slope_deg": None,
             "aspect_bin": None, "is_flat": False, "surface_area_m2": None},
        ],
        "edges": [],
    }
    # no annotation -> facet 1 is unresolved
    fuse_into_report_input(ri, {})
    assert ri["facets"][0]["needs_review"] is True
    res = score_report(ri)
    checks = {c["id"]: c for c in res["checks"]}
    assert checks["pitch_resolved"]["ok"], checks["pitch_resolved"]["detail"]
    assert checks["slope_applied"]["ok"], checks["slope_applied"]["detail"]


def _attribution_records(caplog):
    return [r for r in caplog.records if "LiDAR attribution" in r.getMessage()]


def test_attribution_diagnostic_names_bad_srs_when_points_are_kilometres_off(caplog):
    """Points supplied, none land in any facet -> ONE decisive WARNING naming the
    cause. This is the 1600 Sarno failure (550 points, 0/6 facets) made
    self-diagnosing; kilometres of offset means the reprojection, not the roof."""
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    far = box(5000, 5000, 5010, 5010)            # same-size roof, ~7 km away
    pts = _grid_points(far, lambda x, y: 0.5 * (x - 5000) + 20.0)
    with caplog.at_level(logging.WARNING, logger="src.roofs.fuse_sam_lidar"):
        ann = annotate_facets_with_lidar([f], pts)
    assert ann == {}                              # behaviour unchanged
    recs = _attribution_records(caplog)
    assert len(recs) == 1, caplog.text
    msg = recs[0].getMessage()
    assert recs[0].levelno == logging.WARNING
    assert "0/1 facet(s) annotated" in msg
    assert "EPT header SRS" in msg


def test_attribution_diagnostic_names_footprint_offset_at_tens_of_metres(caplog):
    """A 25 m offset is the wrong-building signature (the pin sat 22 m from the
    selected footprint), NOT a reprojection error — the verdict must say so."""
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    near = box(25, 0, 35, 10)                     # same size, disjoint, 25 m east
    pts = _grid_points(near, lambda x, y: 0.5 * (x - 25) + 20.0)
    with caplog.at_level(logging.WARNING, logger="src.roofs.fuse_sam_lidar"):
        annotate_facets_with_lidar([f], pts)
    recs = _attribution_records(caplog)
    assert len(recs) == 1, caplog.text
    assert "wrong building" in recs[0].getMessage()


def test_attribution_diagnostic_silent_on_the_happy_path(caplog):
    """Must cost nothing and say nothing when attribution works."""
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = _grid_points(f.polygon, lambda x, y: 0.5 * x)
    with caplog.at_level(logging.INFO, logger="src.roofs.fuse_sam_lidar"):
        assert annotate_facets_with_lidar([f], pts)
    assert not _attribution_records(caplog)


def test_plane_fit_on_a_handful_of_inliers_is_rejected():
    """A ratio can clear the sparse floor on very few points: 36 points at 41.7%
    is 15 inliers. 1600 Sarno produced a 59.8-degree 'facet' that way on an 8-degree
    roof, which then polluted the edge graph. Require an absolute inlier count."""
    f = Facet(facet_id=1, polygon=box(0, 0, 6, 6))
    rng = np.random.RandomState(11)
    # 14 points on a steep plane + 22 scattered: ratio can pass, inliers cannot
    n_in = 14
    xs = rng.uniform(0, 6, n_in); ys = rng.uniform(0, 6, n_in)
    good = np.column_stack([xs, ys, 1.7 * xs])
    gx, gy = rng.uniform(0, 6, 22), rng.uniform(0, 6, 22)
    junk = np.column_stack([gx, gy, rng.uniform(0, 12, 22)])
    ann = annotate_facets_with_lidar([f], np.vstack([good, junk]), min_points=30)
    assert ann == {}          # absent, not a bogus 60-degree roof plane


def test_undersegmented_facet_is_detected_against_lidar():
    """The gate is blind to under-segmentation: one big facet satisfies
    facets_partition, facets_coverage and edges_typed trivially while
    under-reporting surface area. LiDAR is the independent evidence."""
    from src.roofs.fuse_sam_lidar import detect_multiplane_facets
    # ONE facet covering a roof whose points are actually two planes (a ridge)
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = _grid_points(f.polygon,
                       lambda x, y: np.where(x < 5, 0.5 * x, 0.5 * (10 - x)),
                       step=0.3)
    assert detect_multiplane_facets([f], pts) == [1]


def test_single_plane_facet_is_not_flagged_as_undersegmented():
    from src.roofs.fuse_sam_lidar import detect_multiplane_facets
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = _grid_points(f.polygon, lambda x, y: 0.4 * x + 0.1 * y + 2.0, step=0.3)
    assert detect_multiplane_facets([f], pts) == []


def test_flat_roof_clutter_is_not_called_undersegmented():
    """A flat commercial roof's HVAC clutter is not a second roof plane — flagging
    it would cry wolf on every legitimate flat roof (the Tampa regression)."""
    from src.roofs.fuse_sam_lidar import detect_multiplane_facets
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    flat = _grid_points(f.polygon, lambda x, y: np.full_like(x, 5.0), step=0.3)
    rng = np.random.RandomState(21)
    cx, cy = rng.uniform(0, 3, 400), rng.uniform(0, 3, 400)
    clutter = np.column_stack([cx, cy, 5.0 + 1.2 * cx])
    assert detect_multiplane_facets([f], np.vstack([flat, clutter])) == []


def test_flat_facet_spanning_two_roof_levels_is_flagged():
    """2725 Judge Fran shipped 43,029 sqft as ONE flat facet with zero ridges,
    hips, valleys, parapets or transitions — and passed the gate, because the
    angle test cannot see a level change. A commercial roof at two elevations is
    two sections, and the parapet between them is a real edge."""
    from src.roofs.fuse_sam_lidar import detect_multiplane_facets
    f = Facet(facet_id=1, polygon=box(0, 0, 20, 20))
    # two flat halves 1.2 m apart in elevation — a parapet step, not a slope
    pts = _grid_points(f.polygon,
                       lambda x, y: np.where(x < 10, 5.0, 6.2), step=0.4)
    assert detect_multiplane_facets([f], pts) == [1]


def test_flat_roof_with_hvac_is_not_called_two_levels():
    """Rooftop units sit above the deck but are a small minority — they must not
    read as a second roof level, or every flat commercial roof fails."""
    from src.roofs.fuse_sam_lidar import detect_multiplane_facets
    f = Facet(facet_id=1, polygon=box(0, 0, 20, 20))
    deck = _grid_points(f.polygon, lambda x, y: np.full_like(x, 5.0), step=0.4)
    rng = np.random.RandomState(31)
    n = int(0.08 * len(deck))                       # 8% of returns are plant
    hx, hy = rng.uniform(0, 4, n), rng.uniform(0, 4, n)
    hvac = np.column_stack([hx, hy, np.full(n, 6.5)])
    assert detect_multiplane_facets([f], np.vstack([deck, hvac])) == []


def test_single_level_flat_roof_is_not_flagged():
    from src.roofs.fuse_sam_lidar import detect_multiplane_facets
    f = Facet(facet_id=1, polygon=box(0, 0, 20, 20))
    rng = np.random.RandomState(32)
    pts = _grid_points(f.polygon, lambda x, y: np.full_like(x, 5.0), step=0.4)
    pts[:, 2] += rng.normal(0, 0.03, len(pts))      # normal membrane noise
    assert detect_multiplane_facets([f], pts) == []


def test_pitched_roof_is_not_mistaken_for_two_levels():
    """A real slope has continuously varying z and no elevation gap — the level
    test must not fire on it, or every pitched roof gets flagged."""
    from src.roofs.fuse_sam_lidar import detect_multiplane_facets
    f = Facet(facet_id=1, polygon=box(0, 0, 20, 20))
    pts = _grid_points(f.polygon, lambda x, y: 0.5 * x + 3.0, step=0.4)
    assert detect_multiplane_facets([f], pts) == []


def test_flat_roof_splits_at_a_level_change():
    """A commercial roof at two heights is two sections. The angle split cannot
    see it (parallel planes never intersect), so 2725 Judge Fran shipped 43,029
    sqft as one facet with no internal edges at all."""
    from src.roofs.fuse_sam_lidar import split_level_facets
    f = Facet(facet_id=1, polygon=box(0, 0, 20, 20))
    pts = _grid_points(f.polygon, lambda x, y: np.where(x < 10, 5.0, 6.4), step=0.4)
    out, changed = split_level_facets([f], pts)
    assert changed and len(out) == 2
    assert abs(sum(g.polygon.area for g in out) - 400.0) < 2.0   # area conserved
    assert all(g.polygon.area > 80 for g in out)                 # two real sections


def test_single_level_flat_roof_is_not_split():
    from src.roofs.fuse_sam_lidar import split_level_facets
    f = Facet(facet_id=1, polygon=box(0, 0, 20, 20))
    pts = _grid_points(f.polygon, lambda x, y: np.full_like(x, 5.0), step=0.4)
    out, changed = split_level_facets([f], pts)
    assert not changed and len(out) == 1


def test_hvac_does_not_trigger_a_level_split():
    from src.roofs.fuse_sam_lidar import split_level_facets
    f = Facet(facet_id=1, polygon=box(0, 0, 20, 20))
    deck = _grid_points(f.polygon, lambda x, y: np.full_like(x, 5.0), step=0.4)
    rng = np.random.RandomState(41)
    n = int(0.08 * len(deck))
    hvac = np.column_stack([rng.uniform(0, 4, n), rng.uniform(0, 4, n),
                            np.full(n, 6.6)])
    out, changed = split_level_facets([f], np.vstack([deck, hvac]))
    assert not changed and len(out) == 1


def test_flat_facet_does_not_report_a_pitch_it_was_not_charged_for():
    """A facet just under FLAT_SLOPE_DEG gets no slope multiplier on its area, so
    it must not print a pitch either. 755 E Eau Gallie listed 2,481 sqft under a
    1/12 row while counting the same area as flat and excluding it from pitched
    area — the report contradicting itself."""
    import math
    from src.roofs.metrics import FLAT_SLOPE_DEG
    f = Facet(facet_id=1, polygon=box(0, 0, 12, 12))
    grad = math.tan(math.radians(FLAT_SLOPE_DEG - 0.3))      # just inside "flat"
    pts = _grid_points(f.polygon, lambda x, y: grad * x + 4.0, step=0.3)
    a = annotate_facets_with_lidar([f], pts)[1]
    assert a["is_flat"] is True
    assert a["pitch_string"] == "0:12"                       # not "1:12"
    # and the area carries no slope multiplier, consistent with that
    assert a["surface_area_m2"] == pytest.approx(f.polygon.area, rel=1e-6)


def test_genuinely_pitched_facet_still_reports_its_pitch():
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = _grid_points(f.polygon, lambda x, y: 0.5 * x)      # 6:12
    a = annotate_facets_with_lidar([f], pts)[1]
    assert a["is_flat"] is False and a["pitch_string"] == "6:12"


def test_rooftop_plant_is_not_a_level_change():
    """755 E Eau Gallie facets 5 and 6: a clean 3.4-3.7 m step, both clusters well
    supported — but the two plan hulls summed to 1.79 and 1.59 of a polygon that is
    1.00. The levels are SUPERIMPOSED: a mechanical unit or stair bulkhead above
    the deck, not two sections. Those facets had the HIGHEST explained fractions on
    the roof (0.88, 0.97), so calling them under-segmented was a false positive."""
    from src.roofs.fuse_sam_lidar import detect_multiplane_facets, split_level_facets
    f = Facet(facet_id=1, polygon=box(0, 0, 12, 12))
    deck = _grid_points(f.polygon, lambda x, y: np.full_like(x, 5.0), step=0.3)
    # plant sitting 3.5 m up, over the MIDDLE of the same deck (superimposed)
    rng = np.random.RandomState(51)
    n = int(0.35 * len(deck))
    plant = np.column_stack([rng.uniform(3, 9, n), rng.uniform(3, 9, n),
                             np.full(n, 8.5)])
    pts = np.vstack([deck, plant])
    assert detect_multiplane_facets([f], pts) == []        # not under-segmented
    out, changed = split_level_facets([f], pts)
    assert not changed and len(out) == 1                   # nothing to cut


def test_genuine_side_by_side_levels_still_split():
    """A real level change — two sections on different ground — must still split."""
    from src.roofs.fuse_sam_lidar import detect_multiplane_facets, split_level_facets
    f = Facet(facet_id=1, polygon=box(0, 0, 20, 20))
    pts = _grid_points(f.polygon, lambda x, y: np.where(x < 10, 5.0, 6.4), step=0.4)
    assert detect_multiplane_facets([f], pts) == [1]
    out, changed = split_level_facets([f], pts)
    assert changed and len(out) == 2


# --- density-scaled plane requirement + no silent declines -------------------
# MIN_PLANE_INLIERS is a COUNT, so it is secretly an AREA that moves by 15x
# across US LiDAR surveys: ~0.7 m^2 at Brevard's ~30 pts/m^2, but ~10 m^2 at
# rural 3DEP QL2's ~2 pts/m^2, where it silently discards every dormer.

def test_plane_requirement_holds_area_not_count_across_survey_densities():
    from src.roofs.fuse_sam_lidar import (ABS_MIN_PLANE_INLIERS,
                                          MIN_PLANE_INLIERS,
                                          required_plane_inliers)
    # Brevard-grade survey: unchanged, so validated roofs cannot regress
    assert required_plane_inliers(30.0) == MIN_PLANE_INLIERS
    assert required_plane_inliers(100.0) == MIN_PLANE_INLIERS   # capped, never tightens
    # rural QL2: relaxed, or a real dormer needs 10 m^2 to be seen at all
    assert required_plane_inliers(2.0) == ABS_MIN_PLANE_INLIERS
    assert ABS_MIN_PLANE_INLIERS <= required_plane_inliers(8.0) < MIN_PLANE_INLIERS
    # unmeasurable density is "no evidence", never "sparse"
    assert required_plane_inliers(0.0) == MIN_PLANE_INLIERS


def test_survey_density_is_scene_level_not_per_facet():
    """Density is a property of the collection, so one number describes it. A
    per-facet figure would read low exactly where a facet is occluded — the case
    where relaxing the plane requirement is least safe."""
    from src.roofs.fuse_sam_lidar import survey_density
    f1 = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    f2 = Facet(facet_id=2, polygon=box(10, 0, 20, 10))
    pts = _grid_points(f1.polygon, lambda x, y: 0.0 * x)      # 4 pts/m^2, f1 only
    d = survey_density([f1, f2], pts)
    assert 1.5 < d < 2.5, d          # 400 pts over 200 m^2 of roof, not over f1
    assert survey_density([], pts) == 0.0
    assert survey_density([f1], np.empty((0, 3))) == 0.0


def test_every_unannotated_facet_says_why():
    """A facet absent from `annotations` has five possible causes and the report
    could not tell them apart, so "unspecified" cost a re-run of the address to
    diagnose. Two of the five paths were a bare `continue` with no log at all."""
    starved = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    declines: dict = {}
    ann = annotate_facets_with_lidar(
        [starved], _grid_points(starved.polygon, lambda x, y: 0.0 * x, step=5.0),
        declines=declines)
    assert ann == {}                      # absent, as before
    assert 1 in declines                  # ...but no longer silent
    assert "LiDAR points" in declines[1]
    # a facet with no geometry at all is reported too, not skipped invisibly
    from shapely.geometry import Polygon
    declines.clear()
    annotate_facets_with_lidar([Facet(facet_id=2, polygon=Polygon())],
                               _grid_points(box(0, 0, 10, 10), lambda x, y: 0.0 * x),
                               declines=declines)
    assert declines.get(2) == "no polygon"


def test_annotation_is_unchanged_when_no_declines_sink_is_passed():
    """The sink is optional: every existing caller keeps its exact behaviour."""
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = _grid_points(f.polygon, lambda x, y: 0.5 * x)
    assert annotate_facets_with_lidar([f], pts)[1]["pitch_string"] == "6:12"


def test_density_relaxation_does_not_readmit_the_sarno_noise_plane():
    """Scaling MIN_PLANE_INLIERS by density ALONE regresses the bug it was added
    for: the 36-point Sarno facet spans 36 m^2, so it reads as 1 pt/m^2 "sparse"
    and the relaxed floor of 8 accepts its 14-inlier 59.8-degree plane. Relaxing
    the count is only safe when the plane explains the facet — here 14/36 = 39%,
    well under the bar, so it stays out of the edge graph."""
    from src.roofs.fuse_sam_lidar import RELAXED_EXPLAINS_MIN
    f = Facet(facet_id=1, polygon=box(0, 0, 6, 6))
    rng = np.random.RandomState(11)
    xs, ys = rng.uniform(0, 6, 14), rng.uniform(0, 6, 14)
    good = np.column_stack([xs, ys, 1.7 * xs])          # steep, 14 points
    gx, gy = rng.uniform(0, 6, 22), rng.uniform(0, 6, 22)
    junk = np.column_stack([gx, gy, rng.uniform(0, 12, 22)])
    declines: dict = {}
    ann = annotate_facets_with_lidar([f], np.vstack([good, junk]),
                                     min_points=30, declines=declines)
    assert ann == {}
    assert "explains only" in declines[1]
    assert 14 / 36 < RELAXED_EXPLAINS_MIN


def test_a_clean_sparse_facet_is_recovered_not_dropped():
    """The case the relaxation exists for: a real dormer on a rural QL2 survey.
    Few points, but they are a clean plane — 20 inliers is an ARBITRARY bar that
    only happens to suit Brevard's ~30 pts/m^2."""
    f = Facet(facet_id=1, polygon=box(0, 0, 12, 12))     # 144 m^2
    # ~0.25 pts/m^2 over the facet: 36 clean points on a 6:12 plane
    rng = np.random.RandomState(3)
    xs, ys = rng.uniform(0, 12, 36), rng.uniform(0, 12, 36)
    pts = np.column_stack([xs, ys, 0.5 * xs])
    declines: dict = {}
    ann = annotate_facets_with_lidar([f], pts, min_points=30, declines=declines)
    assert ann, declines                                 # recovered, not dropped
    assert ann[1]["pitch_string"] == "6:12"
    assert ann[1]["explained_frac"] > 0.9                # it earned the relaxation
