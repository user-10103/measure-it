"""Roofs are piecewise-planar surfaces, and that constrains what a seam can be.

Reference geometry: a 12x8 gable, ridge along y=4, both halves at 4:12.
"""
import math

import pytest

from src.roofs.roof_rules import (
    LEVEL_RISE, audit_partition, classify_boundary, classify_seam,
    fold_angle_deg, rise_along,
)

S = math.tan(math.radians(18.435))          # 4:12
SOUTH = (0.0,  S, 3.0)                      # rises northward to the ridge
NORTH = (0.0, -S, 3.0 + 8 * S)              # falls northward from the ridge
RIDGE = [[0, 4], [12, 4]]                   # level, runs E-W


def test_a_real_ridge_is_level_convex_and_closes():
    v = classify_seam(SOUTH, NORTH, RIDGE)
    assert v.edge_type == "ridge"
    assert v.real and v.convex is True
    assert v.step_m < 1e-9                  # the planes meet exactly here
    assert v.rise <= LEVEL_RISE
    assert v.fold_deg == pytest.approx(2 * 18.435, abs=0.2)


def test_a_hip_is_the_same_fold_on_a_sloping_line():
    """Ridge vs hip is not a different mechanism — only whether the crease is level."""
    west = (S, 0.0, 3.0)
    hip = [[0, 0], [6, 6]]                  # diagonal crease
    v = classify_seam(SOUTH, west, hip)
    assert v.real and v.convex is True
    assert v.edge_type == "hip"
    assert v.rise > LEVEL_RISE


def test_a_valley_is_the_concave_case():
    """Two planes rising away from the seam — an inside corner."""
    a = (0.0, -S, 3.0)                      # rises southward (toward -y)
    b = (0.0,  S, 3.0)                      # rises northward
    v = classify_seam(a, b, [[0, 0], [12, 0]])
    assert v.edge_type == "valley"
    assert v.convex is False and v.real


def test_the_same_plane_twice_is_a_false_seam_not_an_edge():
    """Two facets on one plane are ONE facet. The line between them is an artefact."""
    v = classify_seam(SOUTH, SOUTH, [[0, 2], [12, 2]])
    assert v.edge_type == "false_seam"
    assert v.real is False
    assert "one facet, not two" in v.detail


def test_parallel_planes_at_different_heights_are_a_step():
    higher = (0.0, S, 3.0 + 1.2)
    v = classify_seam(SOUTH, higher, [[0, 2], [12, 2]])
    assert v.edge_type == "step"
    assert v.real                            # a parapet is real roof geometry
    assert v.step_m == pytest.approx(1.2, abs=1e-6)


def test_601_gulf_way_the_fictional_seam():
    """Different planes whose intersection line is nowhere near the seam drawn.

    601 Gulf Way shipped four parallel strips at 1/12, 4/12 and 7/12 with ZERO
    ridge and ZERO hip between them — three planes fitted to what the diagram
    shows as one continuous surface. The old classifier asked only whether the
    compass aspects were within 45 degrees, said "transition", and passed the
    report clean.
    """
    p1 = (0.0, math.tan(math.radians(4.76)),  3.0)      # 1/12
    p2 = (0.0, math.tan(math.radians(30.26)), 3.0 + 2.0)  # 7/12, offset upward
    v = classify_seam(p1, p2, [[0, 2], [12, 2]])
    assert v.edge_type == "fictional"
    assert v.real is False
    assert "does not pass through this seam" in v.detail


def test_a_seam_with_an_unmeasured_flank_says_so_rather_than_guessing():
    v = classify_seam(SOUTH, None, RIDGE)
    assert v.edge_type == "unspecified" and v.real


def test_eave_is_level_and_rake_climbs():
    """The eave is where water leaves, so it runs ACROSS the slope."""
    eave, r_e = classify_boundary(SOUTH, [[0, 0], [12, 0]])   # E-W, across slope
    rake, r_r = classify_boundary(SOUTH, [[0, 0], [0, 4]])    # N-S, up the slope
    assert eave == "eave" and r_e <= LEVEL_RISE
    assert rake == "rake" and r_r == pytest.approx(S, abs=1e-9)
    assert classify_boundary(None, [[0, 0], [1, 0]])[0] == "unspecified"


def test_a_hip_roof_has_no_rakes():
    """Every perimeter edge of a hip faces a facet that slopes toward it.

    212 13th Ave N shipped 155 ft of rake against 71 ft of eave — 69% of the
    perimeter as rake on a roof the same report calls hip-dominant. Rakes only
    exist on gable ends.
    """
    planes = {"S": (0.0, S, 3.0), "N": (0.0, -S, 3.0 + 8 * S),
              "W": (S, 0.0, 3.0), "E": (-S, 0.0, 3.0 + 12 * S)}
    edges = {"S": [[0, 0], [12, 0]], "N": [[0, 8], [12, 8]],
             "W": [[0, 0], [0, 8]],  "E": [[12, 0], [12, 8]]}
    for k in planes:
        assert classify_boundary(planes[k], edges[k])[0] == "eave", k


def test_audit_counts_unreal_seams_and_their_length():
    seams = [
        {"geometry_xy": RIDGE, "facets": (1, 2)},                  # real ridge
        {"geometry_xy": [[0, 2], [12, 2]], "facets": (1, 3)},      # false seam
        {"geometry_xy": [[0, 6], [6, 6]], "facets": (2, 4)},       # fictional
    ]
    # facet 4 must differ from facet 2 in GRADIENT as well as height — same
    # gradient at a different height is a step, which is real roof geometry.
    # (My first fixture here got that wrong and the test caught it.)
    planes = {1: SOUTH, 2: NORTH, 3: SOUTH,
              4: (0.0, -math.tan(math.radians(40.0)), 3.0 + 8 * S + 1.5)}
    rep = audit_partition(seams, planes)
    assert rep.total_seams == 3
    assert rep.false_seams == 1 and rep.fictional_seams == 1
    assert rep.false_len_m == pytest.approx(12.0)
    assert rep.unreal_fraction == pytest.approx(2 / 3)
    assert rep.to_dict()["unreal_seam_fraction"] == pytest.approx(0.667, abs=1e-3)


def test_helpers_are_sane():
    assert fold_angle_deg(SOUTH, SOUTH) == pytest.approx(0.0, abs=1e-9)
    assert rise_along(SOUTH, [[0, 0], [1, 0]]) == pytest.approx(0.0, abs=1e-12)
    assert rise_along(SOUTH, [[0, 0], [0, 1]]) == pytest.approx(S, abs=1e-12)
