"""Unit tests for src/roofs/model_segment.py (Phase A plumbing).

Covers the pure-geometry parts that run without torch/rfdetr: converting an
RF-DETR pixel-space prediction into metric Facets, and attaching LiDAR points to
each facet. The model-loading path (segment_facets_model) is exercised only on
Colab where the checkpoint + deps exist.
"""
import numpy as np
import pytest
from affine import Affine

from src.roofs.model_segment import (
    model_facets_to_metric, assign_lidar_points, preprocess_like_training,
    upscale_esrgan,
)

# 0.3 m/px NAIP-like transform: world_x = 356000 + 0.3*col, world_y = 3111500 - 0.3*row
TF = Affine(0.3, 0, 356000.0, 0, -0.3, 3111500.0)
CRS = "EPSG:26917"


def _pred(*rings):
    return {"outline": None, "facets": [{"polygon": r} for r in rings]}


def test_pixel_to_metric_area():
    # 10x10 px square -> 3m x 3m = 9 m^2
    ring = [[10, 10], [20, 10], [20, 20], [10, 20]]
    facets = model_facets_to_metric(_pred(ring), TF, CRS, CRS)
    assert len(facets) == 1
    assert facets[0].polygon.geom_type == "Polygon"
    assert abs(facets[0].polygon.area - 9.0) < 0.2


def test_min_area_filter_drops_tiny():
    tiny = [[0, 0], [2, 0], [2, 2], [0, 2]]          # 0.6m x 0.6m = 0.36 m^2 < 1
    big = [[0, 0], [40, 0], [40, 40], [0, 40]]        # 12m x 12m = 144 m^2
    facets = model_facets_to_metric(_pred(tiny, big), TF, CRS, CRS, min_area_m2=1.0)
    assert len(facets) == 1
    assert facets[0].polygon.area > 100


def test_degenerate_ring_skipped():
    facets = model_facets_to_metric(_pred([[1, 1], [2, 2]]), TF, CRS, CRS)  # <3 pts
    assert facets == []


def test_reprojection_path_runs():
    # src != dst exercises the reprojection branch. Use a METRIC target CRS
    # (an adjacent UTM zone) so the area filter still sees square metres.
    ring = [[10, 10], [30, 10], [30, 30], [10, 30]]
    facets = model_facets_to_metric(_pred(ring), TF, CRS, "EPSG:26918")
    assert len(facets) == 1 and facets[0].polygon.is_valid


def test_upscaled_polygon_maps_back_to_same_area():
    # A facet detected on a x4-upscaled chip has 4x-larger pixel coords; dividing
    # the transform by the upscale factor must map it back to the same metres.
    from affine import Affine
    ring_1x = [[10, 10], [20, 10], [20, 20], [10, 20]]
    ring_4x = [[c * 4 for c in pt] for pt in ring_1x]
    f1 = model_facets_to_metric(_pred(ring_1x), TF, CRS, CRS)
    f4 = model_facets_to_metric(_pred(ring_4x), TF * Affine.scale(0.25, 0.25), CRS, CRS)
    assert abs(f1[0].polygon.area - f4[0].polygon.area) < 0.1


def test_upscale_esrgan_fallback_no_deps():
    # realesrgan isn't installed locally -> must fall back to the original chip,
    # not crash.
    img = np.zeros((16, 16, 3), np.uint8)
    out = upscale_esrgan(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def _points(coords):
    arr = np.zeros(len(coords), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
    for i, (x, y, z) in enumerate(coords):
        arr[i] = (x, y, z)
    return arr


def test_assign_lidar_points_inside_and_drop():
    # One facet ~ world square (356003..356006, 3111494..3111497)
    ring = [[10, 10], [20, 10], [20, 20], [10, 20]]
    facets = model_facets_to_metric(_pred(ring), TF, CRS, CRS)
    inside = [(356004, 3111495, 20), (356005, 3111496, 21), (356004.5, 3111495.5, 20.5)]
    outside = [(356020, 3111450, 19)]
    kept = assign_lidar_points(facets, _points(inside + outside))
    assert len(kept) == 1
    assert kept[0].points is not None and kept[0].count == 3


def test_assign_drops_facet_without_support():
    ring = [[10, 10], [20, 10], [20, 20], [10, 20]]
    facets = model_facets_to_metric(_pred(ring), TF, CRS, CRS)
    kept = assign_lidar_points(facets, _points([(356020, 3111450, 19)]))  # all outside
    assert kept == []


def test_preprocess_shape_and_dtype():
    # (C, H, W) raw chip -> (H, W, 3) uint8 RGB. 4 bands exercises the NIR path.
    bands = np.random.default_rng(0).integers(0, 256, size=(4, 64, 64), dtype=np.uint8)
    out = preprocess_like_training(bands, dtype_max=255.0)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_preprocess_contrast_stretches():
    # a low-contrast chip should get its dynamic range expanded
    bands = np.full((3, 32, 32), 120, np.uint8)
    bands[:, 8:24, 8:24] = 135                          # small bright square
    out = preprocess_like_training(bands, dtype_max=255.0)
    assert out.max() - out.min() >= 15


def test_assign_none_points_noop():
    ring = [[10, 10], [20, 10], [20, 20], [10, 20]]
    facets = model_facets_to_metric(_pred(ring), TF, CRS, CRS)
    kept = assign_lidar_points(facets, None)
    assert len(kept) == 1                              # unchanged when no LiDAR
