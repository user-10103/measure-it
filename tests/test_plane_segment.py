"""Sequential-RANSAC plane segmentation on a synthetic noisy hip point cloud."""
import numpy as np
from src.roofs.plane_segment import segment_facets_planes
from src.roofs.plane_fit import fit_plane_ransac


def _hip_points(W=40, H=40, s=0.3, n=90, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, W, n * n); ys = rng.uniform(0, H, n * n)
    z = s * np.minimum.reduce([xs, W - xs, ys, H - ys]) + rng.normal(0, noise, n * n)
    arr = np.zeros(len(xs), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
    arr["x"], arr["y"], arr["z"] = xs + 356000, ys + 3111000, z + 20
    return arr


def test_hip_pointcloud_gives_four_planes():
    facets = segment_facets_planes(_hip_points(), dist_thresh=0.15, min_inliers=80)
    assert 4 <= len(facets) <= 9                       # ~4 planes (+ maybe slivers)
    for f in facets:
        assert f.points is not None and f.count >= 80
    # the 4 biggest facets face different directions
    big = sorted(facets, key=lambda f: -f.count)[:4]
    grads = set()
    for f in big:
        p = fit_plane_ransac(f.points)
        grads.add((round(p.a, 1), round(p.b, 1)))
    assert len(grads) >= 3


def test_empty():
    assert segment_facets_planes(None) == []
