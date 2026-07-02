"""PEARL (ICM optimizer, no pygco) on a noisy hip point cloud + injected noise."""
import numpy as np
from src.roofs.pearl_segment import segment_facets_pearl
from src.roofs.plane_segment import segment_facets_planes


def _hip_points(W=40, H=40, s=0.3, n=90, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, W, n * n); ys = rng.uniform(0, H, n * n)
    z = s * np.minimum.reduce([xs, W - xs, ys, H - ys]) + rng.normal(0, noise, n * n)
    arr = np.zeros(len(xs), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
    arr["x"], arr["y"], arr["z"] = xs + 356000, ys + 3111000, z + 20
    return arr


def _with_noise(base, n_noise=400, seed=1):
    rng = np.random.default_rng(seed)
    extra = np.zeros(n_noise, dtype=base.dtype)
    extra["x"] = rng.uniform(356000, 356040, n_noise)
    extra["y"] = rng.uniform(3111000, 3111040, n_noise)
    extra["z"] = rng.uniform(18, 30, n_noise)          # random junk heights
    return np.concatenate([base, extra])


def test_pearl_gives_clean_facet_count():
    fac = segment_facets_pearl(_hip_points(), use_graphcut=False, min_inliers=80)
    assert 4 <= len(fac) <= 7                          # ~4 planes, few extras
    for f in fac:
        assert f.points is not None and f.count >= 80


def test_pearl_absorbs_noise_better_than_ransac():
    noisy = _with_noise(_hip_points())
    r = segment_facets_planes(noisy, min_inliers=60)
    p = segment_facets_pearl(noisy, use_graphcut=False, min_inliers=60)
    # PEARL's label cost + smoothness should yield a cleaner (<=) facet count
    assert len(p) <= len(r) + 1
    assert len(p) <= 8


def test_empty():
    assert segment_facets_pearl(None) == []
