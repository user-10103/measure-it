"""Unit tests for src/roofs/ndsm_segment.py — facet extraction from the height
surface by gradient clustering (no LiDAR/model, pure numpy/scipy/skl)."""
import math
import numpy as np
from affine import Affine
from src.roofs.ndsm_segment import ndsm_facets, segment_by_gradient

TF = Affine(0.5, 0, 356000, 0, -0.5, 3111500)   # 0.5 m/px


def _hip(H=60, W=100, s=0.25):
    yy, xx = np.mgrid[0:H, 0:W]
    return s * np.minimum.reduce([xx, W - 1 - xx, yy, H - 1 - yy]).astype(float)


def test_hip_cut_into_four_planes():
    fac = ndsm_facets(_hip(), TF, min_region_px=60)
    big = [f for f in fac if f["area_px"] > 250]
    assert len(big) >= 4                              # 4 hip planes
    # facets face multiple directions -> boundaries cut along the hips (not blobs)
    quads = {round(f["aspect_deg"] / 90) % 4 for f in big}
    assert len(quads) >= 3
    for f in big:
        assert 5.0 < f["slope_deg"] < 60.0            # real slope, not flat/vertical
        assert f["polygon"].is_valid and f["polygon"].area > 0


def test_flat_roof_one_region():
    z = np.full((50, 50), 3.0) + np.random.default_rng(0).normal(0, 0.02, (50, 50))
    fac = ndsm_facets(z, TF, min_region_px=60)
    # a flat roof -> few facets, all near-flat
    assert all(f["slope_deg"] < 10 for f in fac) or len(fac) <= 2


def test_empty_ndsm():
    assert ndsm_facets(np.zeros((40, 40)), TF) == []
