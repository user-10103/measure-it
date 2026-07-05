"""Shadow theory, verified on synthetic scenes with known answers."""
import datetime
import math

import numpy as np
from affine import Affine

from src.lidar.shadow_cast import (
    build_dsm,
    cast_shadows,
    fit_sun,
    sun_position,
)


def test_sun_position_florida_summer_noon():
    # solar noon, late June, ~28N: sun nearly overhead, azimuth arbitrary
    when = datetime.datetime(2026, 6, 21, 17, 30)          # ~12:30 local EDT
    az, el = sun_position(28.03, -80.70, when)
    assert el > 75.0


def test_sun_position_morning_is_east():
    when = datetime.datetime(2026, 6, 21, 13, 0)           # ~9:00 local EDT
    az, el = sun_position(28.03, -80.70, when)
    assert 60.0 < az < 120.0                                # east-ish
    assert 30.0 < el < 65.0


def test_tower_casts_shadow_away_from_sun():
    # 10 m tower on flat ground, sun in the WEST at 45 deg ->
    # shadow extends EAST, length ~= height
    dsm = np.zeros((80, 80))
    dsm[38:42, 38:42] = 10.0
    shadow = cast_shadows(dsm, res_m=1.0, az_deg=270.0, el_deg=45.0)
    assert shadow[40, 45] and shadow[40, 48]                # east of tower ✓
    assert not shadow[40, 30]                               # west stays lit
    assert not shadow[40, 52]                               # beyond ~10 m: lit
    assert not shadow[20, 40] and not shadow[60, 40]        # not sideways


def test_higher_sun_shorter_shadow():
    dsm = np.zeros((80, 80))
    dsm[38:42, 38:42] = 10.0
    low = cast_shadows(dsm, 1.0, 270.0, 30.0).sum()
    high = cast_shadows(dsm, 1.0, 270.0, 65.0).sum()
    assert low > high > 0


def test_fit_sun_recovers_azimuth():
    # synthesize: tower scene + an "image" whose dark pixels are exactly the
    # true shadow (sun az=135 SE, el=40) -> the fit must find ~135
    dsm = np.zeros((80, 80))
    dsm[38:42, 38:42] = 8.0
    truth = cast_shadows(dsm, 1.0, 135.0, 40.0)
    chip = np.full((80, 80, 3), 200, np.uint8)
    chip[truth] = 60
    az, el, score, mask = fit_sun(dsm, chip, 1.0)
    assert az is not None and score > 0.3
    assert min(abs(az - 135.0), 360 - abs(az - 135.0)) <= 15.0


def test_build_dsm_from_points():
    tr = Affine(1.0, 0, 0, 0, -1.0, 40.0)                  # 1 m px, y-down
    pts = np.array([[10.5, 20.5, 7.0], [10.6, 20.4, 9.0],  # same cell: max=9
                    [30.5, 10.5, 3.0]])
    dsm = build_dsm(pts, tr, (40, 40))
    assert abs(dsm[19, 10] - 9.0) < 1e-6                   # row = 40-20.5
    assert abs(dsm[29, 30] - 3.0) < 1e-6
    assert np.isfinite(dsm).all()                          # holes filled
