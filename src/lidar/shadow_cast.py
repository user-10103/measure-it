"""Physics-predicted shadow masks: sun geometry + LiDAR DSM -> shadow pixels.

The theory under test: shadows in the aerial image are DETERMINISTIC — given
the sun's position at capture time and a 3D model of the scene (trees +
buildings from the LiDAR returns we already fetch), ray-casting predicts
exactly which pixels were shadowed. Uses:

  * discount facet edges that coincide with predicted shadow boundaries
  * de-shadow the chip before segmentation
  * free per-pixel shadow labels for training

NAIP metadata carries the acquisition DATE but rarely the exact time, so
``fit_sun`` solves for the sun angle from the image itself: try candidate
azimuth/elevation pairs, ray-cast each, keep the one whose predicted mask
best matches the image's dark pixels. The scene tells us where the sun was.

Pure numpy — no new dependencies. Experimental module: nothing in the report
pipeline imports it yet.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── solar geometry (NOAA approximation, good to ~0.2 deg) ────────────────────
def sun_position(lat: float, lon: float, when_utc) -> Tuple[float, float]:
    """(azimuth deg from North cw, elevation deg) for a UTC datetime."""
    doy = when_utc.timetuple().tm_yday
    hour = when_utc.hour + when_utc.minute / 60 + when_utc.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    tst = hour * 60 + eqtime + 4 * lon          # true solar time (min), lon +E
    ha = math.radians(tst / 4 - 180)            # hour angle
    la = math.radians(lat)
    cos_zen = (math.sin(la) * math.sin(decl)
               + math.cos(la) * math.cos(decl) * math.cos(ha))
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    el = 90.0 - math.degrees(zen)
    az = math.degrees(math.acos(max(-1.0, min(1.0,
        (math.sin(decl) - math.sin(la) * cos_zen)
        / (math.cos(la) * math.sin(zen) + 1e-12)))))
    if ha > 0:
        az = 360.0 - az
    return az % 360.0, el


# ── DSM from the (full-scene) point cloud ────────────────────────────────────
def build_dsm(points: np.ndarray, transform, shape: Tuple[int, int]) -> np.ndarray:
    """Max-z surface on the chip grid. ``transform`` = the chip's rasterio
    Affine; ``points`` = (N,3) in the chip CRS (keep_all_classes=True!).
    Empty cells fall back to the scene's low percentile (ground-ish)."""
    h, w = shape
    inv = ~transform
    cols, rows = inv * (points[:, 0], points[:, 1])
    cols = np.floor(cols).astype(int)            # raster convention: floor
    rows = np.floor(rows).astype(int)
    ok = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    dsm = np.full(shape, -np.inf)
    np.maximum.at(dsm, (rows[ok], cols[ok]), points[ok, 2])
    floor = float(np.percentile(points[:, 2], 2))
    dsm[~np.isfinite(dsm)] = floor
    return dsm


# ── ray casting ──────────────────────────────────────────────────────────────
def cast_shadows(dsm: np.ndarray, res_m: float, az_deg: float, el_deg: float,
                 max_dist_m: float = 60.0, clearance_m: float = 0.35) -> np.ndarray:
    """Bool mask: True where the surface is shadowed for the given sun.

    March toward the sun; a cell is shadowed if any surface along the ray
    rises above the sight line (slope tan(elevation))."""
    az = math.radians(az_deg)
    dx, dy = math.sin(az), math.cos(az)          # toward the sun (x=E, y=N)
    drop = math.tan(math.radians(max(el_deg, 1.0))) * res_m
    steps = max(1, int(max_dist_m / res_m))
    h, w = dsm.shape
    shadow = np.zeros(dsm.shape, bool)
    for k in range(1, steps + 1):
        oc = int(round(k * dx))                  # +east = +col
        orow = int(round(-k * dy))               # +north = -row
        shifted = np.full(dsm.shape, -np.inf)
        src_r = slice(max(0, orow), min(h, h + orow))
        src_c = slice(max(0, oc), min(w, w + oc))
        dst_r = slice(max(0, -orow), min(h, h - orow))
        dst_c = slice(max(0, -oc), min(w, w - oc))
        shifted[dst_r, dst_c] = dsm[src_r, src_c]
        shadow |= (shifted - k * drop) > (dsm + clearance_m)
    return shadow


# ── fit the sun from the image itself ────────────────────────────────────────
def fit_sun(dsm: np.ndarray, chip_rgb: np.ndarray, res_m: float,
            az_step: float = 15.0, els=(25.0, 40.0, 55.0, 70.0)):
    """Solve (azimuth, elevation) by maximizing dark-pixel agreement.

    Returns (az, el, score, mask). Score = mean darkness inside the predicted
    mask minus outside (higher = better; ~0 = no shadow signal)."""
    gray = chip_rgb.astype(float).mean(axis=2)
    gray = (gray - gray.min()) / (np.ptp(gray) + 1e-6)
    dark = 1.0 - gray
    best = (None, None, -np.inf, None)
    for el in els:
        for az in np.arange(0.0, 360.0, az_step):
            m = cast_shadows(dsm, res_m, float(az), float(el))
            if m.sum() < 50 or m.mean() > 0.8:
                continue
            score = float(dark[m].mean() - dark[~m].mean())
            if score > best[2]:
                best = (float(az), float(el), score, m)
    if best[0] is not None:
        logger.info("fit_sun: az=%.0f el=%.0f score=%.3f (mask %.0f%%)",
                    best[0], best[1], best[2], 100 * best[3].mean())
    return best


def predicted_shadow_mask(chip_rgb: np.ndarray, transform, scene_points: np.ndarray,
                          when_utc=None, lat: Optional[float] = None,
                          lon: Optional[float] = None):
    """Chip + full-scene LiDAR -> (shadow_mask, info dict).

    With a capture datetime (+lat/lon) the sun is computed; otherwise it is
    FITTED from the image. ``scene_points`` must include vegetation/ground
    (fetch_roof_points(..., keep_all_classes=True))."""
    res = abs(transform.a)
    dsm = build_dsm(scene_points, transform, chip_rgb.shape[:2])
    if when_utc is not None and lat is not None:
        az, el = sun_position(lat, lon, when_utc)
        mask = cast_shadows(dsm, res, az, el)
        gray = chip_rgb.astype(float).mean(axis=2)
        gray = (gray - gray.min()) / (np.ptp(gray) + 1e-6)
        score = float((1 - gray)[mask].mean() - (1 - gray)[~mask].mean()) \
            if mask.any() and (~mask).any() else 0.0
        return mask, {"az": az, "el": el, "score": score, "source": "timestamp"}
    az, el, score, mask = fit_sun(dsm, chip_rgb, res)
    return mask, {"az": az, "el": el, "score": score, "source": "fitted"}
