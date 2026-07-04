"""LiDAR pitch annotation for SAM facets — read-only fusion.

"SAM for shape, LiDAR for pitch": the facet polygons and roof outline are
FROZEN by the time this module runs. For each facet we only *read* the LiDAR
heights inside its polygon, fit one plane, and write numbers onto the facet:

    pitch_string ("6:12"), slope_deg, aspect_bin, surface_area_m2, is_flat

Nothing here can move a boundary, split, or merge a facet — the data flow is
one-directional (segmentation -> annotation), which is what keeps the good SAM
facets safe. A facet with too few points or a failed fit simply keeps
pitch "unspecified" (today's report is the guaranteed floor).

Asking LiDAR "how tilted is this known polygon?" is the easy question it was
always good at (Holland Ln baseline: −3.7% vs Roofr) — unlike boundary
*drawing*, which fragmented and drove the pivot to SAM.

CRS contract: ``points`` x/y must be in the SAME planar CRS as the facet
polygons (the NAIP/LiDAR UTM the pipeline aligns to).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from src.roofs.metrics import (
    FLAT_SLOPE_DEG,
    compute_aspect_bin,
    compute_aspect_deg,
    compute_pitch_string,
    compute_slope_deg,
    compute_surface_area,
)
from src.roofs.plane_fit import fit_plane_ransac

logger = logging.getLogger(__name__)

MIN_FACET_POINTS = 30          # below this a plane fit is noise, not signal


def _xyz(points) -> np.ndarray:
    """Structured (x,y,z) or (N,3) array -> plain float (N,3)."""
    if hasattr(points, "dtype") and points.dtype.names:
        return np.column_stack([points["x"], points["y"], points["z"]]).astype(float)
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("points must be structured (x,y,z) or an (N,3) array")
    return arr[:, :3]


def annotate_facets_with_lidar(
    facets: List,
    points,
    min_points: int = MIN_FACET_POINTS,
) -> Dict[int, dict]:
    """Per-facet pitch annotation. Returns {facet_id: annotation} — facets that
    can't be annotated are simply absent (they keep "unspecified" downstream).

    Args:
        facets: SAM facets (``.facet_id``, ``.polygon`` in the points' CRS).
        points: LiDAR roof points, structured (x,y,z) or (N,3).
    """
    from shapely import contains_xy

    xyz = _xyz(points)
    out: Dict[int, dict] = {}
    for f in facets:
        poly = getattr(f, "polygon", None)
        if poly is None or poly.is_empty:
            continue
        inside = contains_xy(poly, xyz[:, 0], xyz[:, 1])
        n = int(inside.sum())
        if n < min_points:
            logger.info("facet %s: %d LiDAR pts (<%d) — leaving unspecified",
                        f.facet_id, n, min_points)
            continue
        try:
            plane = fit_plane_ransac(xyz[inside])
        except Exception as e:  # noqa: BLE001 — annotation is best-effort
            logger.warning("facet %s: plane fit failed (%s)", f.facet_id, e)
            continue
        if not plane.success:
            continue
        slope = compute_slope_deg(plane)
        is_flat = slope < FLAT_SLOPE_DEG
        aspect = compute_aspect_deg(plane)
        out[f.facet_id] = {
            "slope_deg": float(slope),
            "pitch_string": compute_pitch_string(slope),
            "aspect_bin": compute_aspect_bin(aspect),
            "is_flat": bool(is_flat),
            "surface_area_m2": float(
                compute_surface_area(poly.area, 0.0 if is_flat else slope)),
            "n_points": n,
            "residual_m": float(plane.residual_median),
        }
    logger.info("LiDAR annotated %d/%d facet(s)", len(out), len(facets))
    return out


def fuse_into_report_input(report_input: dict,
                           annotations: Dict[int, dict]) -> dict:
    """Fill the pitch fields sam_report left as None. Geometry untouched:
    only per-facet numbers change; facets without an annotation keep
    "unspecified". Returns the same dict (mutated) for chaining."""
    for f in report_input.get("facets", []):
        ann = annotations.get(f.get("facet_id"))
        if not ann:
            continue
        f["pitch_string"] = ann["pitch_string"]
        f["slope_deg"] = ann["slope_deg"]
        f["aspect_bin"] = ann["aspect_bin"]
        f["is_flat"] = ann["is_flat"]
        f["surface_area_m2"] = ann["surface_area_m2"]
    return report_input
