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


TWO_STORY_LEVEL_STEP_M = 2.4   # a facet a story-height ABOVE the building's
                               # lowest eave = an upper roof level. Relative,
                               # not absolute: a tall single-story building
                               # (Holland benchmark) must read 0 — Roofr's
                               # semantics (roof-over-roof access), not height.


def annotate_facets_with_lidar(
    facets: List,
    points,
    min_points: int = MIN_FACET_POINTS,
    ground_z: Optional[float] = None,
) -> Dict[int, dict]:
    """Per-facet pitch annotation. Returns {facet_id: annotation} — facets that
    can't be annotated are simply absent (they keep "unspecified" downstream).

    Args:
        facets: SAM facets (``.facet_id``, ``.polygon`` in the points' CRS).
        points: LiDAR roof points, structured (x,y,z) or (N,3).
        ground_z: optional ground elevation (m, same datum as the points).
            When given, each facet gets ``eave_height_m`` (5th-percentile roof
            z minus ground) and ``is_two_story``.
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
            "aspect_deg": float(aspect),        # downslope compass deg (rake relabel)
            "grad": (float(plane.a), float(plane.b)),   # plane gradient (3D lengths, merging)
            "aspect_bin": compute_aspect_bin(aspect),
            "is_flat": bool(is_flat),
            "surface_area_m2": float(
                compute_surface_area(poly.area, 0.0 if is_flat else slope)),
            "n_points": n,
            "residual_m": float(plane.residual_median),
        }
        # eave elevation = the facet's low edge (5th percentile rides outliers)
        out[f.facet_id]["_eave_z"] = float(np.percentile(xyz[inside, 2], 5))

    # two-story = a roof LEVEL above the building's lowest eave (relative —
    # Roofr semantics). Needs either >=2 facets (levels comparable) or ground.
    if out and (ground_z is not None or len(out) >= 2):
        base = min(a["_eave_z"] for a in out.values())
        for fid, a in sorted(out.items()):
            logger.info("facet %s: eave z=%.2f (rel %.2f m)%s%s", fid,
                        a["_eave_z"], a["_eave_z"] - base,
                        " flat" if a.get("is_flat") else "",
                        f" ground={ground_z:.2f}" if ground_z is not None else "")
        for a in out.values():
            rel = a["_eave_z"] - base
            a["is_two_story"] = bool(rel >= TWO_STORY_LEVEL_STEP_M)
            a["eave_height_m"] = (a["_eave_z"] - float(ground_z)
                                  if ground_z is not None else rel)
    for a in out.values():
        a.pop("_eave_z", None)
    logger.info("LiDAR annotated %d/%d facet(s)", len(out), len(facets))
    return out


def merge_coplanar_facets(
    facets: List,
    points,
    annotations: Dict[int, dict],
    angle_tol_deg: float = 5.0,
    residual_max: float = 0.12,
    touch_tol: float = 0.5,
):
    """Merge ADJACENT facets that LiDAR proves are the same physical plane.

    Fixes model over-segmentation (a gable slope split into parallel bands)
    with evidence, not heuristics: two facets merge only if (a) they touch,
    (b) their fitted normals agree within ``angle_tol_deg``, (c) both share
    flat/pitched status, and (d) a SINGLE plane refit over their combined
    points is tight (median residual < ``residual_max`` and >=70% inliers) —
    which rejects same-slope roofs at different heights (a step/parapet).
    Merging conserves area exactly (polygon union). Returns (facets, changed).
    """
    import math

    from shapely import contains_xy
    from shapely.ops import unary_union

    from src.roofs.segment import Facet

    xyz = _xyz(points)
    by_id = {f.facet_id: f for f in facets}
    ids = [f.facet_id for f in facets]
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def normal(g):
        n = np.array([-g[0], -g[1], 1.0])
        return n / np.linalg.norm(n)

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = by_id[ids[i]], by_id[ids[j]]
            aa, ab = annotations.get(a.facet_id), annotations.get(b.facet_id)
            if not aa or not ab or aa["is_flat"] != ab["is_flat"]:
                continue
            ang = math.degrees(math.acos(min(1.0, float(
                normal(aa["grad"]) @ normal(ab["grad"])))))
            if ang > angle_tol_deg:
                continue
            if a.polygon.distance(b.polygon) > touch_tol:
                continue
            m = (contains_xy(a.polygon, xyz[:, 0], xyz[:, 1])
                 | contains_xy(b.polygon, xyz[:, 0], xyz[:, 1]))
            if int(m.sum()) < MIN_FACET_POINTS:
                continue
            try:
                pl = fit_plane_ransac(xyz[m])
            except Exception:  # noqa: BLE001
                continue
            if (not pl.success or pl.residual_median > residual_max
                    or pl.inlier_count < 0.7 * int(m.sum())):
                continue                       # e.g. same slope, different height
            parent[find(ids[j])] = find(ids[i])

    groups: Dict[int, list] = {}
    for fid in ids:
        groups.setdefault(find(fid), []).append(fid)
    if all(len(g) == 1 for g in groups.values()):
        return list(facets), False

    merged: List = []
    for new_id, members in enumerate(groups.values(), start=1):
        poly = unary_union([by_id[m].polygon.buffer(0.05) for m in members]).buffer(-0.05)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        merged.append(Facet(facet_id=new_id, points=None, label=new_id, polygon=poly))
    logger.info("plane merge: %d facet(s) -> %d (LiDAR coplanarity)",
                len(facets), len(merged))
    return merged, True


def absorb_unannotated_orphans(
    facets: List,
    annotations: Dict[int, dict],
    max_frac: float = 0.05,
    touch_tol: float = 0.5,
):
    """Absorb SMALL unverifiable facets into their touching annotated neighbor.

    A facet with no LiDAR annotation (too few points — shadow slivers, canopy
    gaps, tiling spillover) can't be measured OR merged by evidence, and its
    seams pollute the edge graph ('unspecified' footage). If it's small
    (< max_frac of the roof) and touches an annotated facet, fold it into the
    neighbor sharing the longest boundary. LARGE unannotated facets are kept —
    deleting significant area would be dishonest. Returns (facets, changed).
    """
    from shapely.ops import unary_union

    from src.roofs.segment import Facet

    total = sum(f.polygon.area for f in facets
                if f.polygon is not None and not f.polygon.is_empty)
    if total <= 0:
        return list(facets), False
    orphans = [f for f in facets if f.facet_id not in annotations
               and f.polygon is not None and not f.polygon.is_empty
               and f.polygon.area < max_frac * total]
    if not orphans:
        return list(facets), False

    keep = {f.facet_id: f.polygon for f in facets
            if f not in orphans and f.polygon is not None}
    absorbed = 0
    for o in orphans:
        best_id, best_len = None, 0.0
        for fid, poly in keep.items():
            if fid not in annotations:
                continue
            if o.polygon.distance(poly) > touch_tol:
                continue
            shared = o.polygon.buffer(touch_tol).intersection(poly).area
            if shared > best_len:
                best_id, best_len = fid, shared
        if best_id is None:
            keep[o.facet_id] = o.polygon        # nothing to join — keep it
            continue
        merged = unary_union([keep[best_id].buffer(0.05),
                              o.polygon.buffer(0.05)]).buffer(-0.05)
        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda g: g.area)
        keep[best_id] = merged
        absorbed += 1
    if not absorbed:
        return list(facets), False
    out = [Facet(facet_id=i, points=None, label=i, polygon=p)
           for i, (fid, p) in enumerate(sorted(keep.items()), start=1)]
    logger.info("absorbed %d unverifiable orphan facet(s) into neighbors",
                absorbed)
    return out, True


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
        if "is_two_story" in ann:
            f["two_story"] = ann["is_two_story"]
            f["eave_height_m"] = ann["eave_height_m"]
    return report_input
