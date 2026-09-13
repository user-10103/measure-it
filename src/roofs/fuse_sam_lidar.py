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
from src.roofs.plane_fit import DEFAULT_MIN_INLIER_RATIO, fit_plane_ransac

logger = logging.getLogger(__name__)

MIN_FACET_POINTS = 30          # below this a plane fit is noise, not signal
MIN_ROOF_CLEARANCE_M = 1.0     # points within 1 m of ground are bare-earth / low
                               # vegetation, not roof — exclude from fit + eave
DENSE_FACET_POINTS = 80        # >= this points, use the full 25% inlier floor;
SPARSE_INLIER_RATIO = 0.15     # smaller facets (3DEP sparsity) get a relaxed floor
MIN_PLANE_INLIERS = 20         # ...but a RATIO alone is not enough: 36 points at
                               # 41.7% is 15 inliers, which cannot define a plane.
                               # 1600 Sarno produced a 59.8 deg "facet" that way on
                               # an 8 deg roof, and it polluted the edge graph.

# split_multiplane_facets thresholds (the complement to merge_coplanar_facets)
SPLIT_RESIDUAL_M = 0.30        # a point this far off the primary plane is "off it"
SPLIT_MIN_OFF_FRAC = 0.25      # facet is multiplane only if this many points are off
SPLIT_ANGLE_DEG = 15.0         # ... and the second plane differs by at least this
SPLIT_MIN_PIECE_FRAC = 0.15    # reject a cut that shaves a sliver — both pieces
                               # must be >= this fraction of the facet area
FLAT_LEVEL_STEP_M = 0.60       # two flat sections this far apart in elevation are
                               # different roof LEVELS, not one plane with clutter
FLAT_LEVEL_MIN_FRAC = 0.25     # ...and each level needs this share of the points,
                               # so an HVAC cluster is not mistaken for a level
FLAT_ACCEPT_RESIDUAL_M = 0.30  # accept a sub-floor fit as flat (0:12) only if it's
                               # near-level AND this tight (real flat roof, not noise)


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


def _attribution_report(facets, xyz, annotated: int, n_facets: int) -> str:
    """Why did LiDAR points fail to land in the facets? Compare WHERE THE POINTS ARE
    with WHERE THE FACETS ARE — both already in the caller's CRS — and name the cause.

    The failure this exists for (1600 Sarno Rd): the EPT fetch reported "550 roof
    point(s) for the footprint" and then 0 of 6 facets were annotated. That is
    indistinguishable, from the existing logs, between three very different bugs:

      * BAD DECLARED SRS in the EPT header — the footprint clip in ept_fetch happens
        in EPT space and succeeds even when the header's SRS is wrong (the footprint
        is transformed into the same wrong space), so the error only appears after
        the reprojection to the chip CRS. Signature: centroids kilometres apart.
      * WRONG BUILDING — the geocode pin sat 22 m from the selected footprint, so
        points were queried for one structure and facets segmented on another.
        Signature: centroids tens of metres apart, bboxes similar in size.
      * NEITHER — the points really are over the roof but fall between the facet
        polygons. Signature: the bboxes overlap.

    Bounding boxes (not unions) keep this cheap; it only runs off the happy path.
    """
    px0, py0 = float(xyz[:, 0].min()), float(xyz[:, 1].min())
    px1, py1 = float(xyz[:, 0].max()), float(xyz[:, 1].max())
    bounds = [f.polygon.bounds for f in facets
              if getattr(f, "polygon", None) is not None and not f.polygon.is_empty]
    fx0 = min(b[0] for b in bounds); fy0 = min(b[1] for b in bounds)
    fx1 = max(b[2] for b in bounds); fy1 = max(b[3] for b in bounds)
    pcx, pcy = (px0 + px1) / 2.0, (py0 + py1) / 2.0
    fcx, fcy = (fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0
    dist = float(np.hypot(pcx - fcx, pcy - fcy))
    overlap = (px0 <= fx1 and fx0 <= px1 and py0 <= fy1 and fy0 <= py1)
    if dist > 1000.0:
        verdict = (">1km apart -> suspect EPT header SRS / reprojection "
                   "(points landed in the wrong coordinate space)")
    elif overlap:
        verdict = ("bboxes overlap -> points are over the roof but fall between the "
                   "facet polygons (not a placement error)")
    else:
        verdict = ("~tens of metres apart -> suspect wrong building / footprint "
                   "offset (points queried for a different structure)")
    return ("LiDAR attribution: %d/%d facet(s) annotated from %d point(s)\n"
            "  points bbox=(%.1f, %.1f, %.1f, %.1f) centroid=(%.1f, %.1f)\n"
            "  facets bbox=(%.1f, %.1f, %.1f, %.1f) centroid=(%.1f, %.1f)\n"
            "  centroid distance = %.1f m -> %s"
            % (annotated, n_facets, len(xyz),
               px0, py0, px1, py1, pcx, pcy,
               fx0, fy0, fx1, fy1, fcx, fcy,
               dist, verdict))


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
        pts = xyz[inside]
        # Drop near-ground returns (bare earth, driveway, low vegetation) that
        # bleed into the polygon: they drag the plane fit below the inlier floor
        # and pull the eave elevation down to ground (the observed "eave = ground"
        # contamination). Roof surfaces sit well above grade.
        if ground_z is not None and len(pts):
            roof_only = pts[pts[:, 2] > ground_z + MIN_ROOF_CLEARANCE_M]
            if len(roof_only) >= min_points:
                pts = roof_only
        n = len(pts)
        if n < min_points:
            logger.info("facet %s: %d LiDAR pts (<%d) — leaving unspecified",
                        f.facet_id, n, min_points)
            continue
        # Area-aware inlier floor: a small facet at 3DEP density (~2-8 pts/m^2)
        # has too few points for a 25% floor to be meaningful — relax it (it still
        # needs a coherent core plane). Recovers pitch on tiny facets that were
        # dropped and left silently unspecified.
        floor = DEFAULT_MIN_INLIER_RATIO if n >= DENSE_FACET_POINTS else SPARSE_INLIER_RATIO
        try:
            plane = fit_plane_ransac(pts, min_inlier_ratio=floor)
        except Exception as e:  # noqa: BLE001 — annotation is best-effort
            logger.warning("facet %s: plane fit failed (%s)", f.facet_id, e)
            continue
        # A ratio can clear the floor on a handful of points; require an absolute
        # inlier count too, or a noise plane fit to ~15 points is treated as roof
        # geometry and reaches the edge classifier.
        if plane.success and plane.inlier_count < MIN_PLANE_INLIERS:
            logger.info("facet %s: plane fit has only %d inlier(s) (<%d) — "
                        "leaving unspecified", f.facet_id, plane.inlier_count,
                        MIN_PLANE_INLIERS)
            continue
        if not plane.success:
            # A flat roof with rooftop clutter (HVAC, parapets, ponding) rarely
            # clears the inlier floor at a 0.25 m RANSAC threshold, yet its near-
            # LEVEL best-fit plane with a tight residual IS the 0:12 answer. Accept
            # that; anything else stays honestly unspecified (-> needs_review).
            if not (compute_slope_deg(plane) < FLAT_SLOPE_DEG
                    and plane.inlier_count >= min_points
                    and plane.residual_median < FLAT_ACCEPT_RESIDUAL_M):
                continue
            logger.info("facet %s: accepted as flat (%.0f%% inliers, level fit)",
                        f.facet_id, 100 * plane.inlier_count / n)
        slope = compute_slope_deg(plane)
        is_flat = slope < FLAT_SLOPE_DEG
        aspect = compute_aspect_deg(plane)
        out[f.facet_id] = {
            "slope_deg": float(slope),
            # Report the SAME slope the area math used. A facet at 4.76 deg is
            # under FLAT_SLOPE_DEG, so surface_area applies no multiplier (below
            # it "the slope is membrane-roof LiDAR noise") - but this printed
            # "1:12" anyway, so 755 E Eau Gallie showed 2,481 sqft under a 1/12
            # pitch row while the same area was counted as flat and excluded from
            # pitched area. If the slope is not trusted enough to use, it is not
            # trusted enough to print.
            "pitch_string": compute_pitch_string(0.0 if is_flat else slope),
            "aspect_deg": float(aspect),        # downslope compass deg (rake relabel)
            "grad": (float(plane.a), float(plane.b)),   # plane gradient (3D lengths, merging)
            "aspect_bin": compute_aspect_bin(aspect),
            "is_flat": bool(is_flat),
            "surface_area_m2": float(
                compute_surface_area(poly.area, 0.0 if is_flat else slope)),
            "n_points": n,
            # residual_median is computed over INLIERS ONLY, so the worse a fit
            # is, the better this number looks - it reports on the points the fit
            # already agreed with. explained_frac is the honest companion: how
            # much of the facet the accepted plane actually accounts for.
            "residual_m": float(plane.residual_median),
            "explained_frac": float(plane.inlier_count) / max(n, 1),
            # elevation of this facet, so the edge classifier can tell a PARAPET
            # (two flat sections at different heights) from a mere transition
            "median_z": float(np.median(pts[:, 2])),
        }
        # eave elevation = the facet's low edge (5th percentile rides outliers);
        # computed on the ground-filtered points so a driveway can't pull it down.
        out[f.facet_id]["_eave_z"] = float(np.percentile(pts[:, 2], 5))

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
    # Points were supplied but (almost) nothing stuck -> say WHY, once, decisively.
    # WARNING when NOTHING was annotated (always a real bug). INFO when under half
    # stuck, because partial attribution is routine on sparse 3DEP (~1-2 pts/m^2)
    # and would otherwise cry wolf on every small facet.
    n_facets = sum(1 for f in facets
                   if getattr(f, "polygon", None) is not None and not f.polygon.is_empty)
    if len(xyz) and n_facets and len(out) * 2 < n_facets:
        report = _attribution_report(facets, xyz, len(out), n_facets)
        (logger.info if out else logger.warning)(report)
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


def split_multiplane_facets(facets: List, points):
    """Split a facet whose LiDAR points fit TWO distinct planes into one facet per
    plane — the complement to ``merge_coplanar_facets``, fixing model UNDER-
    segmentation (a hip wing returned as one blob). Evidence-based and
    conservative: a facet splits only when a large minority of its points are off
    its primary plane AND those points form a second plane at a clear angle, and
    the geometric cut (the two planes' line of intersection) yields two
    substantial pieces. Area is conserved (polygon split). Returns (facets, changed).
    """
    import math

    from shapely import contains_xy
    from shapely.geometry import LineString
    from shapely.ops import split as shp_split

    from src.roofs.segment import Facet

    xyz = _xyz(points)
    out: List = []
    changed = False
    for f in facets:
        poly = getattr(f, "polygon", None)
        if poly is None or poly.is_empty:
            out.append(f)
            continue
        inside = contains_xy(poly, xyz[:, 0], xyz[:, 1])
        pts = xyz[inside]
        if len(pts) < 2 * MIN_FACET_POINTS:            # need enough for two planes
            out.append(f)
            continue
        try:
            p1 = fit_plane_ransac(pts)
        except Exception:  # noqa: BLE001
            out.append(f)
            continue
        if not p1.success:
            out.append(f)
            continue
        if compute_slope_deg(p1) < FLAT_SLOPE_DEG:
            out.append(f)                              # flat roof is ONE plane —
            continue                                   # residual is clutter, not a 2nd facet
        resid = np.abs(pts[:, 2] - (p1.a * pts[:, 0] + p1.b * pts[:, 1] + p1.c))
        off = resid > SPLIT_RESIDUAL_M
        if off.sum() < max(MIN_FACET_POINTS, SPLIT_MIN_OFF_FRAC * len(pts)):
            out.append(f)                              # essentially one plane
            continue
        try:
            p2 = fit_plane_ransac(pts[off])
        except Exception:  # noqa: BLE001
            out.append(f)
            continue
        n1, n2 = np.array(p1.normal), np.array(p2.normal)
        ang = math.degrees(math.acos(min(1.0, abs(float(n1 @ n2)))))
        if not p2.success or ang < SPLIT_ANGLE_DEG:
            out.append(f)                              # second "plane" too similar
            continue

        # crease = the two planes' line of intersection, projected to xy:
        #   (a1-a2)x + (b1-b2)y + (c1-c2) = 0
        da, db, dc = p1.a - p2.a, p1.b - p2.b, p1.c - p2.c
        if da == 0 and db == 0:
            out.append(f)
            continue
        cx, cy = poly.centroid.x, poly.centroid.y
        t = (da * cx + db * cy + dc) / (da * da + db * db)
        fx, fy = cx - t * da, cy - t * db              # foot of centroid on line
        norm = math.hypot(da, db)
        ux, uy = -db / norm, da / norm                 # along-line unit direction
        minx, miny, maxx, maxy = poly.bounds
        span = 2.0 * math.hypot(maxx - minx, maxy - miny)
        line = LineString([(fx - span * ux, fy - span * uy),
                           (fx + span * ux, fy + span * uy)])
        try:
            pieces = [g for g in shp_split(poly, line).geoms
                      if g.geom_type == "Polygon" and g.area > 0]
        except Exception:  # noqa: BLE001
            out.append(f)
            continue
        # A clean two-plane facet cuts into exactly two substantial pieces. More
        # pieces means the line raked across a concave boundary (a messy cut), not
        # a real crease.
        if len(pieces) != 2 or min(g.area for g in pieces) < SPLIT_MIN_PIECE_FRAC * poly.area:
            out.append(f)
            continue
        # Each piece must carry enough LiDAR points to fit its OWN plane. Without
        # this, splitting a sparsely-sampled facet (e.g. 1.3 pts/m^2 3DEP) just
        # manufactures sub-floor pieces that all come back pitch-less — the Tampa
        # regression: one flat facet -> eight unspecified slivers.
        if any(int(contains_xy(g, xyz[:, 0], xyz[:, 1]).sum()) < MIN_FACET_POINTS
               for g in pieces):
            logger.info("facet %s: no level split — a piece would hold < %d points",
                        f.facet_id, MIN_FACET_POINTS)
            out.append(f)
            continue
        logger.info("facet %s split into %d planes (normals %.0f° apart)",
                    f.facet_id, len(pieces), ang)
        out.extend(Facet(facet_id=-1, points=None, label=-1, polygon=g) for g in pieces)
        changed = True

    if not changed:
        return list(facets), False
    # renumber to a clean sequential partition (like merge_coplanar_facets)
    final = [Facet(facet_id=i, points=None, label=i, polygon=f.polygon)
             for i, f in enumerate(out, start=1)]
    logger.info("plane split: %d facet(s) -> %d (LiDAR multiplane)",
                len(facets), len(final))
    return final, True


def _spans_two_levels(pts) -> bool:
    """True when a FLAT facet's points sit at two distinct elevations.

    A flat facet cannot be "two planes at an angle", so the slope test that finds
    under-segmentation on a pitched roof is blind to it. But a large commercial
    roof is routinely several flat sections at DIFFERENT HEIGHTS, separated by
    parapets or level changes — and reporting them as one plane loses every
    internal edge. 2725 Judge Fran shipped 43,029 sqft as a single facet with
    zero ridges, hips, valleys, parapets or transitions, and passed the gate,
    because nothing was looking for a step.

    Trim the tails before measuring the gap: rooftop units and noise live there,
    and they are not a roof level. Both sides must carry a real share of the
    points, which is what separates a second SECTION from an HVAC cluster.
    """
    clusters = _level_clusters(pts)
    if clusters is None:
        return False
    # A step alone is not a second SECTION: rooftop plant sits above the deck
    # through the same plan area. Only count it when the two levels occupy
    # different ground.
    return _levels_side_by_side(*clusters)


CLUTTER_OVERLAP_MAX = 0.50     # if this much of the smaller cluster's plan hull
                               # sits inside the other's, the two elevations are
                               # SUPERIMPOSED (rooftop plant above the deck), not
                               # two sections side by side


def _levels_side_by_side(low, high) -> bool:
    """Do two elevation clusters occupy DIFFERENT ground, or the same ground?

    A real level change is two sections meeting at a parapet - their plan hulls
    sit beside each other. Rooftop plant (a mechanical unit, a stair bulkhead, a
    canopy) is SUPERIMPOSED: LiDAR sees the deck and something 3 m above it
    through the same plan area, so the hulls overlap almost completely.

    Measured on 755 E Eau Gallie, facets 5 and 6: a clean 3.4-3.7 m step, both
    clusters well over MIN_FACET_POINTS, and hulls summing to 1.79 and 1.59 of a
    polygon that is 1.00. Nothing to cut between - and those same facets had the
    HIGHEST explained fractions on the roof (0.88, 0.97), because once RANSAC
    drops the plant what remains is an excellent plane. Flagging them as
    under-segmented was a false positive.
    """
    from shapely.geometry import MultiPoint

    try:
        hull_lo = MultiPoint([tuple(q) for q in low[:, :2]]).convex_hull
        hull_hi = MultiPoint([tuple(q) for q in high[:, :2]]).convex_hull
    except Exception:  # noqa: BLE001
        return False
    smaller = min(hull_lo.area, hull_hi.area)
    if smaller <= 0:
        return False
    return (hull_lo.intersection(hull_hi).area / smaller) < CLUTTER_OVERLAP_MAX


def _level_clusters(pts):
    """Split a facet's points at its elevation gap -> (low, high) or None."""
    z = np.sort(pts[:, 2])
    lo, hi = int(0.05 * len(z)), int(0.95 * len(z))
    core = z[lo:hi]
    if len(core) < 2 * MIN_PLANE_INLIERS:
        return None
    gaps = np.diff(core)
    k = int(np.argmax(gaps))
    if gaps[k] < FLAT_LEVEL_STEP_M:
        return None
    cut = (core[k] + core[k + 1]) / 2.0
    low, high = pts[pts[:, 2] <= cut], pts[pts[:, 2] > cut]
    need = max(MIN_PLANE_INLIERS, int(FLAT_LEVEL_MIN_FRAC * len(core)))
    if min(len(low), len(high)) < need:
        return None
    return low, high


def split_level_facets(facets: List, points):
    """Split a facet whose points sit at two roof LEVELS into one facet per level.

    The complement to split_multiplane_facets for FLAT roofs. That one cuts along
    the line where two planes intersect, which does not exist here: a commercial
    roof's sections are PARALLEL, separated by a step and a parapet rather than a
    crease. 2725 Judge Fran came back as 43,029 sqft in a single facet with no
    internal edges at all, because nothing was looking for a step.

    The cut is the perpendicular bisector between the two clusters' XY centroids —
    the natural divide between two adjacent sections. Guards mirror the angle
    split: exactly two pieces, both substantial, both carrying real point support,
    or the facet is left whole. Area is conserved (polygon split).
    """
    import math

    from shapely import contains_xy
    from shapely.geometry import LineString
    from shapely.ops import split as shp_split

    from src.roofs.segment import Facet

    xyz = _xyz(points)
    out: List = []
    changed = False
    for f in facets:
        poly = getattr(f, "polygon", None)
        if poly is None or poly.is_empty:
            out.append(f)
            continue
        pts = xyz[contains_xy(poly, xyz[:, 0], xyz[:, 1])]
        if len(pts) < 2 * MIN_FACET_POINTS:
            logger.debug("facet %s: no level split — %d pts (<%d)", f.facet_id,
                         len(pts), 2 * MIN_FACET_POINTS)
            out.append(f)
            continue
        clusters = _level_clusters(pts)
        if clusters is None:
            logger.debug("facet %s: no level split — no elevation gap >= %.2f m "
                         "with support on both sides", f.facet_id, FLAT_LEVEL_STEP_M)
            out.append(f)
            continue
        low, high = clusters
        if not _levels_side_by_side(low, high):
            # SUPERIMPOSED: rooftop plant above the deck, not two sections. There
            # is nothing to cut between, and RANSAC already ignores it.
            logger.info("facet %s: two elevations but SUPERIMPOSED in plan — "
                        "rooftop plant, not a level change; left whole", f.facet_id)
            out.append(f)
            continue
        cx1, cy1 = float(low[:, 0].mean()), float(low[:, 1].mean())
        cx2, cy2 = float(high[:, 0].mean()), float(high[:, 1].mean())
        dx, dy = cx2 - cx1, cy2 - cy1
        sep = math.hypot(dx, dy)
        if sep < 1e-6:
            logger.info("facet %s: no level split — cluster centroids coincide",
                        f.facet_id)
            out.append(f)
            continue
        mx, my = (cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0  # midpoint of the centroids
        ux, uy = -dy / sep, dx / sep                   # perpendicular = the divide
        minx, miny, maxx, maxy = poly.bounds
        span = 2.0 * math.hypot(maxx - minx, maxy - miny)
        line = LineString([(mx - span * ux, my - span * uy),
                           (mx + span * ux, my + span * uy)])
        try:
            pieces = [g for g in shp_split(poly, line).geoms
                      if g.geom_type == "Polygon" and g.area > 0]
        except Exception as e:  # noqa: BLE001
            logger.info("facet %s: no level split — cut failed (%s)", f.facet_id, e)
            out.append(f)
            continue
        if len(pieces) != 2:
            logger.info("facet %s: no level split — cut yielded %d piece(s), not 2",
                        f.facet_id, len(pieces))
            out.append(f)
            continue
        if min(g.area for g in pieces) < SPLIT_MIN_PIECE_FRAC * poly.area:
            logger.info("facet %s: no level split — a piece is only %.0f%% of the "
                        "facet", f.facet_id,
                        100 * min(g.area for g in pieces) / poly.area)
            out.append(f)
            continue
        if any(int(contains_xy(g, xyz[:, 0], xyz[:, 1]).sum()) < MIN_FACET_POINTS
               for g in pieces):
            out.append(f)
            continue
        logger.info("facet %s split at a roof level change (step %.2f m)",
                    f.facet_id, abs(float(high[:, 2].mean() - low[:, 2].mean())))
        out.extend(Facet(facet_id=-1, points=None, label=-1, polygon=g) for g in pieces)
        changed = True

    if not changed:
        return list(facets), False
    final = [Facet(facet_id=i, points=None, label=i, polygon=g.polygon)
             for i, g in enumerate(out, start=1)]
    logger.info("level split: %d facet(s) -> %d", len(facets), len(final))
    return final, True


def detect_multiplane_facets(facets: List, points) -> List[int]:
    """Facet ids whose LiDAR points STILL support more than one plane.

    The world-class gate is structurally BLIND to under-segmentation. A roof
    returned as one big facet trivially satisfies facets_partition (nothing to
    overlap), facets_coverage (100% by construction) and edges_typed (the
    single-facet branch returns "ridge/hip N/A"). So a roof cut into too few
    planes passes every check while UNDER-REPORTING SURFACE AREA — and that is
    the failure that actually reaches a customer, because material is ordered off
    that number. A refusal is recoverable; a confident short number is not.

    This supplies the independent evidence the gate lacked. It runs the same
    two-plane test ``split_multiplane_facets`` uses, but on the FINAL facets —
    after splitting and coplanar merging — so what it reports is what the report
    actually ships. split declines some cuts deliberately (a point-starved piece,
    a messy cut across a concave boundary); those are precisely the facets that
    stay under-segmented invisibly.

    Flat-primary facets are skipped for the same reason split skips them: on a
    flat commercial roof the off-plane returns are rooftop clutter, not a second
    roof plane, and flagging them would cry wolf on every legitimate flat roof.
    """
    import math

    from shapely import contains_xy

    xyz = _xyz(points)
    flagged: List[int] = []
    for f in facets:
        poly = getattr(f, "polygon", None)
        if poly is None or poly.is_empty:
            continue
        pts = xyz[contains_xy(poly, xyz[:, 0], xyz[:, 1])]
        if len(pts) < 2 * MIN_FACET_POINTS:
            continue                                   # too sparse to judge
        try:
            p1 = fit_plane_ransac(pts)
        except Exception:  # noqa: BLE001
            continue
        if not p1.success:
            continue
        # A STEP in elevation means separate sections, checked BEFORE the slope
        # tests rather than only on flat facets. Two flat levels side by side fit
        # as a shallow RAMP — the 2725 Judge Fran geometry fits at 5.4 deg, just
        # over FLAT_SLOPE_DEG — so gating this on "is flat" lets the case dodge
        # both tests, which is how 43,029 sqft shipped as a single plane with no
        # internal edges. A genuinely pitched roof has continuously varying z and
        # no such gap, so this does not fire on it.
        if _spans_two_levels(pts):
            logger.warning(
                "facet %s spans two roof LEVELS per LiDAR — separate sections "
                "reported as one plane, losing the parapet / level-change edges "
                "between them", f.facet_id)
            flagged.append(f.facet_id)
            continue
        if compute_slope_deg(p1) < FLAT_SLOPE_DEG:
            continue                                   # flat, single level
        resid = np.abs(pts[:, 2] - (p1.a * pts[:, 0] + p1.b * pts[:, 1] + p1.c))
        off = resid > SPLIT_RESIDUAL_M
        if off.sum() < max(MIN_FACET_POINTS, SPLIT_MIN_OFF_FRAC * len(pts)):
            continue                                   # essentially one plane
        try:
            p2 = fit_plane_ransac(pts[off])
        except Exception:  # noqa: BLE001
            continue
        if not p2.success or p2.inlier_count < MIN_PLANE_INLIERS:
            continue
        n1, n2 = np.array(p1.normal), np.array(p2.normal)
        ang = math.degrees(math.acos(min(1.0, abs(float(n1 @ n2)))))
        if ang < SPLIT_ANGLE_DEG:
            continue
        logger.warning(
            "facet %s spans >1 plane per LiDAR (%d of %d points off the primary "
            "plane form a second at %.0f deg) — roof is under-segmented and its "
            "surface area is under-reported", f.facet_id, int(off.sum()),
            len(pts), ang)
        flagged.append(f.facet_id)
    return flagged


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
        if "explained_frac" in ann:
            f["explained_frac"] = ann["explained_frac"]
        if "is_two_story" in ann:
            f["two_story"] = ann["is_two_story"]
            f["eave_height_m"] = ann["eave_height_m"]

    # Honesty guard: a facet whose plane fit failed (no annotation -> slope still
    # None, not flat) must be FLAGGED, never left silently unspecified — else the
    # report prints its plan area as if it were surface area. report_qc treats a
    # needs_review facet as honestly disclosed (verify in oblique), so flagging
    # is the intended path for a genuinely unresolvable pitch, not a silent error.
    # This is the fix for the recurring gate FAIL (pitch_resolved / slope_applied):
    # e.g. tiny facets below the RANSAC inlier floor at 3DEP point density.
    unresolved = 0
    for f in report_input.get("facets", []):
        if f.get("slope_deg") is None and not f.get("is_flat"):
            f["needs_review"] = True
            unresolved += 1
    if unresolved:
        logger.info("flagged %d facet(s) needs_review (LiDAR pitch unresolved)",
                    unresolved)
    return report_input
