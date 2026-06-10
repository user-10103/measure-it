"""
NAIP + nDSM guided roof facet segmentation.

Primary method: aspect-based segmentation on nDSM.
  - Compute gradient direction (aspect) at every nDSM pixel.
  - Pixels facing the same direction belong to the same physical plane.
  - Connected-component labeling separates spatially distinct regions.
  - Hough lines (from NAIP NIR) optionally split regions at ridge/hip lines.

This outperforms spatial k-means because it partitions by SLOPE DIRECTION
(which physical plane a point belongs to) rather than by XY position.
For a hip roof the result is naturally: N slope, S slope, NE hip, NW hip,
SE hip, SW hip — each as a single polygon.

Pipeline:
  1. Load nDSM clipped to footprint; optionally load NAIP for Hough refinement
  2. Compute Sobel gradient → aspect direction at each pixel
  3. Bin aspect into 8 compass directions + 1 flat bin
  4. Connected components within each bin → initial facet regions
  5. (Optional) split regions with Hough lines from NAIP NIR
  6. Polygonize → clip to footprint → drop tiny regions
  7. Assign ALL LiDAR points to polygons (nearest-centroid for unclaimed pts)
  8. Return list of Facet objects (same interface as k-means)
"""
import logging
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape, mapping, MultiPolygon
from shapely.ops import transform as shp_transform
from pyproj import Transformer

from src.roofs.segment import Facet

logger = logging.getLogger(__name__)

# ── tuning ────────────────────────────────────────────────────────────────────

# Gaussian smoothing sigma before gradient (pixels).  Higher = smoother slope
# map, fewer tiny fragments; lower = more detail but noisier aspect map.
GRADIENT_SMOOTH_SIGMA = 2.5

# Height thresholds for the roof mask.
#
# Two-threshold design for USA-wide production:
#   CORE  — applied inside the MS Buildings footprint polygon.  1.5 m is safe
#           against ground clutter (cars, shrubs, AC units) everywhere in the
#           USA; anything inside the footprint at ≥1.5 m is almost certainly
#           roof surface.
#   PORCH — applied only in the 8 m expansion ring OUTSIDE the footprint.
#           Lowered to 1.0 m to catch single-storey screened enclosures and
#           covered patios that sit at ~1.0–1.3 m above grade (common in FL,
#           TX, SE states).  The connectivity filter below prevents cars,
#           shrubs, and neighbour structures from being included: only
#           expansion-ring pixels that form a contiguous blob touching the
#           core roof mask survive.
MIN_NDSM_HEIGHT_M = 1.5       # core footprint threshold (all USA states)
MIN_NDSM_HEIGHT_PORCH_M = 1.0  # expansion-ring threshold (attached structures)

# Minimum region size (pixels) to keep before polygonisation.
MIN_REGION_PX = 30

# Minimum facet polygon area (sq m) to keep.
# Lowered from 4.0 → 2.5 to let small hip/valley transition facets survive.
MIN_FACET_AREA_M2 = 2.5

# Minimum LiDAR points to include a facet in RANSAC.
# Lowered from 15 → 8 to recover small porch-perimeter facets.
MIN_FACET_LIDAR_PTS = 8

# If we end up with fewer than this many facets the caller should fall back.
MIN_EXPECTED_FACETS = 3

# Polygon straightening: Douglas-Peucker tolerance (metres) for removing pixel
# staircase artifacts from rasterio.features.shapes() output.
SIMPLIFY_TOLERANCE_M = 0.5

# Orthogonal regularization: snap all polygon vertices to this grid (metres) in
# the building's principal-axis frame.  Must be ≥ 1 pixel width (~0.4 m) so the
# staircase is fully absorbed.  Smaller = more detail; larger = cleaner lines.
SNAP_GRID_M = 0.5

# Hough parameters (pixels at NAIP resolution ≈ 0.30 m/px)
HOUGH_SIGMA = 1.5
HOUGH_LOW = 0.05
HOUGH_HIGH = 0.20
HOUGH_THRESHOLD = 15
HOUGH_MIN_LINE_PX = 30        # ≈ 9 m — only long architectural lines
HOUGH_LINE_GAP_PX = 6
HOUGH_LINE_WIDTH_PX = 2       # half-width of rasterised line


# ── raster helpers ────────────────────────────────────────────────────────────

def _load_clipped(path: Path, footprint_wgs84, buffer_m: float = 2.0):
    """Load raster clipped to footprint + buffer.  Returns (arr, tfm, crs_str)."""
    with rasterio.open(path) as src:
        crs_str = str(src.crs)
        to_src = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
        geom_src = shp_transform(to_src, footprint_wgs84).buffer(buffer_m)
        arr, tfm = rio_mask(src, [mapping(geom_src)], crop=True, filled=True, nodata=0)
    return arr.astype(np.float32), tfm, crs_str


def _reproject_match(src_arr, src_tfm, src_crs, dst_tfm, dst_crs, dst_hw):
    """Reproject src_arr onto dst grid."""
    nb = src_arr.shape[0]
    dh, dw = dst_hw
    dst = np.zeros((nb, dh, dw), dtype=np.float32)
    reproject(
        source=src_arr, destination=dst,
        src_transform=src_tfm, src_crs=src_crs,
        dst_transform=dst_tfm, dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    return dst


def _footprint_mask(footprint_wgs84, transform, crs_str, hw):
    """Boolean mask: True inside footprint."""
    from rasterio.features import geometry_mask
    to_dst = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True).transform
    geom_dst = shp_transform(to_dst, footprint_wgs84)
    return geometry_mask([mapping(geom_dst)], out_shape=hw, transform=transform, invert=True)


# ── aspect-based segmentation (primary) ──────────────────────────────────────

def _aspect_labels(ndsm_2d: np.ndarray, roof_mask: np.ndarray,
                   n_bins: int = 8) -> np.ndarray:
    """
    Assign each roof pixel a label based on gradient direction (aspect).

    Returns integer label array.  Label 0 = background (not roof / flat).
    Each non-zero label corresponds to one connected region of consistent
    slope direction.
    """
    from scipy.ndimage import gaussian_filter, sobel
    from skimage.measure import label as sk_label
    from skimage.morphology import remove_small_objects

    # Smooth before gradient to reduce high-freq noise
    smooth = gaussian_filter(ndsm_2d, sigma=GRADIENT_SMOOTH_SIGMA)

    # Gradient via Sobel (sx = ∂z/∂x = slope eastward, sy = ∂z/∂y = slope northward)
    sx = sobel(smooth, axis=1)
    sy = sobel(smooth, axis=0)
    grad_mag = np.hypot(sx, sy)

    # Flat pixels (near-zero gradient) get their own bin
    flat_thresh = float(np.percentile(grad_mag[roof_mask], 15)) if roof_mask.any() else 0.0
    flat_mask = (grad_mag < flat_thresh) & roof_mask

    # Aspect in degrees [0, 360)
    aspect = (np.degrees(np.arctan2(-sy, sx)) + 360) % 360   # compass convention

    # Quantise into n_bins directions
    bin_size = 360.0 / n_bins
    aspect_bin = (aspect / bin_size).astype(int) % n_bins

    labels = np.zeros(ndsm_2d.shape, dtype=np.int32)
    next_label = 1

    for b in range(n_bins):
        bin_mask = (aspect_bin == b) & roof_mask & ~flat_mask
        if bin_mask.sum() < MIN_REGION_PX:
            continue
        # Connected components within this aspect bin
        comp = sk_label(bin_mask, connectivity=2)
        for c in range(1, int(comp.max()) + 1):
            region = comp == c
            if region.sum() < MIN_REGION_PX:
                continue
            labels[region] = next_label
            next_label += 1

    # Flat regions as one additional label per connected component
    if flat_mask.sum() >= MIN_REGION_PX:
        comp_flat = sk_label(flat_mask, connectivity=2)
        for c in range(1, int(comp_flat.max()) + 1):
            region = comp_flat == c
            if region.sum() < MIN_REGION_PX:
                continue
            labels[region] = next_label
            next_label += 1

    return labels


# ── Hough refinement (optional) ───────────────────────────────────────────────

def _aspect_bin_map(ndsm_2d: np.ndarray, n_bins: int = 8):
    """Per-pixel aspect bin (same smoothing/binning as _aspect_labels).

    Returns (aspect_bin int array, grad_mag float array).
    """
    from scipy.ndimage import gaussian_filter, sobel
    smooth = gaussian_filter(ndsm_2d, sigma=GRADIENT_SMOOTH_SIGMA)
    sx = sobel(smooth, axis=1)
    sy = sobel(smooth, axis=0)
    grad_mag = np.hypot(sx, sy)
    aspect = (np.degrees(np.arctan2(-sy, sx)) + 360) % 360
    aspect_bin = (aspect / (360.0 / n_bins)).astype(int) % n_bins
    return aspect_bin, grad_mag


def _hough_split_labels(
    labels: np.ndarray,
    naip_nir: np.ndarray,
    ndsm_2d: np.ndarray = None,
) -> np.ndarray:
    """
    Split label regions at confirmed Hough lines from NAIP NIR.

    NIR intensity edges are also produced by SHADOWS and by pale/discoloured
    roof patches — neither is an architectural line. A real ridge/hip/valley
    separates surfaces draining in DIFFERENT directions, so when an nDSM is
    available every candidate line is validated against the aspect map: keep
    it only if the two sides disagree in drainage direction along most of the
    line. Shadow and pale-patch boundaries lie WITHIN one slope plane and are
    rejected.

    Lines that survive are rasterised as zero, forcing connected-component
    re-labelling — and the painted pixels (plus any sub-MIN_REGION_PX
    fragments) are then reassigned to their nearest region so the split
    leaves NO unlabeled gaps in the roof.
    """
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line
    from skimage.draw import line as sk_line

    nir = naip_nir.astype(np.float32)
    finite = np.isfinite(nir)
    if not finite.any():
        return labels
    # Percentile normalisation: a global min-max stretch lets one deep shadow
    # or one blown-out pale patch compress contrast for the whole roof, which
    # destabilises the Canny thresholds.
    lo, hi = np.percentile(nir[finite], [2.0, 98.0])
    if hi <= lo:
        return labels
    nir_norm = np.clip((nir - lo) / (hi - lo), 0.0, 1.0)

    edges = canny(nir_norm, sigma=HOUGH_SIGMA, low_threshold=HOUGH_LOW,
                  high_threshold=HOUGH_HIGH)

    lines = probabilistic_hough_line(
        edges,
        threshold=HOUGH_THRESHOLD,
        line_length=HOUGH_MIN_LINE_PX,
        line_gap=HOUGH_LINE_GAP_PX,
        rng=42,
    )
    if not lines:
        return labels

    H, W = labels.shape

    # ── aspect validation: reject shadow / pale-patch intensity lines ──
    if ndsm_2d is not None:
        aspect_bin, _ = _aspect_bin_map(ndsm_2d)
        kept = []
        for (c0, r0), (c1, r1) in lines:
            dx, dy = c1 - c0, r1 - r0
            L = math.hypot(dx, dy)
            if L < 1e-6:
                continue
            # unit perpendicular (pixel space)
            px, py = -dy / L, dx / L
            off = 3  # px ≈ 1.5 m at 0.5 m grid
            n_samples = max(int(L // 5), 3)
            differ = valid = 0
            for k in range(n_samples):
                t = (k + 0.5) / n_samples
                mc, mr = c0 + dx * t, r0 + dy * t
                ca, ra = int(round(mc + px * off)), int(round(mr + py * off))
                cb, rb = int(round(mc - px * off)), int(round(mr - py * off))
                if not (0 <= ra < H and 0 <= rb < H and 0 <= ca < W and 0 <= cb < W):
                    continue
                if labels[ra, ca] == 0 or labels[rb, cb] == 0:
                    continue        # off-roof side: can't judge
                valid += 1
                if aspect_bin[ra, ca] != aspect_bin[rb, cb]:
                    differ += 1
            if valid >= 3 and differ / valid >= 0.6:
                kept.append(((c0, r0), (c1, r1)))
        if len(kept) < len(lines):
            logger.info(
                f"Hough validation: {len(lines) - len(kept)}/{len(lines)} intensity "
                f"lines rejected (same drainage both sides — shadow/pale patch)"
            )
        lines = kept
    if len(lines) < 2:
        return labels

    # Paint surviving lines onto a copy of labels as 0 (split boundary)
    split = labels.copy()
    for (c0, r0), (c1, r1) in lines:
        rr, cc = sk_line(int(r0), int(c0), int(r1), int(c1))
        for dr in range(-HOUGH_LINE_WIDTH_PX, HOUGH_LINE_WIDTH_PX + 1):
            for dc in range(-HOUGH_LINE_WIDTH_PX, HOUGH_LINE_WIDTH_PX + 1):
                rr2 = np.clip(rr + dr, 0, H - 1)
                cc2 = np.clip(cc + dc, 0, W - 1)
                split[rr2, cc2] = 0

    # Re-label connected components on the split label image
    new_labels = np.zeros_like(split)
    next_l = 1
    for orig_val in np.unique(split):
        if orig_val == 0:
            continue
        from skimage.measure import label as _lbl
        comp = _lbl(split == orig_val, connectivity=2)
        for c in range(1, int(comp.max()) + 1):
            region = comp == c
            if region.sum() < MIN_REGION_PX:
                continue
            new_labels[region] = next_l
            next_l += 1

    # ── close the gaps: painted line pixels and dropped small fragments ──
    # were roof in the input labels but are 0 now. Assign each to its
    # nearest labelled pixel so the split never leaves holes in the roof.
    holes = (labels > 0) & (new_labels == 0)
    if holes.any() and (new_labels > 0).any():
        from scipy.ndimage import distance_transform_edt
        _, (ir, ic) = distance_transform_edt(new_labels == 0, return_indices=True)
        new_labels[holes] = new_labels[ir[holes], ic[holes]]

    n_before = int(labels.max())
    n_after = int(new_labels.max())
    if n_after > n_before:
        logger.info(f"Hough refinement: {n_before} → {n_after} regions "
                    f"({len(lines)} validated architectural lines)")
    return new_labels


# ── polygon straightening / orthogonal regularization ─────────────────────────

def _principal_axis_angle(footprint_native) -> float:
    """
    Angle (degrees, east=0, CCW) of the building's longest axis derived from
    the minimum rotated rectangle of the footprint.  All facet polygons will be
    snapped to edges parallel or perpendicular to this direction.
    """
    rect = footprint_native.minimum_rotated_rectangle  # property in Shapely 2.x
    coords = list(rect.exterior.coords)
    edges = [(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in edges]
    a, b = edges[lengths.index(max(lengths))]
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _snap_poly_to_axis(poly, cos_t: float, sin_t: float, snap_m: float):
    """
    Rotate *poly* into the building-axis frame, snap every vertex to the
    *snap_m* grid, then rotate back.  Returns the regularized polygon.
    Falls back to a simplified version of the original on any failure so that
    facet count and attributes are never affected.
    """
    from shapely.affinity import affine_transform
    from shapely.geometry import Polygon as _Poly

    try:
        # Forward rotation: x' = cos·x + sin·y,  y' = -sin·x + cos·y
        rotated = affine_transform(poly, [cos_t, sin_t, -sin_t, cos_t, 0, 0])
        simplified = rotated.simplify(snap_m * 0.9, preserve_topology=True)

        if simplified.geom_type != "Polygon" or simplified.is_empty:
            return poly.simplify(snap_m, preserve_topology=True)

        # Snap exterior ring vertices to the grid
        raw = list(simplified.exterior.coords)
        snapped = [
            (round(x / snap_m) * snap_m, round(y / snap_m) * snap_m)
            for x, y in raw
        ]
        # Drop consecutive duplicates that arise from grid quantisation
        deduped = [snapped[0]]
        for pt in snapped[1:]:
            if pt != deduped[-1]:
                deduped.append(pt)
        # Ensure ring is closed and has enough vertices
        if deduped[0] != deduped[-1]:
            deduped.append(deduped[0])
        if len(deduped) < 4:
            return poly.simplify(snap_m, preserve_topology=True)

        reg = _Poly(deduped)
        if not reg.is_valid:
            reg = reg.buffer(0)
        if reg.is_empty or reg.area < 0.01:
            return poly.simplify(snap_m, preserve_topology=True)

        # Inverse rotation: x = cos·x' - sin·y',  y = sin·x' + cos·y'
        return affine_transform(reg, [cos_t, -sin_t, sin_t, cos_t, 0, 0])

    except Exception:
        # Never drop a facet due to regularization failure
        try:
            return poly.simplify(snap_m, preserve_topology=True)
        except Exception:
            return poly


def _regularize_polys_native(polys_native, footprint_native,
                              snap_m: float = SNAP_GRID_M):
    """
    Snap all facet polygon edges to be parallel or perpendicular to the
    building's principal axis.  Operates in native CRS (metres).

    Design invariants:
    - Returns the **same number** of polygons as input (no merges/drops).
    - Falls back to simplified original for any polygon that degenerates.
    - After snapping, a gap-fill pass re-merges any sub-pixel gaps introduced
      by grid quantisation so the tiling remains 100% of the footprint.
    - An overlap-trim pass ensures no two polygons claim the same area
      (largest-area-first priority), preserving dominant facets intact.
    """
    from shapely.ops import unary_union

    if not polys_native:
        return polys_native

    theta = _principal_axis_angle(footprint_native)
    theta_rad = math.radians(theta)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    # Snap every polygon to the shared rotated grid
    regularized = [_snap_poly_to_axis(p, cos_t, sin_t, snap_m) for p in polys_native]

    # Clip each regularized polygon to the footprint so no polygon escapes the
    # building boundary (snapping can push vertices slightly outward)
    clipped = []
    for orig, reg in zip(polys_native, regularized):
        try:
            c = reg.intersection(footprint_native)
            clipped.append(c if not c.is_empty and c.area > 0.01 else orig)
        except Exception:
            clipped.append(orig)
    regularized = clipped

    # Gap-fill: areas inside the footprint not covered after snapping → merge
    # into the nearest polygon (centroid distance).  Same logic as _fill_footprint.
    covered = unary_union(regularized)
    gap = footprint_native.difference(covered)
    if not gap.is_empty:
        gap_parts = (
            list(gap.geoms)
            if gap.geom_type in ("MultiPolygon", "GeometryCollection")
            else [gap]
        )
        for piece in gap_parts:
            if piece.is_empty or piece.area < 1e-8:
                continue
            idx = min(
                range(len(regularized)),
                key=lambda i: piece.centroid.distance(regularized[i].centroid),
            )
            regularized[idx] = regularized[idx].union(piece)

    # Overlap-trim: process largest facets first so dominant planes keep their
    # full area; smaller facets are trimmed away from already-claimed regions.
    order = sorted(range(len(regularized)), key=lambda i: -regularized[i].area)
    trimmed = [None] * len(regularized)
    claimed = None
    for i in order:
        p = regularized[i]
        if claimed is not None:
            try:
                diff = p.difference(claimed)
                p = diff if not diff.is_empty and diff.area > 0.01 else p
            except Exception:
                pass
        trimmed[i] = p
        claimed = unary_union([x for x in trimmed if x is not None])

    return trimmed


# ── polygon extraction ────────────────────────────────────────────────────────

def _polygonize(label_arr: np.ndarray, transform, crs_str: str,
                footprint_wgs84, min_area_m2: float) -> List:
    """Convert labelled raster to WGS84 shapely polygons."""
    from rasterio.features import shapes as rio_shapes

    to_wgs = Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True).transform
    polys = []

    for geom_dict, val in rio_shapes(
        label_arr.astype(np.int32),
        mask=(label_arr > 0).astype(np.uint8),
        transform=transform,
    ):
        if int(val) == 0:
            continue
        poly_native = shape(geom_dict)
        # Pass 1 — remove pixel-grid staircase before any WGS84 conversion.
        # Tolerance in metres (native CRS); preserve_topology keeps the ring valid.
        poly_native = poly_native.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=True)

        # Flatten MultiPolygon
        parts = list(poly_native.geoms) if isinstance(poly_native, MultiPolygon) else [poly_native]
        for part in parts:
            if part.area < 1e-6:
                continue
            poly_wgs84 = shp_transform(to_wgs, part)
            # Rough area check in metric (degrees² → m² at mid-lat)
            lat_mid = poly_wgs84.centroid.y
            m2_per_deg2 = (111_320 * math.cos(math.radians(lat_mid))) * 110_540
            if poly_wgs84.area * m2_per_deg2 >= min_area_m2:
                polys.append(poly_wgs84)
    return polys


def _clip_polys(polys, footprint_wgs84, min_area_m2):
    """Clip each polygon to footprint, drop sub-threshold results."""
    result = []
    for p in polys:
        try:
            clipped = p.intersection(footprint_wgs84)
        except Exception:
            continue
        if clipped.is_empty:
            continue
        parts = list(clipped.geoms) if isinstance(clipped, MultiPolygon) else [clipped]
        for part in parts:
            lat_mid = part.centroid.y
            m2_per_deg2 = (111_320 * math.cos(math.radians(lat_mid))) * 110_540
            if part.area * m2_per_deg2 >= min_area_m2:
                result.append(part)
    return result


# ── LiDAR assignment ──────────────────────────────────────────────────────────

def _assign_lidar(polys_wgs84, lidar_pts, points_crs: str):
    """
    Assign LiDAR points to polygons.  Unclaimed points go to nearest centroid.
    Returns list of point arrays (one per polygon).
    """
    import shapely as _shp
    from pyproj import CRS as _CRS

    pts_crs = _CRS.from_user_input(points_crs)
    to_pts = Transformer.from_crs("EPSG:4326", pts_crs, always_xy=True).transform

    px = lidar_pts['x'].astype(float)
    py = lidar_pts['y'].astype(float)

    # Per-polygon containment
    sets = []
    claimed = np.zeros(len(lidar_pts), dtype=bool)
    for poly_wgs84 in polys_wgs84:
        poly_pts = shp_transform(to_pts, poly_wgs84)
        m = _shp.contains_xy(poly_pts, px, py)
        sets.append(lidar_pts[m])
        claimed |= m

    # Residual: assign unclaimed points to nearest polygon centroid
    n_unclaimed = int((~claimed).sum())
    if n_unclaimed > 0 and len(polys_wgs84) > 0:
        centroids_native = []
        for p in polys_wgs84:
            c = shp_transform(to_pts, p).centroid
            centroids_native.append((c.x, c.y))
        cx = np.array([c[0] for c in centroids_native])
        cy = np.array([c[1] for c in centroids_native])
        ux = px[~claimed]
        uy = py[~claimed]
        dists = np.sqrt((ux[:, None] - cx[None, :])**2 + (uy[:, None] - cy[None, :])**2)
        nearest = np.argmin(dists, axis=1)
        unclaimed_pts = lidar_pts[~claimed]
        for pid in range(len(polys_wgs84)):
            extra = unclaimed_pts[nearest == pid]
            if len(extra) > 0:
                sets[pid] = np.concatenate([sets[pid], extra]) if len(sets[pid]) else extra
        logger.info(f"Residual: {n_unclaimed} unclaimed pts distributed to nearest polygon")

    return sets


# ── footprint tiling ──────────────────────────────────────────────────────────

def _fill_footprint(polys, footprint_wgs84):
    """
    Expand polygons so they collectively tile the complete footprint.

    Any gap between the union of polys and the footprint boundary is merged
    into the nearest polygon (by centroid distance).  After this call the
    polygons cover 100% of the footprint so edge classification finds all
    perimeter and inter-facet boundaries.
    """
    from shapely.ops import unary_union

    if not polys:
        return polys

    covered = unary_union(polys)
    uncovered = footprint_wgs84.difference(covered)
    if uncovered.is_empty:
        return polys

    parts = (list(uncovered.geoms)
             if uncovered.geom_type in ('MultiPolygon', 'GeometryCollection')
             else [uncovered])

    result = list(polys)
    for piece in parts:
        if piece.is_empty or piece.area < 1e-12:
            continue
        nearest_idx = min(
            range(len(result)),
            key=lambda i: piece.centroid.distance(result[i].centroid),
        )
        result[nearest_idx] = result[nearest_idx].union(piece)

    covered_after = sum(p.area for p in result)
    logger.info(
        f"Footprint tiling: filled {uncovered.area / footprint_wgs84.area * 100:.0f}% gap; "
        f"coverage {covered_after / footprint_wgs84.area * 100:.0f}%"
    )
    return result


# ── public entry point ────────────────────────────────────────────────────────

def naip_guided_segmentation(
    naip_clipped_path: Path,
    ndsm_path: Path,
    footprint_wgs84,
    lidar_points: np.ndarray,
    points_crs: str,
    min_facets: int = MIN_EXPECTED_FACETS,
    use_hough: bool = True,
) -> Optional[List[Facet]]:
    """
    Segment roof facets using nDSM aspect + optional NAIP Hough refinement.

    Args:
        naip_clipped_path: Clipped NAIP TIF (RGBIR, ≈30 cm).
        ndsm_path:         nDSM TIF (metres above ground).
        footprint_wgs84:   Building footprint in WGS84.
        lidar_points:      Structured array with x, y, z fields (roof-level only).
        points_crs:        CRS of lidar_points x/y.
        min_facets:        Return None if fewer than this many facets survive.
        use_hough:         Whether to apply NAIP Hough refinement after aspect seg.

    Returns:
        List of Facet objects, or None to trigger k-means fallback.
    """
    try:
        from skimage.measure import label as _sk_label  # noqa: F401
        from scipy.ndimage import gaussian_filter  # noqa: F401
    except ImportError as e:
        logger.warning(f"scikit-image/scipy not available: {e} — skip NAIP segmentation")
        return None

    logger.info("NAIP-guided segmentation: loading nDSM…")

    # ── 1. Load nDSM clipped to footprint ────────────────────────────────────
    ndsm_arr, ndsm_tfm, ndsm_crs = _load_clipped(ndsm_path, footprint_wgs84)
    H, W = ndsm_arr.shape[1], ndsm_arr.shape[2]
    if H < 8 or W < 8:
        logger.warning("nDSM too small after clipping")
        return None

    ndsm_2d = ndsm_arr[0]

    # ── 2. Footprint mask ─────────────────────────────────────────────────────
    # Use the exact footprint for the core mask, but also check a 10m expanded
    # zone to catch screened enclosures / porches attached to the main building.
    fp_mask = _footprint_mask(footprint_wgs84, ndsm_tfm, ndsm_crs, (H, W))

    # Core roof mask: inside the MS Buildings footprint AND above the main
    # height threshold.  1.5 m is conservative enough to exclude cars, AC units,
    # and shrubs everywhere in the USA while capturing all pitched/flat surfaces.
    roof_mask = fp_mask & (ndsm_2d > MIN_NDSM_HEIGHT_M)
    if roof_mask.sum() < 50:
        logger.warning("Too few roof pixels in nDSM")
        return None

    # Attached-structure detection: expand the footprint by 8 m to include
    # screened enclosures, covered patios, and carports that sit just outside
    # the MS Buildings polygon.  Two-threshold design:
    #   • Core pixels  : fp_mask        & nDSM > 1.5 m (already in roof_mask)
    #   • Porch pixels : expansion ring & nDSM > 1.0 m  AND connected to core
    # The connectivity filter removes isolated false positives (parked cars,
    # hedges, neighbour roofs) that are in the 8 m ring but don't touch the
    # building.  Works correctly in every USA state: FL lanais connect, CO
    # shrubs don't.
    try:
        from skimage.measure import label as _sk_label2
        from shapely.geometry import mapping as _mapping2
        from rasterio.features import geometry_mask as _gm2

        to_ndsm = Transformer.from_crs("EPSG:4326", ndsm_crs, always_xy=True).transform
        fp_native = shp_transform(to_ndsm, footprint_wgs84)
        fp_expanded_native = fp_native.buffer(8.0)
        fp_expanded_mask = ~_gm2(
            [_mapping2(fp_expanded_native)],
            out_shape=(H, W), transform=ndsm_tfm, invert=False,
        )

        # Candidate porch pixels: expansion ring only, lower height threshold
        expansion_ring = fp_expanded_mask & ~fp_mask
        porch_candidates = expansion_ring & (ndsm_2d > MIN_NDSM_HEIGHT_PORCH_M)

        # Connectivity filter: keep only blobs that touch the core roof mask
        combined = roof_mask | porch_candidates
        labeled_combined = _sk_label2(combined, connectivity=2)
        # Find which labels include at least one core pixel
        core_labels = set(int(v) for v in np.unique(labeled_combined[roof_mask]) if v > 0)
        # Porch pixels belong to a core-connected label → include them
        connected_porch = porch_candidates & np.isin(labeled_combined, list(core_labels))

        roof_mask_expanded = roof_mask | connected_porch

        n_porch = int(connected_porch.sum())
        n_isolated = int(porch_candidates.sum()) - n_porch
        logger.info(
            f"Aspect segmentation on {W}×{H} nDSM "
            f"({int(roof_mask.sum())} footprint px + "
            f"{n_porch} connected porch px"
            + (f", {n_isolated} isolated discarded" if n_isolated > 0 else "") + ")…"
        )
    except Exception as _e:
        logger.debug(f"Footprint expansion failed: {_e}")
        fp_expanded_native = fp_native if 'fp_native' in dir() else None
        roof_mask_expanded = roof_mask
        logger.info(
            f"Aspect segmentation on {W}×{H} nDSM "
            f"({int(roof_mask.sum())} footprint px, no expansion)…"
        )

    # ── 3. Aspect-based segmentation ─────────────────────────────────────────
    labels = _aspect_labels(ndsm_2d, roof_mask_expanded)
    n_regions = int(labels.max())
    logger.info(f"Aspect regions: {n_regions}")

    # ── 4. Optional Hough refinement from NAIP ────────────────────────────────
    if use_hough and naip_clipped_path and Path(naip_clipped_path).exists():
        try:
            naip_arr, naip_tfm, naip_crs = _load_clipped(naip_clipped_path, footprint_wgs84)
            # Reproject NAIP to nDSM grid so Hough lines are pixel-aligned
            naip_reprojected = _reproject_match(
                naip_arr, naip_tfm, naip_crs, ndsm_tfm, ndsm_crs, (H, W)
            )
            # Use NIR band (band 4 = index 3) or green as proxy
            nir_band = naip_reprojected[3] if naip_reprojected.shape[0] >= 4 else naip_reprojected[1]
            labels = _hough_split_labels(labels, nir_band, ndsm_2d=ndsm_2d)
        except Exception as e:
            logger.debug(f"Hough refinement skipped: {e}")

    if int(labels.max()) < min_facets:
        logger.warning(f"Only {int(labels.max())} aspect regions — falling back")
        return None

    # ── 5. Polygonise ─────────────────────────────────────────────────────────
    # Clip to the expanded footprint so porch polygons survive the filter.
    # If expansion failed (no fp_expanded_native), fall back to bare footprint.
    try:
        to_wgs_back = Transformer.from_crs(ndsm_crs, "EPSG:4326", always_xy=True).transform
        fp_expanded_wgs84 = shp_transform(to_wgs_back, fp_expanded_native) \
            if fp_expanded_native is not None else footprint_wgs84
    except Exception:
        fp_expanded_wgs84 = footprint_wgs84
    raw_polys = _polygonize(labels, ndsm_tfm, ndsm_crs, fp_expanded_wgs84, MIN_FACET_AREA_M2)
    clipped_polys = _clip_polys(raw_polys, fp_expanded_wgs84, MIN_FACET_AREA_M2)
    logger.info(f"Polygons after clip/filter: {len(clipped_polys)}")

    # Snap: expand polygons to collectively tile the full footprint so that
    # every perimeter and inter-facet boundary is available for edge
    # classification.  This eliminates the 1.3-1.5× normalization scale factor.
    if clipped_polys:
        clipped_polys = _fill_footprint(clipped_polys, fp_expanded_wgs84)

    if len(clipped_polys) < min_facets:
        logger.warning(f"Too few polygons ({len(clipped_polys)}) — falling back")
        return None

    # ── 6. Assign LiDAR points ────────────────────────────────────────────────
    point_sets = _assign_lidar(clipped_polys, lidar_points, points_crs)

    # ── 7. Build Facet list ────────────────────────────────────────────────────
    facets = []
    skipped = 0
    for idx, (poly, pts) in enumerate(zip(clipped_polys, point_sets)):
        if len(pts) < MIN_FACET_LIDAR_PTS:
            skipped += 1
            continue
        to_pts_crs = Transformer.from_crs("EPSG:4326", points_crs, always_xy=True).transform
        poly_pts_crs = shp_transform(to_pts_crs, poly)
        facets.append(Facet(
            facet_id=len(facets),
            points=pts,
            label=idx,
            polygon=poly_pts_crs,
        ))

    if skipped:
        logger.info(f"Dropped {skipped} polygon(s) with <{MIN_FACET_LIDAR_PTS} pts")

    if len(facets) < min_facets:
        logger.warning(f"{len(facets)} facets after LiDAR filter — falling back")
        return None

    logger.info(
        f"NAIP/nDSM segmentation: {len(facets)} facets "
        f"(sizes: {[f.count for f in facets]})"
    )
    return facets
