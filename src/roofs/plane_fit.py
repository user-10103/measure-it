"""
Plane Fitting Module

RANSAC-based plane fitting for roof surfaces.
Fits plane equation: z = a*x + b*y + c

Reference: agent.md §6
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


# Default RANSAC parameters - relaxed for sparse raster-derived data
# Original spec values (0.15m, 0.6) are for dense LiDAR; real-world fused
# rasters yield ~50 points/facet, requiring more tolerance
DEFAULT_DIST_THRESH = 0.25  # meters
# Sparse 2011 LiDAR (2.8 pts/sq m) means k-means clusters often straddle two
# physical planes, so the dominant plane only captures ~25-35% of the cluster.
# 0.25 lets us recover the correct slope while success=False still signals low
# confidence to downstream QC.
DEFAULT_MIN_INLIER_RATIO = 0.25  # was 0.40
DEFAULT_MAX_TRIALS = 1000


@dataclass
class PlaneModel:
    """
    Represents a fitted plane: z = a*x + b*y + c

    Attributes:
        a: X coefficient (slope in X direction)
        b: Y coefficient (slope in Y direction)
        c: Intercept (base elevation)
        inlier_mask: Boolean array of inlier points
        inlier_count: Number of inlier points
        residual_median: Median fitting residual
        success: Whether fitting succeeded
    """
    a: float
    b: float
    c: float
    inlier_mask: np.ndarray
    inlier_count: int
    residual_median: float
    success: bool

    def predict(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Predict Z from X, Y coordinates."""
        return self.a * x + self.b * y + self.c

    @property
    def normal(self) -> Tuple[float, float, float]:
        """Surface normal vector (-a, -b, 1) normalized."""
        n = np.array([-self.a, -self.b, 1.0])
        return tuple(n / np.linalg.norm(n))

    @property
    def gradient_magnitude(self) -> float:
        """Gradient magnitude sqrt(a² + b²)."""
        return np.sqrt(self.a**2 + self.b**2)

    def to_dict(self) -> dict:
        """Export plane parameters as dict."""
        return {
            "a": float(self.a),
            "b": float(self.b),
            "c": float(self.c),
            "inlier_count": self.inlier_count,
            "residual_median_m": float(self.residual_median),
            "success": self.success
        }


def fit_plane_ransac(
    points: np.ndarray,
    dist_thresh: float = DEFAULT_DIST_THRESH,
    min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
    max_trials: int = DEFAULT_MAX_TRIALS
) -> PlaneModel:
    """
    Fit plane z = a*x + b*y + c using RANSAC.

    Reference: agent.md §6.1, agent-2.md fit_plane_ransac tool

    RANSAC loop:
    1. Sample 3 random points
    2. Fit temporary plane
    3. Compute distance of all points → inliers
    4. Select plane with max inliers
    5. Refit least-squares using inliers

    Args:
        points: Structured array with x, y, z fields OR (N, 3) array
        dist_thresh: Maximum residual for inlier classification (meters)
        min_inlier_ratio: Minimum fraction of points that must be inliers
        max_trials: Maximum RANSAC iterations

    Returns:
        PlaneModel with fitted parameters

    Raises:
        ValueError: If insufficient points for fitting
    """
    from sklearn.linear_model import RANSACRegressor, LinearRegression

    # Handle both structured and regular arrays
    if hasattr(points, 'dtype') and points.dtype.names:
        x = points['x']
        y = points['y']
        z = points['z']
    else:
        # Assume (N, 3) array
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

    n_points = len(x)

    if n_points < 3:
        logger.error(f"Insufficient points for plane fitting: {n_points}")
        return PlaneModel(
            a=0.0, b=0.0, c=0.0,
            inlier_mask=np.zeros(n_points, dtype=bool),
            inlier_count=0,
            residual_median=np.inf,
            success=False
        )

    logger.info(f"Fitting plane to {n_points:,} points (RANSAC thresh={dist_thresh}m)")

    # Prepare data for sklearn
    X = np.column_stack([x, y])
    y_target = z

    try:
        # RANSAC with linear regression
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=dist_thresh,
            max_trials=max_trials,
            random_state=42
        )

        ransac.fit(X, y_target)

        # Extract plane coefficients
        a, b = ransac.estimator_.coef_
        c = ransac.estimator_.intercept_

        # Get inlier mask
        inlier_mask = ransac.inlier_mask_
        inlier_count = inlier_mask.sum()
        inlier_ratio = inlier_count / n_points

        # Compute residuals for inliers
        predictions = ransac.predict(X)
        residuals = np.abs(y_target - predictions)
        residual_median = np.median(residuals[inlier_mask])

        # Check success criteria
        success = inlier_ratio >= min_inlier_ratio

        if not success:
            logger.warning(
                f"Plane fit below threshold: {inlier_ratio:.1%} inliers "
                f"(need {min_inlier_ratio:.1%})"
            )
        else:
            logger.info(
                f"Plane fit: z = {a:.4f}x + {b:.4f}y + {c:.2f}, "
                f"{inlier_count:,} inliers ({inlier_ratio:.1%}), "
                f"median residual {residual_median:.3f}m"
            )

        return PlaneModel(
            a=float(a),
            b=float(b),
            c=float(c),
            inlier_mask=inlier_mask,
            inlier_count=inlier_count,
            residual_median=float(residual_median),
            success=success
        )

    except Exception as e:
        logger.error(f"Plane fitting failed: {str(e)}")
        return PlaneModel(
            a=0.0, b=0.0, c=0.0,
            inlier_mask=np.zeros(n_points, dtype=bool),
            inlier_count=0,
            residual_median=np.inf,
            success=False
        )


def fit_plane_least_squares(
    points: np.ndarray
) -> Tuple[float, float, float]:
    """
    Simple least-squares plane fit (no outlier rejection).

    Reference: agent.md §6.2

    Solves: [z] = [x y 1] * [a b c]^T

    Args:
        points: Structured array with x, y, z fields

    Returns:
        Tuple of (a, b, c) plane coefficients
    """
    if hasattr(points, 'dtype') and points.dtype.names:
        x = points['x']
        y = points['y']
        z = points['z']
    else:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

    # Build design matrix [x, y, 1]
    A = np.column_stack([x, y, np.ones_like(x)])

    # Solve least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)

    a, b, c = coeffs

    return float(a), float(b), float(c)


def extract_inlier_boundary(
    points: np.ndarray,
    inlier_mask: np.ndarray,
    simplify_tolerance: float = 0.3
) -> Optional[Polygon]:
    """
    Extract 2D boundary polygon from inlier points.

    Uses convex hull with simplification.

    Args:
        points: Structured array with x, y fields
        inlier_mask: Boolean mask of inlier points
        simplify_tolerance: Douglas-Peucker tolerance (meters)

    Returns:
        Shapely Polygon or None if insufficient points
    """
    if hasattr(points, 'dtype') and points.dtype.names:
        x = points['x'][inlier_mask]
        y = points['y'][inlier_mask]
    else:
        x = points[inlier_mask, 0]
        y = points[inlier_mask, 1]

    if len(x) < 3:
        return None

    try:
        from scipy.spatial import ConvexHull
        points_2d = np.column_stack([x, y])
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]

        polygon = Polygon(hull_points)
        polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)

        return polygon

    except Exception as e:
        logger.warning(f"Boundary extraction failed: {str(e)}")
        return None


# Compass aspect bin -> downslope direction in degrees from North (clockwise)
ASPECT_BIN_DEG = {
    "N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
    "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0,
}


def plane_from_pitch_aspect(
    slope_deg: float,
    aspect: Optional[float] = None,
    aspect_bin: Optional[str] = None,
) -> PlaneModel:
    """Synthesize a PlaneModel (a, b, c=0) from a slope + downslope direction.

    Inverse of metrics.compute_slope_deg / compute_aspect_deg: lets an RGB-only
    facet (polygon + pitch class + aspect) feed the existing edge classifier and
    metrics, which only read plane.a / plane.b. A flat facet (slope 0 or no
    aspect) returns a = b = 0.

    For z = a*x + b*y + c the uphill gradient is (a, b); the downhill (aspect)
    direction is (-a, -b). With g = tan(slope) and the compass aspect converted
    to a math angle phi = radians(90 - aspect_deg):
        (-a, -b) = g * (cos phi, sin phi)  ->  a = -g*cos phi, b = -g*sin phi

    Args:
        slope_deg: Slope angle in degrees (0 = flat).
        aspect: Downslope compass bearing in degrees (0=N, 90=E). Optional.
        aspect_bin: Compass bin ("N".."NW", or "flat"/"none") if `aspect` absent.

    Returns:
        PlaneModel with synthesized a, b and c=0 (inlier fields are placeholders).
    """
    if aspect is None and aspect_bin is not None:
        aspect = ASPECT_BIN_DEG.get(str(aspect_bin).strip().upper())

    g = math.tan(math.radians(slope_deg))
    if slope_deg <= 0.0 or aspect is None:
        a = b = 0.0
    else:
        phi = math.radians(90.0 - aspect)
        a = -g * math.cos(phi)
        b = -g * math.sin(phi)

    return PlaneModel(
        a=float(a), b=float(b), c=0.0,
        inlier_mask=np.zeros(0, dtype=bool),
        inlier_count=0,
        residual_median=0.0,
        success=True,
    )


def fit_plane_to_raw_points(
    raw_points: np.ndarray,
    facet_polygon,
    ground_z: float,
    min_height_above_ground: float = 1.5,
    dist_thresh: float = 0.10,
    min_inlier_ratio: float = 0.65,
    max_trials: int = 1000,
) -> Optional[PlaneModel]:
    """Fit a plane to raw LiDAR points clipped to a facet's 2D boundary.

    Bypasses the raster intermediate and operates at the full EPT point density
    (~100–200 pts/facet from FL 3DEP data vs ~50 from the fused raster), which
    lets us tighten RANSAC thresholds from 0.25 m / 40 % to 0.10 m / 65 % and
    recover slope estimates accurate to ~1.5 deg rather than ~4-5 deg.

    Accepts both PDAL-style (uppercase X/Y/Z) and raster-style (lowercase x/y/z)
    structured arrays so it works on any point source.

    Args:
        raw_points: Structured array with X/Y/Z or x/y/z fields.
        facet_polygon: Shapely Polygon in the same planar-metric CRS as the points.
        ground_z: DTM ground elevation (metres) for the building centroid.  Points
            within ``min_height_above_ground`` metres of this are excluded.
        min_height_above_ground: Height filter threshold (default 1.5 m).
        dist_thresh: RANSAC residual threshold in metres (default 0.10 m).
        min_inlier_ratio: Minimum inlier fraction (default 0.65).
        max_trials: RANSAC iteration cap.

    Returns:
        PlaneModel fitted to the filtered raw points, or None when fewer than
        10 points survive the polygon / height filters.
    """
    from shapely.geometry import Point as ShapelyPoint

    if raw_points is None or len(raw_points) == 0:
        return None

    names = raw_points.dtype.names or ()
    if "X" in names:
        px, py, pz = raw_points["X"], raw_points["Y"], raw_points["Z"]
    elif "x" in names:
        px, py, pz = raw_points["x"], raw_points["y"], raw_points["z"]
    else:
        logger.warning("fit_plane_to_raw_points: unknown field names %s", names)
        return None

    # Height filter — strip ground / near-ground returns.
    roof_mask = pz > (ground_z + min_height_above_ground)
    px, py, pz = px[roof_mask], py[roof_mask], pz[roof_mask]
    if len(px) == 0:
        return None

    # Bounding-box pre-filter before the polygon containment test.
    minx, miny, maxx, maxy = facet_polygon.bounds
    bbox = (px >= minx) & (px <= maxx) & (py >= miny) & (py <= maxy)
    px, py, pz = px[bbox], py[bbox], pz[bbox]
    if len(px) == 0:
        return None

    # Precise polygon containment (with a small buffer for edge returns).
    poly_buf = facet_polygon.buffer(0.5)
    in_poly = np.array(
        [poly_buf.contains(ShapelyPoint(xi, yi)) for xi, yi in zip(px, py)]
    )
    px, py, pz = px[in_poly], py[in_poly], pz[in_poly]

    if len(px) < 10:
        logger.debug("fit_plane_to_raw_points: only %d points in facet polygon", len(px))
        return None

    pts = np.zeros(len(px), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
    pts["x"], pts["y"], pts["z"] = px, py, pz

    logger.info(
        "fit_plane_to_raw_points: fitting %d raw EPT points (thresh=%.2f m)",
        len(px), dist_thresh,
    )
    return fit_plane_ransac(pts, dist_thresh=dist_thresh,
                            min_inlier_ratio=min_inlier_ratio,
                            max_trials=max_trials)


def compute_pitch_from_ridge_eave(
    raw_points: np.ndarray,
    ridge_line,
    eave_line,
    sample_radius_m: float = 1.5,
) -> Optional[float]:
    """Compute roof pitch from sampled heights at the ridge and eave lines.

    slope = atan(ΔZ / horizontal_span)
    ΔZ            = median(Z near ridge) − median(Z near eave)
    horizontal_span = perpendicular distance from eave midpoint to the ridge line

    This is the most physically direct measurement available from LiDAR: the
    ridge and eave are strong linear features with high return density and
    minimal noise, unlike the facet interior where scattered points and
    interpolation artefacts dominate.

    Args:
        raw_points: Structured array with X/Y/Z or x/y/z fields.
        ridge_line: Shapely LineString of the ridge edge (planar-metric CRS).
        eave_line: Shapely LineString of an adjacent eave edge (same CRS).
        sample_radius_m: Buffer radius around each line for point sampling.

    Returns:
        Slope in degrees, or None if < 3 points are found near either line or if
        the geometry is degenerate (ridge lower than eave, zero span, etc.).
    """
    from shapely.geometry import Point as ShapelyPoint

    if raw_points is None or len(raw_points) == 0:
        return None

    names = raw_points.dtype.names or ()
    if "X" in names:
        px, py, pz = raw_points["X"], raw_points["Y"], raw_points["Z"]
    elif "x" in names:
        px, py, pz = raw_points["x"], raw_points["y"], raw_points["z"]
    else:
        return None

    def _sample_z(line, radius: float) -> Optional[np.ndarray]:
        buf = line.buffer(radius)
        bx, by, bminx, bminy, bmaxx, bmaxy = (
            buf.bounds[0], buf.bounds[1], buf.bounds[0], buf.bounds[1],
            buf.bounds[2], buf.bounds[3],
        )
        minx2, miny2, maxx2, maxy2 = buf.bounds
        bbox = (px >= minx2) & (px <= maxx2) & (py >= miny2) & (py <= maxy2)
        xi, yi, zi = px[bbox], py[bbox], pz[bbox]
        if len(xi) == 0:
            return None
        in_buf = np.array(
            [buf.contains(ShapelyPoint(x_, y_)) for x_, y_ in zip(xi, yi)]
        )
        z_near = zi[in_buf]
        return z_near if len(z_near) >= 3 else None

    ridge_z = _sample_z(ridge_line, sample_radius_m)
    eave_z = _sample_z(eave_line, sample_radius_m)

    if ridge_z is None or eave_z is None:
        return None

    delta_z = float(np.median(ridge_z) - np.median(eave_z))
    if delta_z <= 0.1:
        logger.debug("compute_pitch_from_ridge_eave: degenerate ΔZ=%.3f m", delta_z)
        return None

    eave_mid = eave_line.interpolate(0.5, normalized=True)
    horiz_span = float(ridge_line.distance(eave_mid))
    if horiz_span < 0.5:
        logger.debug("compute_pitch_from_ridge_eave: span too small (%.2f m)", horiz_span)
        return None

    slope_deg = math.degrees(math.atan(delta_z / horiz_span))
    logger.info(
        "compute_pitch_from_ridge_eave: ΔZ=%.2f m, span=%.2f m → %.1f°",
        delta_z, horiz_span, slope_deg,
    )
    return slope_deg


def compute_plane_residuals(
    points: np.ndarray,
    plane: PlaneModel
) -> np.ndarray:
    """
    Compute signed residuals (vertical distance to plane).

    Positive = above plane, negative = below.

    Args:
        points: Point array with x, y, z fields
        plane: Fitted PlaneModel

    Returns:
        Array of residuals
    """
    if hasattr(points, 'dtype') and points.dtype.names:
        x = points['x']
        y = points['y']
        z = points['z']
    else:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

    predicted = plane.predict(x, y)
    residuals = z - predicted

    return residuals
