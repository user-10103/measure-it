"""
Roof Metrics Module

Computes slope, pitch, aspect, and area from fitted plane parameters.

Reference: agent.md §7, agent-2.md compute_metrics tool
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
from shapely.geometry import Polygon

from src.roofs.plane_fit import PlaneModel

logger = logging.getLogger(__name__)


# Standard roof pitches (rise:12 format)
STANDARD_PITCHES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24]

# Slopes below this are flat roof sections (membrane/built-up). At 1 m LiDAR
# a true membrane fits at 2.5-5 deg (observed: 1,721 sqft of membrane at
# exactly 2.50 deg missed by a 2.5 threshold); 5.0 matches the enclosure
# filter's flat convention. Genuine 1:12 roofs are within noise of flat at
# this data density - binning them flat is the honest call.
FLAT_SLOPE_DEG = 5.0

# Compass bins for aspect
COMPASS_BINS = {
    "N": (337.5, 22.5),
    "NE": (22.5, 67.5),
    "E": (67.5, 112.5),
    "SE": (112.5, 157.5),
    "S": (157.5, 202.5),
    "SW": (202.5, 247.5),
    "W": (247.5, 292.5),
    "NW": (292.5, 337.5)
}


@dataclass
class FacetMetrics:
    """
    Complete metrics for a roof facet.

    Reference: agent.md §9, agent-2.md io_schemas.metrics_json
    """
    facet_id: int
    plane: Dict[str, float]  # {"a": float, "b": float, "c": float}
    slope_deg: float
    pitch_string: str  # "X:12" format
    aspect_deg: float
    aspect_bin: str  # N, NE, E, SE, S, SW, W, NW
    plan_area_m2: float
    surface_area_m2: float
    inliers_count: int
    residual_median_m: float
    is_flat: bool = False

    def to_dict(self) -> dict:
        """Export as dict matching agent spec schema."""
        return {
            "facet_id": self.facet_id,
            "plane": self.plane,
            "slope_deg": round(self.slope_deg, 2),
            "pitch_string": self.pitch_string,
            "aspect_deg": round(self.aspect_deg, 2),
            "aspect_bin": self.aspect_bin,
            "plan_area_m2": round(self.plan_area_m2, 2),
            "surface_area_m2": round(self.surface_area_m2, 2),
            "inliers_count": self.inliers_count,
            "residual_median_m": round(self.residual_median_m, 3),
            "is_flat": self.is_flat
        }


def compute_slope_deg(plane: PlaneModel) -> float:
    """
    Compute slope angle in degrees from plane parameters.

    Reference: agent.md §7.1

    θ = atan(sqrt(a² + b²))
    θ_deg = θ * 180/π

    Args:
        plane: Fitted PlaneModel

    Returns:
        Slope angle in degrees (0 = flat, 90 = vertical)
    """
    gradient_m = plane.gradient_magnitude
    theta_rad = math.atan(gradient_m)
    theta_deg = math.degrees(theta_rad)

    return theta_deg


def compute_pitch_string(slope_deg: float) -> str:
    """
    Convert slope angle to pitch string ("X:12" format).

    Reference: agent.md §7.2

    rise_12 = tan(θ) * 12
    pitch_rounded = round_to_standard(rise_12)
    pitch = f"{pitch_rounded}:12"

    Args:
        slope_deg: Slope angle in degrees

    Returns:
        Pitch string like "6:12", "9:12", etc.
    """
    theta_rad = math.radians(slope_deg)
    gradient_m = math.tan(theta_rad)

    # Rise per 12 units of run
    rise_12 = gradient_m * 12

    # Round to nearest standard pitch
    pitch_rounded = min(STANDARD_PITCHES, key=lambda p: abs(p - rise_12))

    return f"{pitch_rounded}:12"


def compute_aspect_deg(plane: PlaneModel) -> float:
    """
    Compute aspect (downslope direction) in degrees from North.

    Reference: agent.md §7.3

    Uses downhill gradient direction:
    φ = atan2(-b, -a)
    φ_deg = φ * 180/π

    Args:
        plane: Fitted PlaneModel

    Returns:
        Aspect angle in degrees (0-360, 0=North, 90=East)
    """
    # Downslope direction (negative gradient)
    phi_rad = math.atan2(-plane.b, -plane.a)

    # Convert to degrees
    phi_deg = math.degrees(phi_rad)

    # Convert from math convention (0=East, CCW) to compass (0=North, CW)
    # Compass = 90 - math_angle
    compass_deg = 90 - phi_deg

    # Normalize to 0-360
    if compass_deg < 0:
        compass_deg += 360
    if compass_deg >= 360:
        compass_deg -= 360

    return compass_deg


def compute_aspect_bin(aspect_deg: float) -> str:
    """
    Convert aspect angle to compass bin (N, NE, E, etc.).

    Reference: agent.md §7.3

    Args:
        aspect_deg: Aspect angle in degrees (0-360)

    Returns:
        Compass direction bin
    """
    # Handle North wrapping (337.5 to 22.5)
    if aspect_deg >= 337.5 or aspect_deg < 22.5:
        return "N"

    for direction, (start, end) in COMPASS_BINS.items():
        if direction == "N":
            continue  # Handled above
        if start <= aspect_deg < end:
            return direction

    return "N"  # Default


def compute_plan_area(
    points: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute projected (plan) area using convex hull.

    Reference: agent.md §7.4

    Args:
        points: Point array with x, y fields
        inlier_mask: Optional mask for inlier points

    Returns:
        Plan area in square meters
    """
    if hasattr(points, 'dtype') and points.dtype.names:
        x = points['x']
        y = points['y']
    else:
        x = points[:, 0]
        y = points[:, 1]

    if inlier_mask is not None:
        x = x[inlier_mask]
        y = y[inlier_mask]

    if len(x) < 3:
        return 0.0

    try:
        from scipy.spatial import ConvexHull
        points_2d = np.column_stack([x, y])
        hull = ConvexHull(points_2d)
        return float(hull.volume)  # 2D hull "volume" is area

    except Exception as e:
        logger.warning(f"Plan area computation failed: {str(e)}")
        return 0.0


def compute_surface_area(plan_area: float, slope_deg: float) -> float:
    """
    Compute true surface area from plan area and slope.

    Reference: agent.md §7.4

    surface_area = plan_area / cos(θ)

    Args:
        plan_area: Projected plan area (m²)
        slope_deg: Slope angle in degrees

    Returns:
        Surface area in square meters
    """
    theta_rad = math.radians(slope_deg)
    cos_theta = math.cos(theta_rad)

    if cos_theta < 0.01:  # Near-vertical, cap at ~89 degrees
        cos_theta = 0.01

    surface_area = plan_area / cos_theta

    return surface_area


def compute_facet_metrics(
    facet_id: int,
    points: np.ndarray,
    plane: PlaneModel,
    boundary: Optional[Polygon] = None
) -> FacetMetrics:
    """
    Compute all metrics for a single facet.

    Reference: agent-2.md compute_metrics tool

    Args:
        facet_id: Facet identifier
        points: Point array for this facet
        plane: Fitted PlaneModel
        boundary: Optional clipped facet polygon. When given, plan area comes
            from its exact area instead of the inlier convex hull (which
            overcounts on concave facets and overlapping clusters).

    Returns:
        FacetMetrics dataclass with all computed values
    """
    # Slope
    slope_deg = compute_slope_deg(plane)

    # Flat snap: below FLAT_SLOPE_DEG the slope is membrane-roof LiDAR noise.
    is_flat = slope_deg < FLAT_SLOPE_DEG
    if is_flat:
        slope_deg = 0.0

    # Pitch
    pitch_string = compute_pitch_string(slope_deg)

    # Aspect
    aspect_deg = compute_aspect_deg(plane)
    aspect_bin = compute_aspect_bin(aspect_deg)

    # Area
    if boundary is not None and not boundary.is_empty:
        plan_area = float(boundary.area)
    else:
        plan_area = compute_plan_area(points, plane.inlier_mask)
    surface_area = compute_surface_area(plan_area, slope_deg)

    metrics = FacetMetrics(
        facet_id=facet_id,
        plane={"a": plane.a, "b": plane.b, "c": plane.c},
        slope_deg=slope_deg,
        pitch_string=pitch_string,
        aspect_deg=aspect_deg,
        aspect_bin=aspect_bin,
        plan_area_m2=plan_area,
        surface_area_m2=surface_area,
        inliers_count=plane.inlier_count,
        residual_median_m=plane.residual_median,
        is_flat=is_flat
    )

    logger.info(
        f"Facet {facet_id}: slope={slope_deg:.1f}°, pitch={pitch_string}, "
        f"aspect={aspect_bin} ({aspect_deg:.0f}°), "
        f"area={surface_area:.1f}m² ({plan_area:.1f}m² plan)"
        + (" [flat]" if is_flat else "")
    )

    return metrics


def compute_all_facet_metrics(
    facets: List,  # List[Facet] from segment.py
    planes: List[PlaneModel],
    boundaries: Optional[Dict[int, Polygon]] = None
) -> List[FacetMetrics]:
    """
    Compute metrics for all facets.

    Args:
        facets: List of Facet objects from segmentation
        planes: List of PlaneModel objects (one per facet)
        boundaries: Optional {facet_id: clipped polygon} for exact plan areas

    Returns:
        List of FacetMetrics
    """
    if len(facets) != len(planes):
        raise ValueError(
            f"Facet count ({len(facets)}) doesn't match plane count ({len(planes)})"
        )

    metrics_list = []

    for facet, plane in zip(facets, planes):
        metrics = compute_facet_metrics(
            facet_id=facet.facet_id,
            points=facet.points,
            plane=plane,
            boundary=(boundaries or {}).get(facet.facet_id)
        )
        metrics_list.append(metrics)

    return metrics_list


def compute_building_metrics(
    building_id: str,
    facet_metrics: List[FacetMetrics],
    footprint_area_m2: float
) -> Dict[str, Any]:
    """
    Aggregate metrics for entire building.

    Reference: agent.md §9 output schema

    Args:
        building_id: Building identifier
        facet_metrics: List of FacetMetrics for all facets
        footprint_area_m2: Building footprint area

    Returns:
        Dict matching agent spec output schema
    """
    return {
        "building_id": building_id,
        "footprint_area_m2": round(footprint_area_m2, 2),
        "facets": [m.to_dict() for m in facet_metrics],
        "num_facets": len(facet_metrics),
        "total_surface_area_m2": round(
            sum(m.surface_area_m2 for m in facet_metrics), 2
        ),
        "dominant_pitch": _get_dominant_pitch(facet_metrics),
        "dominant_aspect": _get_dominant_aspect(facet_metrics)
    }


def _get_dominant_pitch(facet_metrics: List[FacetMetrics]) -> str:
    """Get pitch of largest facet by area."""
    if not facet_metrics:
        return "0:12"
    largest = max(facet_metrics, key=lambda m: m.surface_area_m2)
    return largest.pitch_string


def _get_dominant_aspect(facet_metrics: List[FacetMetrics]) -> str:
    """Get aspect bin of largest facet by area."""
    if not facet_metrics:
        return "N"
    largest = max(facet_metrics, key=lambda m: m.surface_area_m2)
    return largest.aspect_bin


def pitch_string_to_degrees(pitch_string: str) -> float:
    """
    Convert pitch string to slope degrees.

    Args:
        pitch_string: Pitch like "6:12"

    Returns:
        Slope angle in degrees
    """
    try:
        rise, run = pitch_string.split(":")
        rise = float(rise)
        run = float(run)
        theta_rad = math.atan(rise / run)
        return math.degrees(theta_rad)
    except Exception:
        return 0.0
