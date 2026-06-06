"""
QC Validators Module

Quality control checks for roof metrics and data validation.

Reference: agent-2.md io_schemas.metrics_json.qc, CLAUDE.md §4.2
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Thresholds from agent spec
PITCH_MIN_DEG = 5.0     # Below this is suspiciously flat
PITCH_MAX_DEG = 70.0    # Above this is suspiciously steep
RESIDUAL_MAX_M = 0.15   # Maximum acceptable median residual
VEGETATION_MAX_PCT = 30.0  # Warning if vegetation > this
PLAN_AREA_MIN_M2 = 10.0    # Minimum realistic roof area
PLAN_AREA_MAX_M2 = 2000.0  # Maximum realistic residential roof


class QCFlag(Enum):
    """QC warning/error flags."""
    PITCH_TOO_FLAT = "pitch_below_5_degrees"
    PITCH_TOO_STEEP = "pitch_above_70_degrees"
    HIGH_RESIDUAL = "plane_fit_residual_high"
    LOW_INLIER_RATIO = "low_inlier_ratio"
    HIGH_VEGETATION = "high_vegetation_percentage"
    AREA_TOO_SMALL = "plan_area_too_small"
    AREA_TOO_LARGE = "plan_area_too_large"
    PLANE_FIT_FAILED = "plane_fit_failed"
    INSUFFICIENT_POINTS = "insufficient_points"
    DSM_NOISY = "dsm_high_variance"


@dataclass
class QCResult:
    """
    Quality control result for a building or facet.

    Attributes:
        success: Overall QC pass/fail
        flags: List of QC flags triggered
        ndsm_stats: Height statistics
        vegetation_pct: Percentage of vegetation points
        plane_fit_success: Whether plane fitting succeeded
        residual_median: Median plane fit residual
        confidence_score: Overall confidence (0-1)
        details: Additional QC details
    """
    success: bool = True
    flags: List[QCFlag] = field(default_factory=list)
    ndsm_stats: Dict[str, float] = field(default_factory=dict)
    vegetation_pct: float = 0.0
    plane_fit_success: bool = True
    residual_median: float = 0.0
    confidence_score: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    def add_flag(self, flag: QCFlag) -> None:
        """Add a QC flag and update success status."""
        if flag not in self.flags:
            self.flags.append(flag)
        # Any flag reduces confidence
        self.confidence_score = max(0.0, self.confidence_score - 0.1)

    def to_dict(self) -> dict:
        """Export as dict for JSON serialization."""
        return {
            "success": self.success,
            "flags": [f.value for f in self.flags],
            "ndsm_stats": self.ndsm_stats,
            "vegetation_pct": round(self.vegetation_pct, 1),
            "plane_fit_success": self.plane_fit_success,
            "residual_median_m": round(self.residual_median, 3),
            "confidence_score": round(self.confidence_score, 2),
            "details": self.details
        }


def compute_ndsm_stats(ndsm: np.ndarray) -> Dict[str, float]:
    """
    Compute nDSM statistics for QC.

    Reference: agent-2.md io_schemas.metrics_json.qc.ndsm_stats

    Args:
        ndsm: Normalized DSM array

    Returns:
        Dict with min, max, mean, std values
    """
    # Filter out nodata values
    valid = ndsm[~np.isnan(ndsm) & (ndsm > -9000)]

    if len(valid) == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std())
    }


def validate_pitch(slope_deg: float, qc: QCResult) -> None:
    """
    Validate pitch is within expected range.

    Reference: CLAUDE.md §4.2

    Args:
        slope_deg: Slope angle in degrees
        qc: QCResult to update
    """
    if slope_deg < PITCH_MIN_DEG:
        qc.add_flag(QCFlag.PITCH_TOO_FLAT)
        qc.details["pitch_issue"] = f"Slope {slope_deg:.1f}° below minimum {PITCH_MIN_DEG}°"

    if slope_deg > PITCH_MAX_DEG:
        qc.add_flag(QCFlag.PITCH_TOO_STEEP)
        qc.details["pitch_issue"] = f"Slope {slope_deg:.1f}° above maximum {PITCH_MAX_DEG}°"


def validate_residual(residual_median: float, qc: QCResult) -> None:
    """
    Validate plane fit residual.

    Reference: agent-2.md rules.qc_fail_behavior

    Args:
        residual_median: Median residual from plane fit
        qc: QCResult to update
    """
    qc.residual_median = residual_median

    if residual_median > RESIDUAL_MAX_M:
        qc.add_flag(QCFlag.HIGH_RESIDUAL)
        qc.details["residual_issue"] = (
            f"Median residual {residual_median:.3f}m exceeds {RESIDUAL_MAX_M}m"
        )


def validate_vegetation(vegetation_pct: float, qc: QCResult) -> None:
    """
    Check vegetation percentage.

    Reference: agent-2.md io_schemas.metrics_json.qc.vegetation_pct

    Args:
        vegetation_pct: Percentage of points classified as vegetation
        qc: QCResult to update
    """
    qc.vegetation_pct = vegetation_pct

    if vegetation_pct > VEGETATION_MAX_PCT:
        qc.add_flag(QCFlag.HIGH_VEGETATION)
        qc.details["vegetation_issue"] = (
            f"Vegetation {vegetation_pct:.1f}% exceeds {VEGETATION_MAX_PCT}%"
        )


def validate_plan_area(plan_area_m2: float, qc: QCResult) -> None:
    """
    Validate plan area is reasonable.

    Reference: CLAUDE.md §4.2

    Args:
        plan_area_m2: Plan area in square meters
        qc: QCResult to update
    """
    if plan_area_m2 < PLAN_AREA_MIN_M2:
        qc.add_flag(QCFlag.AREA_TOO_SMALL)
        qc.details["area_issue"] = (
            f"Plan area {plan_area_m2:.1f}m² below minimum {PLAN_AREA_MIN_M2}m²"
        )

    if plan_area_m2 > PLAN_AREA_MAX_M2:
        qc.add_flag(QCFlag.AREA_TOO_LARGE)
        qc.details["area_issue"] = (
            f"Plan area {plan_area_m2:.1f}m² above maximum {PLAN_AREA_MAX_M2}m²"
        )


def validate_plane_fit(
    inlier_ratio: float,
    success: bool,
    qc: QCResult
) -> None:
    """
    Validate plane fitting results.

    Args:
        inlier_ratio: Ratio of inlier points (0-1)
        success: Whether fit succeeded
        qc: QCResult to update
    """
    qc.plane_fit_success = success

    if not success:
        qc.add_flag(QCFlag.PLANE_FIT_FAILED)
        qc.success = False

    if inlier_ratio < 0.6:
        qc.add_flag(QCFlag.LOW_INLIER_RATIO)
        qc.details["inlier_issue"] = (
            f"Inlier ratio {inlier_ratio:.1%} below 60% threshold"
        )


def validate_dsm_quality(ndsm_stats: Dict[str, float], qc: QCResult) -> None:
    """
    Validate DSM/nDSM quality.

    Reference: CLAUDE.md §4.2

    Args:
        ndsm_stats: Dict with min, max, mean, std
        qc: QCResult to update
    """
    qc.ndsm_stats = ndsm_stats

    # Check for excessive variance (noisy data)
    std = ndsm_stats.get("std", 0)
    mean = ndsm_stats.get("mean", 0)

    if mean > 0 and std / mean > 0.5:  # Coefficient of variation > 0.5
        qc.add_flag(QCFlag.DSM_NOISY)
        qc.details["dsm_issue"] = (
            f"High height variance (std={std:.2f}m, mean={mean:.2f}m)"
        )


def validate_point_count(n_points: int, min_points: int, qc: QCResult) -> None:
    """
    Validate sufficient points for processing.

    Args:
        n_points: Number of points
        min_points: Minimum required
        qc: QCResult to update
    """
    if n_points < min_points:
        qc.add_flag(QCFlag.INSUFFICIENT_POINTS)
        qc.details["points_issue"] = (
            f"Only {n_points} points, need at least {min_points}"
        )
        qc.success = False


def compute_qc_stats(
    ndsm: Optional[np.ndarray] = None,
    vegetation_pct: float = 0.0,
    facet_metrics: Optional[List] = None,
    plane_results: Optional[List] = None
) -> QCResult:
    """
    Compute comprehensive QC stats for a building.

    Reference: agent-2.md io_schemas.metrics_json.qc

    Args:
        ndsm: Normalized DSM array
        vegetation_pct: Vegetation percentage
        facet_metrics: List of FacetMetrics objects
        plane_results: List of PlaneModel objects

    Returns:
        QCResult with all validations
    """
    qc = QCResult()

    # nDSM stats
    if ndsm is not None:
        ndsm_stats = compute_ndsm_stats(ndsm)
        validate_dsm_quality(ndsm_stats, qc)

    # Vegetation
    validate_vegetation(vegetation_pct, qc)

    # Facet metrics validation
    if facet_metrics:
        total_area = 0
        for metrics in facet_metrics:
            validate_pitch(metrics.slope_deg, qc)
            total_area += metrics.plan_area_m2

        validate_plan_area(total_area, qc)

    # Plane fit validation
    if plane_results:
        for plane in plane_results:
            inlier_ratio = plane.inlier_count / max(1, len(plane.inlier_mask))
            validate_plane_fit(inlier_ratio, plane.success, qc)
            validate_residual(plane.residual_median, qc)

    # Update overall success
    critical_flags = {
        QCFlag.PLANE_FIT_FAILED,
        QCFlag.INSUFFICIENT_POINTS
    }
    if any(f in qc.flags for f in critical_flags):
        qc.success = False

    return qc


def validate_metrics(
    building_metrics: Dict[str, Any],
    ndsm: Optional[np.ndarray] = None,
    vegetation_pct: float = 0.0
) -> QCResult:
    """
    Validate a building's computed metrics.

    Convenience function for validating final output.

    Args:
        building_metrics: Dict from compute_building_metrics()
        ndsm: Optional nDSM array
        vegetation_pct: Vegetation percentage

    Returns:
        QCResult
    """
    qc = QCResult()

    # Validate nDSM
    if ndsm is not None:
        ndsm_stats = compute_ndsm_stats(ndsm)
        validate_dsm_quality(ndsm_stats, qc)

    # Validate vegetation
    validate_vegetation(vegetation_pct, qc)

    # Validate facets
    facets = building_metrics.get("facets", [])
    total_area = 0

    for facet in facets:
        slope_deg = facet.get("slope_deg", 0)
        validate_pitch(slope_deg, qc)

        residual = facet.get("residual_median_m", 0)
        validate_residual(residual, qc)

        total_area += facet.get("plan_area_m2", 0)

    validate_plan_area(total_area, qc)

    # Compute confidence based on flags
    n_flags = len(qc.flags)
    qc.confidence_score = max(0.0, 1.0 - (n_flags * 0.15))

    return qc


def format_qc_report(qc: QCResult) -> str:
    """
    Format QC result as human-readable report.

    Args:
        qc: QCResult object

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 50)
    lines.append("QUALITY CONTROL REPORT")
    lines.append("=" * 50)

    status = "PASS" if qc.success else "FAIL"
    lines.append(f"Status: {status}")
    lines.append(f"Confidence: {qc.confidence_score:.0%}")

    if qc.flags:
        lines.append("\nFlags:")
        for flag in qc.flags:
            lines.append(f"  - {flag.value}")

    if qc.ndsm_stats:
        lines.append("\nnDSM Statistics:")
        for key, value in qc.ndsm_stats.items():
            lines.append(f"  {key}: {value:.2f}m")

    lines.append(f"\nVegetation: {qc.vegetation_pct:.1f}%")
    lines.append(f"Plane fit success: {qc.plane_fit_success}")
    lines.append(f"Residual median: {qc.residual_median:.3f}m")

    if qc.details:
        lines.append("\nDetails:")
        for key, value in qc.details.items():
            lines.append(f"  {key}: {value}")

    lines.append("=" * 50)

    return "\n".join(lines)
