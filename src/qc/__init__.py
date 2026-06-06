"""
Quality Control Module

Validates roof metrics and flags anomalies.

Reference: agent-2.md qc section, CLAUDE.md §4.2
"""

from src.qc.validators import (
    validate_metrics,
    compute_qc_stats,
    QCResult,
    QCFlag
)

__all__ = [
    "validate_metrics",
    "compute_qc_stats",
    "QCResult",
    "QCFlag"
]
