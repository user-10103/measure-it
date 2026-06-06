"""
LiDAR processing package for roof polygon extraction.

Provides EPT-based point cloud access and RANSAC-based roof extraction.
"""

from src.lidar.dataset_discovery import (
    construct_ept_url,
    discover_ept_via_s3_exploration,
    discover_lidar_dataset,
    validate_ept_url,
)
from src.lidar.ept_client import (
    build_dsm_from_points,
    build_pdal_pipeline,
    extract_roof_polygon,
    fetch_lidar_points,
    get_ept_lidar_for_location,
)

__all__ = [
    "discover_lidar_dataset",
    "construct_ept_url",
    "validate_ept_url",
    "discover_ept_via_s3_exploration",
    "build_pdal_pipeline",
    "fetch_lidar_points",
    "build_dsm_from_points",
    "extract_roof_polygon",
    "get_ept_lidar_for_location",
]
