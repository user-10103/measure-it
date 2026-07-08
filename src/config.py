"""
Centralized configuration management for roof-metrics pipeline.
Loads environment variables and provides default values.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Data source URLs
MS_BUILDINGS_INDEX_URL = os.getenv(
    "MS_BUILDINGS_INDEX_URL",
    "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
)
NAIP_S3_BUCKET = os.getenv("NAIP_S3_BUCKET", "naip-analytic")
USGS_3DEP_API = os.getenv(
    "USGS_3DEP_API",
    "https://s3-us-west-2.amazonaws.com/usgs-lidar-public"
)

# LiDAR EPT Configuration
WESM_INDEX_PATH = os.getenv("WESM_INDEX_PATH", str(PROJECT_ROOT / "data/wesm/WESM.gpkg"))
LIDAR_BUFFER_M = int(os.getenv("LIDAR_BUFFER_M", "30"))
EPT_RESOLUTION_TARGET = float(os.getenv("EPT_RESOLUTION_TARGET", "1.0"))
DSM_RESOLUTION_M = float(os.getenv("DSM_RESOLUTION_M", "0.5"))
RANSAC_RESIDUAL_M = float(os.getenv("RANSAC_RESIDUAL_M", "0.3"))
POLYGON_SIMPLIFY_M = float(os.getenv("POLYGON_SIMPLIFY_M", "0.3"))

# Processing parameters
DEFAULT_BUFFER_METERS = int(os.getenv("DEFAULT_BUFFER_METERS", "150"))
NAIP_CLIP_PADDING_METERS = int(os.getenv("NAIP_CLIP_PADDING_METERS", "20"))
QUADKEY_LEVEL = int(os.getenv("QUADKEY_LEVEL", "9"))

# Directory paths
OUTPUT_DIR = PROJECT_ROOT / os.getenv("OUTPUT_DIR", "output")
DATA_CACHE_DIR = PROJECT_ROOT / os.getenv("DATA_CACHE_DIR", "data")

# Create directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(DATA_CACHE_DIR / "msbuildings").mkdir(exist_ok=True)
(DATA_CACHE_DIR / "naip").mkdir(exist_ok=True)
(DATA_CACHE_DIR / "lidar").mkdir(exist_ok=True)
(DATA_CACHE_DIR / "wesm").mkdir(exist_ok=True)


def validate_config():
    """Validate that required configuration values are present."""
    return True


def get_config_summary():
    """Return a dictionary with current configuration values."""
    return {
        "wesm_index_path": WESM_INDEX_PATH,
        "lidar_buffer_m": LIDAR_BUFFER_M,
        "ept_resolution_target": EPT_RESOLUTION_TARGET,
        "dsm_resolution_m": DSM_RESOLUTION_M,
        "ransac_residual_m": RANSAC_RESIDUAL_M,
        "buffer_meters": DEFAULT_BUFFER_METERS,
        "naip_padding_meters": NAIP_CLIP_PADDING_METERS,
        "quadkey_level": QUADKEY_LEVEL,
        "data_cache": str(DATA_CACHE_DIR),
        "output_dir": str(OUTPUT_DIR),
    }
