# Roof-Metrics MVP

Automated roof identification, polygon extraction, elevation fusion, and metrics output.

## Overview

This pipeline automatically:
1. Geocodes addresses to coordinates
2. Fetches building footprints from Microsoft Buildings dataset
3. Downloads aerial imagery from NAIP
4. Discovers and fetches LiDAR point clouds via EPT (Entwine Point Tiles)
5. Refines roof polygons using RANSAC plane fitting
6. Calculates accurate area and perimeter measurements

## Quick Start

```bash
# 1. Install system dependencies
# macOS (requires Homebrew):
brew install gdal pdal

# Ubuntu/Debian:
sudo apt-get install gdal-bin pdal

# 2. Install Python dependencies (using uv - recommended)
uv sync .

# Or using pip
pip install -e .

# 3. Download WESM spatial index (2.8GB - one-time download)
# Download from: https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/metadata/WESM.gpkg
# Place it in: data/wesm/WESM.gpkg

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_MAPS_API_KEY
# Verify WESM_INDEX_PATH points to data/wesm/WESM.gpkg

# 5. Run healthcheck
measure-it healthcheck

# 6. Process an address
measure-it \
    --address "16347 Heathrow Dr, Tampa, FL 33647" \
    --state fl
```

## Documentation

- **[Setup Guide](docs/setup.md)** - Complete installation and configuration
- **[CLAUDE.md](CLAUDE.md)** - Engineering taskboard and coding rules
- **[architecture.md](architecture.md)** - System architecture and design
- **[experiment-1.ipynb](experiment-1.ipynb)** - Proof-of-concept notebook

## Key Features

- ✅ **Automatic data acquisition** - No manual downloads required
- ✅ **Smart caching** - Downloads once, uses forever
- ✅ **Multi-source fusion** - Combines MS Buildings + NAIP + LiDAR EPT
- ✅ **EPT point cloud access** - Direct access to USGS 3DEP LiDAR via EPT
- ✅ **RANSAC plane fitting** - Robust roof surface extraction
- ✅ **Sub-meter accuracy** - Point cloud resolution for precise measurements
- ✅ **WESM spatial index** - Instant LiDAR dataset discovery
- ✅ **Geodesic measurements** - Accurate area/perimeter calculations
- ✅ **Modular design** - Clean, testable, production-ready code

## Project Structure

```
measure-it/
├── src/                    # Source code
│   ├── pipeline.py         # Main orchestration
│   ├── ingestion/          # Data acquisition modules
│   ├── lidar/              # EPT LiDAR processing
│   ├── roofs/              # Roof analysis modules
│   └── utils/              # Utility functions
├── tests/                  # Test suite
├── docs/                   # Documentation
├── data/                   # Data cache (gitignored)
│   └── wesm/               # WESM spatial index
└── output/                 # Results (gitignored)
```

## Example Output

```json
{
  "measurements": {
    "plan_area_m2": 245.32,
    "perimeter_m": 62.45,
    "n_vertices": 8,
    "source": "ept_lidar_refined"
  },
  "polygon": {
    "source": "ept_lidar_refined",
    "file": "output/roof_polygon.geojson"
  },
  "lidar": {
    "enabled": true,
    "refined": true,
    "dataset_name": "FL_Peninsular_Hernando_2019",
    "point_count": 1247
  },
  "building": {
    "distance_from_pin_m": 0.0,
    "n_candidates": 3
  }
}
```

## Requirements

- Python 3.10+
- GDAL (system-level)
- PDAL (system-level)
- WESM.gpkg spatial index (2.8GB - see setup)
- Google Maps API key
- AWS credentials (for requester-pays S3 access)

## Performance

- **First run**: 2-5 minutes (includes data downloads)
- **Cached runs**: 30-60 seconds
- **Accuracy**: Plan area <3%, Edge lengths <3%, Pitch <5%

## Development Status

**Current Phase**: Deliverables 1 & 2 Complete
- ✅ Data Acquisition & Environment Setup
- ✅ Roof Identification & Extraction
- ✅ Phase 3: Elevation Fusion (complete)
- ⏳ Phase 4: Roof Metrics & QC (research)
- ⏳ Phase 5: Final Outputs (planned)

## License

See project license file.

## Data Attribution

- **Microsoft Buildings Footprints US** - Building footprints
- **USDA NAIP Aerial Imagery** - High-resolution aerial imagery
- **USGS 3D Elevation Program (3DEP)** - LiDAR point clouds via EPT
- **USGS WESM** - Watershed Elevation Model spatial index for LiDAR dataset discovery
