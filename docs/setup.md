# Roof-Metrics Setup Guide

Complete installation and configuration guide for the roof-metrics pipeline.

## Prerequisites

- **Python**: 3.10 or higher
- **Operating System**: macOS, Linux, or Windows (WSL recommended)
- **Disk Space**: ~5GB minimum for data cache
- **Network**: Stable internet connection (AWS requester-pays access)

## System Dependencies

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install GDAL and PDAL
brew install gdal pdal
```

### Ubuntu/Debian Linux

```bash
sudo apt-get update
sudo apt-get install -y \
    gdal-bin \
    libgdal-dev \
    pdal \
    python3-pip \
    python3-dev \
    build-essential
```

### Windows (WSL)

Follow Ubuntu instructions after installing WSL2:
```powershell
wsl --install
```

## Python Environment Setup

### 1. Clone/Navigate to Project

```bash
cd /path/to/measure-it
```

### 2. Install Dependencies

**Option A: Using uv (Recommended - Fast!)**

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the package in development mode
uv sync .

# Or install with development dependencies
uv sync --dev"
```

**Option B: Using pip (Traditional)**

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

**Option C: Using requirements.txt (Fallback)**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -m src.healthcheck

# Or use the installed CLI command
measure-it-healthcheck
```

Expected output:
```
============================================================
ROOF-METRICS SYSTEM HEALTH CHECK
============================================================

[1] Python Version
------------------------------------------------------------
✓ Python 3.11.x

[2] Python Packages
------------------------------------------------------------
✓ geopandas          0.14.x
✓ shapely            2.0.x
✓ pyproj             3.6.x
...

[3] System Tools
------------------------------------------------------------
✓ gdal-config        GDAL 3.x.x
✓ pdal               2.x.x

============================================================
✓ ALL REQUIRED CHECKS PASSED
============================================================
```

## Configuration

### 1. Copy Environment Template

```bash
cp .env.example .env
```

### 2. Configure API Keys

Edit `.env` and add your Google Maps API key:

```bash
GOOGLE_MAPS_API_KEY=your_actual_api_key_here
```

**Get a Google Maps API Key:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable "Geocoding API"
4. Create credentials (API Key)
5. Copy the key to your `.env` file

### 3. Verify Configuration

```bash
python -c "from src.config import validate_config; validate_config(); print('Config OK')"
```

## AWS Configuration (Optional)

The pipeline accesses public AWS datasets with "requester-pays" enabled. Your AWS credentials will be used for billing.

**Option 1: AWS CLI (Recommended)**
```bash
pip install awscli
aws configure
```

**Option 2: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

**Option 3: IAM Role** (for EC2/Lambda)
- Attach role with S3 read permissions
- No explicit configuration needed

## Quick Start Test

Run a test address to verify everything works:

```bash
# Using the CLI command (if installed with -e)
measure-it \
    --address "16347 Heathrow Dr, Tampa, FL 33647" \
    --state fl \
    --output output/test

# Or using the module directly
python -m src.pipeline \
    --address "16347 Heathrow Dr, Tampa, FL 33647" \
    --state fl \
    --output output/test
```

Expected runtime: 2-4 minutes

Expected output:
```
============================================================
SUCCESS!
============================================================
Plan area: 245.32 sq m
Perimeter: 62.45 m
Source: lidar_refined
============================================================
```

## Project Structure

```
measure-it/
├── src/
│   ├── config.py              # Configuration management
│   ├── pipeline.py            # Main orchestration
│   ├── healthcheck.py         # Dependency verification
│   ├── ingestion/             # Data acquisition
│   │   ├── ms_footprints.py   # Microsoft Buildings
│   │   ├── naip.py            # Aerial imagery
│   │   └── dem_client.py      # Elevation data (DEM)
│   ├── roofs/                 # Roof analysis
│   │   ├── select_candidates.py
│   │   ├── extract_polygon.py
│   │   └── measurements.py
│   └── utils/                 # Utilities
│       ├── geocode.py
│       ├── projections.py
│       └── cache.py
├── data/                      # Cached downloads (gitignored)
├── output/                    # Results (gitignored)
├── tests/                     # Test suite
├── docs/                      # Documentation
├── pyproject.toml             # Project metadata and dependencies
├── requirements.txt           # Alternative pip dependencies
└── .env                       # Configuration (gitignored)
```

## Usage Examples

### Basic Usage (Address)

```bash
# Using CLI command
measure-it \
    --address "123 Main St, City, State ZIP" \
    --state fl

# Or using module
python -m src.pipeline \
    --address "123 Main St, City, State ZIP" \
    --state fl
```

### Using Explicit Coordinates

```bash
measure-it \
    --lat 28.1178 \
    --lon -82.3951 \
    --state fl
```

### Skip Optional Data Sources

```bash
measure-it \
    --address "123 Main St, City, State ZIP" \
    --state fl \
    --no-naip \
    --no-lidar
```

### Custom Output Directory

```bash
measure-it \
    --address "123 Main St, City, State ZIP" \
    --state fl \
    --output /path/to/output
```

## Output Files

The pipeline generates these files in the output directory:

- `results.json` - Complete results with all metrics
- `roof_polygon.geojson` - Final roof polygon
- `candidates.geojson` - Top 3 building candidates
- `naip_chip.tif` - Clipped aerial imagery (if available)
- `naip_chip_rgb.png` - RGB preview
- `naip_chip_metadata.json` - Imagery metadata

## Troubleshooting

### Missing GDAL

**Error:** `ImportError: gdal`

**Solution:**
```bash
# macOS
brew install gdal
pip install gdal==$(gdal-config --version)

# Ubuntu
sudo apt-get install gdal-bin libgdal-dev
pip install gdal==$(gdal-config --version | cut -d. -f1,2).*
```

### PDAL Import Error

**Error:** `ImportError: pdal`

**Solution:**
```bash
# macOS
brew install pdal
pip install pdal

# Ubuntu
sudo apt-get install pdal
pip install pdal
```

### AWS Requester-Pays Error

**Error:** `botocore.exceptions.ClientError: An error occurred (403)`

**Solution:**
- Ensure AWS credentials are configured
- Verify credentials have S3 read permissions
- Set environment variable: `export AWS_REQUEST_PAYER=requester`

### Google Geocoding Error

**Error:** `GOOGLE_MAPS_API_KEY is required`

**Solution:**
- Create `.env` file from `.env.example`
- Add valid Google Maps API key
- Enable Geocoding API in Google Cloud Console

### No Buildings Found

**Error:** `ValueError: No buildings found within X meters`

**Solution:**
- Verify coordinates are correct
- Increase search radius: `--buffer 300`
- Check if location is in MS Buildings dataset coverage

## Performance Notes

- **First run**: Slower due to dataset downloads (~2-5 minutes)
- **Subsequent runs**: Faster with cached data (~30-60 seconds)
- **Cache location**: `data/` directory
- **Cache size**: Varies, typically 100MB-2GB per location

## Data Attribution

This pipeline uses publicly available datasets:

- **Microsoft Global Buildings**: ML-derived building footprints
- **NAIP**: USDA National Agriculture Imagery Program
- **USGS 3DEP**: USGS 3D Elevation Program LiDAR

Please review and comply with each dataset's usage terms.

## Next Steps

- Review [CLAUDE.md](../CLAUDE.md) for coding rules
- Review [architecture.md](../architecture.md) for system design
- Run tests: `pytest tests/ -v`
- Process multiple addresses with a script
- Integrate into your application

## Support

For issues or questions:
1. Check this documentation
2. Review error logs (use `--log-level DEBUG`)
3. Run healthcheck: `python -m src.healthcheck`
4. Check experiment notebook for reference: `experiment-1.ipynb`
