"""Tests for PDAL CRS propagation through the LiDAR pipeline.

Missing optional dependencies (boto3, geopandas, pdal, pandas) are stubbed in
sys.modules BEFORE any of the modules under test are imported.

Key isolation strategy
----------------------
src/lidar/__init__.py imports dataset_discovery which needs boto3+pandas+geopandas.
To avoid running __init__.py we pre-register src.lidar as an empty package module
in sys.modules.  Python then loads src/lidar/ept_client.py directly without
re-executing __init__.py.

We do NOT stub scipy or sklearn because they are installed in the test venv.
Stubbing sklearn's pandas proxy breaks other test modules (narwhals calls
pd.DataFrame at RANSAC fit time and the stub can only partly satisfy it).
"""
import sys
import types
import unittest.mock as mock

import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.ops import transform as shp_transform

# ---------------------------------------------------------------------------
# Stub genuinely missing packages
# ---------------------------------------------------------------------------

def _stub(name, **attrs):
    """Add a stub module to sys.modules only if the package is not installed."""
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


botocore_exc = _stub("botocore.exceptions", ClientError=type("ClientError", (Exception,), {}))
_stub("botocore", exceptions=botocore_exc)
_stub("boto3")
# sklearn's narwhals calls pd.DataFrame and pd.Series at RANSAC fit() time to check
# whether the input is a dataframe.  Numpy arrays are not instances of these stubs,
# so isinstance() returns False and the fit proceeds normally.
_stub("pandas",
      DataFrame=type("DataFrame", (), {}),
      Series=type("Series", (), {}))

# geopandas: needs GeoSeries with real to_crs() via pyproj (for the roundtrip test)
class _GeoSeries:
    def __init__(self, geoms, crs=None):
        self._geoms = list(geoms)
        self._crs = crs

    def to_crs(self, target_crs):
        from pyproj import CRS, Transformer
        src = CRS.from_user_input(self._crs)
        tgt = CRS.from_user_input(target_crs)
        xfm = Transformer.from_crs(src, tgt, always_xy=True).transform
        return _GeoSeries([shp_transform(xfm, g) for g in self._geoms], target_crs)

    @property
    def iloc(self):
        geoms = self._geoms
        class _Iloc:
            def __getitem__(self, idx): return geoms[idx]
        return _Iloc()

_stub("geopandas", GeoSeries=_GeoSeries, GeoDataFrame=mock.MagicMock())

# pdal: not installed in test venv
_stub("pdal", Pipeline=mock.MagicMock())

# ---------------------------------------------------------------------------
# Prevent src/lidar/__init__.py from running (it imports dataset_discovery
# which needs boto3+pandas+geopandas).  By pre-registering src.lidar as a
# package stub whose __path__ points at the real directory, Python finds
# ept_client.py but skips __init__.py (it's already "loaded").
# ---------------------------------------------------------------------------
if "src.lidar" not in sys.modules:
    import os as _os
    _lidar_pkg = types.ModuleType("src.lidar")
    _lidar_pkg.__path__ = [
        _os.path.join(_os.path.dirname(__file__), "..", "src", "lidar")
    ]
    _lidar_pkg.__package__ = "src.lidar"
    sys.modules["src.lidar"] = _lidar_pkg

# ---------------------------------------------------------------------------
# Import modules under test
# ---------------------------------------------------------------------------
import src.lidar.ept_client as ept_client          # noqa: E402
from src.roofs import extract_polygon as ep         # noqa: E402

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

# EPSG:32617 = WGS84 / UTM zone 17N — the actual projection Florida EPT tiles use.
# (EPSG:6347 is zone 18N, not 17N — don't use it for Florida coordinates.)
UTM17N_WKT = "EPSG:32617"


def _make_fake_pdal_pipeline(n_points=10, srswkt=UTM17N_WKT):
    """Return a mock that quacks like pdal.Pipeline."""
    dtype = np.dtype([
        ("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Classification", "u1")
    ])
    pts = np.zeros(n_points, dtype=dtype)
    pts["X"] = 490_000.0 + np.arange(n_points)  # UTM 17N easting for Tampa area
    pts["Y"] = 3_110_000.0 + np.arange(n_points)
    pts["Z"] = 10.0 + np.arange(n_points) * 0.1

    p = mock.MagicMock()
    p.execute.return_value = n_points
    p.arrays = [pts]
    p.srswkt = srswkt
    return p


# ---------------------------------------------------------------------------
# fetch_lidar_points — return signature
# ---------------------------------------------------------------------------

def test_fetch_lidar_points_returns_tuple(monkeypatch):
    """fetch_lidar_points must return (array, wkt_string), not a bare array."""
    fake = _make_fake_pdal_pipeline()
    monkeypatch.setattr(ept_client.pdal, "Pipeline", lambda _json: fake)

    result = ept_client.fetch_lidar_points(
        "https://example.com/ept.json", "POLYGON(...)/EPSG:4326"
    )
    assert isinstance(result, tuple) and len(result) == 2
    points, srswkt = result
    assert points is not None and len(points) == 10
    assert isinstance(srswkt, str) and len(srswkt) > 0


def test_fetch_lidar_points_srswkt_matches_pipeline(monkeypatch):
    """The returned srswkt must be exactly what pipeline.srswkt reported."""
    fake = _make_fake_pdal_pipeline(srswkt=UTM17N_WKT)
    monkeypatch.setattr(ept_client.pdal, "Pipeline", lambda _json: fake)

    _, srswkt = ept_client.fetch_lidar_points(
        "https://x/ept.json", "POLYGON(...)/EPSG:4326"
    )
    assert srswkt == UTM17N_WKT


def test_fetch_lidar_points_empty_returns_none_tuple(monkeypatch):
    """Zero-point EPT response must return (None, None)."""
    fake = mock.MagicMock()
    fake.execute.return_value = 0
    monkeypatch.setattr(ept_client.pdal, "Pipeline", lambda _json: fake)

    points, srswkt = ept_client.fetch_lidar_points(
        "https://x/ept.json", "POLYGON(...)/EPSG:4326"
    )
    assert points is None
    assert srswkt is None


# ---------------------------------------------------------------------------
# get_ept_lidar_for_location — points_crs present in result dict
# ---------------------------------------------------------------------------

def test_get_ept_lidar_for_location_propagates_crs(monkeypatch):
    """Result dict from get_ept_lidar_for_location must contain 'points_crs'."""
    fake = _make_fake_pdal_pipeline(srswkt=UTM17N_WKT)
    monkeypatch.setattr(ept_client.pdal, "Pipeline", lambda _json: fake)

    dummy_poly = Polygon([
        (490000, 3110000), (490050, 3110000),
        (490050, 3110030), (490000, 3110030),
    ])
    monkeypatch.setattr(
        ept_client, "build_dsm_from_points",
        lambda pts: (np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))
    )
    monkeypatch.setattr(ept_client, "extract_roof_polygon", lambda gx, gy, gz: dummy_poly)

    building = Polygon([(-82.40, 28.11), (-82.39, 28.11),
                        (-82.39, 28.12), (-82.40, 28.12)])
    result = ept_client.get_ept_lidar_for_location(
        lat=28.1178, lon=-82.3951,
        building_polygon_wgs84=building,
        ept_url="https://example.com/ept.json",
    )

    assert result is not None
    assert "points_crs" in result, f"keys: {list(result.keys())}"
    assert result["points_crs"] == UTM17N_WKT


# ---------------------------------------------------------------------------
# refine_with_ept_lidar — uses points_crs, not aeqd_crs
# ---------------------------------------------------------------------------

def test_refine_with_ept_lidar_uses_points_crs(monkeypatch):
    """refine_with_ept_lidar must reproject using the EPT CRS (UTM 17N), not AEQD.

    The test works in three steps:
    1. Convert a known Tampa WGS84 point to EPSG:32617 (WGS84 / UTM 17N).
    2. Feed those UTM coordinates as the "EPT roof polygon".
    3. Verify the WGS84 output lands within 100 m of the original point.

    Under the old AEQD bug, the UTM easting/northing (~363k, ~3111k) would be
    interpreted as metres from the AEQD origin (lat/lon), putting the result
    ~363 km east and ~3111 km north of Tampa — obviously wrong.
    """
    from pyproj import CRS, Transformer

    # Canonical Tampa reference point
    ref_lon, ref_lat = -82.395, 28.12

    # Convert to UTM 17N to get realistic EPT coordinates
    utm_crs = CRS.from_epsg(32617)
    t_fwd = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True).transform
    cx_utm, cy_utm = t_fwd(ref_lon, ref_lat)

    # Build a small test polygon around the reference point in UTM space
    utm_poly = Polygon([
        (cx_utm,      cy_utm),
        (cx_utm + 50, cy_utm),
        (cx_utm + 50, cy_utm + 30),
        (cx_utm,      cy_utm + 30),
    ])

    building = Polygon([(-82.40, 28.11), (-82.39, 28.11),
                        (-82.39, 28.12), (-82.40, 28.12)])
    fake_lidar = {
        "points": np.zeros(10),
        "points_crs": "EPSG:32617",
        "grid_x": np.zeros((5, 5)),
        "grid_y": np.zeros((5, 5)),
        "grid_z": np.zeros((5, 5)),
        "roof_polygon": utm_poly,
        "dataset_name": "test",
    }
    monkeypatch.setattr(ep, "get_ept_lidar_for_location", lambda **kw: fake_lidar)

    refined = ep.refine_with_ept_lidar(
        building_polygon_wgs84=building,
        lat=ref_lat, lon=ref_lon,
        ept_url="https://example.com/ept.json",
    )

    out_cx = refined["refined_polygon_wgs84"].centroid.x
    out_cy = refined["refined_polygon_wgs84"].centroid.y

    # Must be in Florida, not somewhere random (~490 km off as the AEQD bug produced)
    assert -85 < out_cx < -79, f"Longitude out of Florida range: {out_cx:.4f}"
    assert 24 < out_cy < 31, f"Latitude out of Florida range: {out_cy:.4f}"
    # Must be within ~0.1° of the reference point (100 m precision)
    assert abs(out_cx - ref_lon) < 0.1, f"Longitude too far from reference: {out_cx:.4f}"
    assert abs(out_cy - ref_lat) < 0.1, f"Latitude too far from reference: {out_cy:.4f}"
