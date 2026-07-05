"""Two-story detection: ground (EPT class-2, OT DTM fallback) -> eave height
-> per-facet flag -> report field (replaces the hardcoded '0 sqft')."""
import io

import numpy as np
import pytest
from shapely.geometry import box

from src.roofs.fuse_sam_lidar import annotate_facets_with_lidar, fuse_into_report_input
from src.roofs.segment import Facet


def _grid(poly, z_fn, step=0.5):
    minx, miny, maxx, maxy = poly.bounds
    xs, ys = np.meshgrid(np.arange(minx + 0.25, maxx, step),
                         np.arange(miny + 0.25, maxy, step))
    x, y = xs.ravel(), ys.ravel()
    return np.column_stack([x, y, z_fn(x, y)])


def test_two_story_flag_from_eave_height():
    tall = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    low = Facet(facet_id=2, polygon=box(20, 0, 30, 10))
    pts = np.vstack([
        _grid(tall.polygon, lambda x, y: 6.5 + 0.2 * y),   # eave ~6.5 m
        _grid(low.polygon, lambda x, y: 3.0 + 0.2 * y),    # eave ~3.0 m
    ])
    ann = annotate_facets_with_lidar([tall, low], pts, ground_z=0.0)
    assert ann[1]["is_two_story"] and ann[1]["eave_height_m"] > 5.5
    assert not ann[2]["is_two_story"]


def test_no_ground_no_two_story_fields():
    f = Facet(facet_id=1, polygon=box(0, 0, 10, 10))
    pts = _grid(f.polygon, lambda x, y: 6.5 + 0.2 * y)
    ann = annotate_facets_with_lidar([f], pts)          # ground unknown
    assert "is_two_story" not in ann[1]                  # honest: not claimed


def test_two_story_reaches_the_report_model():
    from src.output.report_data import build_report_model
    ri = {"facets": [
        {"facet_id": 1, "polygon_xy": [[0, 0]], "plan_area_m2": 100.0,
         "pitch_string": "6:12", "is_flat": False, "two_story": True,
         "surface_area_m2": 110.0},
        {"facet_id": 2, "polygon_xy": [[0, 0]], "plan_area_m2": 50.0,
         "pitch_string": "6:12", "is_flat": False,
         "surface_area_m2": 55.0},
    ], "edges": []}
    model = build_report_model(ri)
    assert abs(model.two_story_area_sqft - 110.0 * 10.7639) < 1.0
    assert model.total_area_sqft > model.two_story_area_sqft


def test_ept_with_ground(monkeypatch):
    # reuse the synthetic EPT from test_ept_fetch, adding class-2 ground points
    import json as _json

    import src.lidar.ept_fetch as ef
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer
    import laspy

    CRS_UTM = "EPSG:32617"
    ROOT = [499000.0, 3099000.0, 0.0, 500024.0, 3100024.0, 1024.0]
    FP = box(499500.0, 3099500.0, 499540.0, 3099530.0)
    inv = Transformer.from_crs(CRS_UTM, "EPSG:4326", always_xy=True).transform
    fp_wgs = shp_transform(inv, FP)

    xs, ys = np.meshgrid(np.arange(499500.5, 499540.0, 1.0),
                         np.arange(3099500.5, 3099530.0, 1.0))
    x, y = xs.ravel(), ys.ravel()
    roof_z = np.full_like(x, 14.0)
    ground_z = np.full_like(x, 8.0)
    X = np.concatenate([x, x]); Y = np.concatenate([y, y])
    Z = np.concatenate([roof_z, ground_z])
    C = np.concatenate([np.full(len(x), 6), np.full(len(x), 2)]).astype(np.uint8)

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = [X.min(), Y.min(), Z.min()]
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x, las.y, las.z = X, Y, Z
    las.classification = C
    bio = io.BytesIO(); las.write(bio, do_compress=True)

    class _R:
        def __init__(s, p): s._p = p
        def json(s): return s._p if isinstance(s._p, dict) else _json.loads(s._p)
        @property
        def content(s): return s._p

    files = {"ept.json": _R({"bounds": ROOT, "dataType": "laszip",
                             "srs": {"authority": "EPSG", "horizontal": "32617"}}),
             "ept-hierarchy/0-0-0-0.json": _R({"0-0-0-0": len(X)}),
             "ept-data/0-0-0-0.laz": _R(bio.getvalue())}
    monkeypatch.setattr(ef, "_get", lambda url, timeout=60: next(
        v for k, v in files.items() if url.endswith(k)))

    pts, gz = ef.fetch_roof_points(28.0, -81.0, fp_wgs, CRS_UTM,
                                   ept_url="https://fake/ept/ept.json",
                                   with_ground=True)
    assert pts is not None and abs(gz - 8.0) < 0.01
    assert pts[:, 2].min() > 13.0                        # ground points dropped


def test_opentopo_ground(monkeypatch, tmp_path):
    # fake a tiny GeoTIFF DTM response
    import rasterio
    from rasterio.transform import from_origin

    tif = tmp_path / "dtm.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=4, height=4, count=1,
                       dtype="float32", crs="EPSG:4326",
                       transform=from_origin(-80.7, 28.04, 0.0002, 0.0002)) as dst:
        dst.write(np.full((1, 4, 4), 7.5, np.float32))
    payload = tif.read_bytes()

    class _R:
        status_code = 200
        content = payload
    import requests as _requests
    monkeypatch.setattr(_requests, "get", lambda *a, **k: _R())
    monkeypatch.setenv("OPENTOPO_API_KEY", "test-key")

    from src.lidar.opentopo import get_ground_elevation
    assert abs(get_ground_elevation(28.03, -80.69) - 7.5) < 0.01


def test_opentopo_without_key_returns_none(monkeypatch):
    monkeypatch.delenv("OPENTOPO_API_KEY", raising=False)
    from src.lidar.opentopo import get_ground_elevation
    assert get_ground_elevation(28.03, -80.69) is None
