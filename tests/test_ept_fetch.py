"""Pure-python EPT fetch against a synthetic in-memory EPT archive."""
import io
import json
import math

import numpy as np
import pytest
from shapely.geometry import box
from shapely.ops import transform as shp_transform

import src.lidar.ept_fetch as ef

CRS_UTM = "EPSG:32617"
# cubic EPT bounds around a fake neighborhood in UTM 17N
ROOT = [499000.0, 3099000.0, 0.0, 500024.0, 3100024.0, 1024.0]
FP_UTM = box(499500.0, 3099500.0, 499540.0, 3099530.0)     # 40 x 30 m building


def _fp_wgs84():
    from pyproj import Transformer
    inv = Transformer.from_crs(CRS_UTM, "EPSG:4326", always_xy=True).transform
    return shp_transform(inv, FP_UTM)


def _laz_bytes(x, y, z, classification=6):
    import laspy
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = [float(np.min(x)), float(np.min(y)), float(np.min(z))]
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.classification = np.full(len(x), classification, np.uint8)
    bio = io.BytesIO()
    las.write(bio, do_compress=True)
    return bio.getvalue()


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return json.loads(self._p) if isinstance(self._p, (bytes, str)) else self._p

    @property
    def content(self):
        return self._p


@pytest.fixture
def fake_ept(monkeypatch):
    """A one-node EPT with a 6:12 sloped roof plane inside FP_UTM."""
    xs, ys = np.meshgrid(np.arange(499500.5, 499540.0, 0.5),
                         np.arange(3099500.5, 3099530.0, 0.5))
    x, y = xs.ravel(), ys.ravel()
    z = 0.5 * (x - 499500.0) + 10.0                      # gradient 0.5 -> 6:12
    files = {
        "ept.json": _Resp({"bounds": ROOT, "dataType": "laszip",
                           "srs": {"authority": "EPSG", "horizontal": "32617"}}),
        "ept-hierarchy/0-0-0-0.json": _Resp({"0-0-0-0": len(x)}),
        "ept-data/0-0-0-0.laz": _Resp(_laz_bytes(x, y, z)),
    }

    def fake_get(url, timeout=60):
        for suffix, resp in files.items():
            if url.endswith(suffix):
                return resp
        raise AssertionError(f"unexpected URL {url}")
    monkeypatch.setattr(ef, "_get", fake_get)
    return files


def test_fetch_returns_roof_points_in_target_crs(fake_ept):
    pts = ef.fetch_roof_points(28.0, -81.0, _fp_wgs84(), CRS_UTM,
                               ept_url="https://fake/ept/ept.json")
    assert pts is not None and pts.shape[1] == 3
    # clipped to the (buffered) footprint
    assert pts[:, 0].min() > 499490 and pts[:, 0].max() < 499550
    # the slope survived the round trip: fit z = a*x + ...
    a = np.polyfit(pts[:, 0], pts[:, 2], 1)[0]
    assert abs(a - 0.5) < 0.02


def test_end_to_end_pitch_from_ept(fake_ept):
    """EPT tiles -> fetch -> fuse -> '6:12' on a SAM facet. The full LiDAR leg."""
    from src.roofs.fuse_sam_lidar import annotate_facets_with_lidar
    from src.roofs.segment import Facet

    pts = ef.fetch_roof_points(28.0, -81.0, _fp_wgs84(), CRS_UTM,
                               ept_url="https://fake/ept/ept.json")
    facet = Facet(facet_id=1, polygon=FP_UTM)
    ann = annotate_facets_with_lidar([facet], pts)
    assert ann[1]["pitch_string"] == "6:12"
    assert abs(ann[1]["slope_deg"] - math.degrees(math.atan(0.5))) < 0.6


def test_hierarchy_pruning(monkeypatch, fake_ept):
    # a node box far from the footprint must not be downloaded
    calls = []
    real_get = ef._get

    def spy(url, timeout=60):
        calls.append(url)
        return real_get(url, timeout)
    monkeypatch.setattr(ef, "_get", spy)
    ef.fetch_roof_points(28.0, -81.0, _fp_wgs84(), CRS_UTM,
                         ept_url="https://fake/ept/ept.json")
    assert not any("ept-data" in u and "0-0-0-0" not in u for u in calls)


def test_no_intersecting_nodes_returns_none(monkeypatch, fake_ept):
    from shapely.geometry import box as _box
    from pyproj import Transformer
    inv = Transformer.from_crs(CRS_UTM, "EPSG:4326", always_xy=True).transform
    far = shp_transform(inv, _box(400000, 3000000, 400040, 3000030))  # off-cube
    pts = ef.fetch_roof_points(27.0, -82.0, far, CRS_UTM,
                               ept_url="https://fake/ept/ept.json")
    assert pts is None


def test_node_bbox_math():
    b = ef._node_bbox("1-1-0-0", [0, 0, 0, 100, 100, 100])
    assert b == (50.0, 0.0, 100.0, 50.0)
