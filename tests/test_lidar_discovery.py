"""EPT source selection: freshness, collection-year ranking, and fallback.

Every test here is offline — the entwine index and the EPT HTTP fetches are mocked.
"""
import io
import json
import time

import numpy as np
import pytest
from shapely.geometry import box
from shapely.ops import transform as shp_transform

import src.lidar.dataset_discovery as dd
import src.lidar.ept_fetch as ef

CRS_UTM = "EPSG:32617"
ROOT = [499000.0, 3099000.0, 0.0, 500024.0, 3100024.0, 1024.0]
FP_UTM = box(499500.0, 3099500.0, 499540.0, 3099530.0)


def _fp_wgs84():
    from pyproj import Transformer
    inv = Transformer.from_crs(CRS_UTM, "EPSG:4326", always_xy=True).transform
    return shp_transform(inv, FP_UTM)


# --------------------------------------------------------------------------
# 2. collection-year parsing (the bug that picked the wrong survey for Sarno)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    # collected 2017, LAS released 2019 — the trailing year is NOT the collection
    ("USGS_LPC_FL_Upper_Saint_Johns_2017_LAS_2019", 2017),
    ("FL_Peninsular_FDEM_Brevard_2018", 2018),
    ("FL_Elgin_2006_2008", 2008),        # a range -> the year it finished
    ("FL_LPC_2019", 2019),               # LPC is a product tag, not a release marker
    ("NoYearAtAll", 0),
    ("FL_Old_1975", 0),                  # implausibly old -> ignored
])
def test_collection_year_rule(name, expected):
    assert dd._collection_year(name) == expected


def test_release_year_no_longer_outranks_a_newer_collection():
    """The exact Sarno mis-selection: a 2017 survey whose LAS shipped in 2019 must
    NOT outrank a genuine 2018 collection."""
    older_but_late_release = "USGS_LPC_FL_Upper_Saint_Johns_2017_LAS_2019"
    newer = "FL_Peninsular_FDEM_Brevard_2018"
    assert dd._collection_year(newer) > dd._collection_year(older_but_late_release)


# --------------------------------------------------------------------------
# index fixture
# --------------------------------------------------------------------------
def _index(*names_with_points):
    """A FeatureCollection whose every polygon covers (lon=-81, lat=28)."""
    feats = []
    for entry in names_with_points:
        name, pts = entry if isinstance(entry, tuple) else (entry, None)
        props = {"name": name}
        if pts is not None:
            props["points"] = pts
        feats.append({
            "type": "Feature", "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [[
                [-82.0, 27.0], [-80.0, 27.0], [-80.0, 29.0], [-82.0, 29.0], [-82.0, 27.0]]]},
        })
    return {"type": "FeatureCollection", "features": feats}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Point the module at a throwaway cache file."""
    p = tmp_path / "ept_resources.geojson"
    monkeypatch.setattr(dd, "ENTWINE_CACHE", p)
    monkeypatch.delenv("MEASURE_IT_ENTWINE_REFRESH", raising=False)
    monkeypatch.delenv("MEASURE_IT_ENTWINE_TTL_DAYS", raising=False)
    return p


def _write(cache, payload):
    cache.write_text(json.dumps(payload))


# --------------------------------------------------------------------------
# 3. deterministic ranking + 5. ranked candidate list
# --------------------------------------------------------------------------
def test_candidates_ranked_newest_collection_first(cache):
    _write(cache, _index("USGS_LPC_FL_Upper_Saint_Johns_2017_LAS_2019",
                         "FL_Peninsular_FDEM_Brevard_2018"))
    got = dd.discover_ept_candidates(28.0, -81.0)
    assert [c["name"] for c in got] == ["FL_Peninsular_FDEM_Brevard_2018",
                                        "USGS_LPC_FL_Upper_Saint_Johns_2017_LAS_2019"]
    assert got[0]["year"] == 2018 and got[1]["year"] == 2017


def test_ties_break_on_density_then_name_deterministically(cache):
    # same collection year: denser index wins; equal density falls back to name
    _write(cache, _index(("FL_Zulu_2020", 900), ("FL_Alpha_2020", 100),
                         ("FL_Mike_2020", 900)))
    names = [c["name"] for c in dd.discover_ept_candidates(28.0, -81.0)]
    assert names == ["FL_Mike_2020", "FL_Zulu_2020", "FL_Alpha_2020"]
    # and it is stable across calls
    assert names == [c["name"] for c in dd.discover_ept_candidates(28.0, -81.0)]


def test_non_covering_datasets_excluded(cache):
    payload = _index("FL_Covers_2020")
    payload["features"].append({
        "type": "Feature", "properties": {"name": "FL_Elsewhere_2021"},
        "geometry": {"type": "Polygon", "coordinates": [[
            [-70.0, 40.0], [-69.0, 40.0], [-69.0, 41.0], [-70.0, 41.0], [-70.0, 40.0]]]},
    })
    _write(cache, payload)
    assert [c["name"] for c in dd.discover_ept_candidates(28.0, -81.0)] == ["FL_Covers_2020"]


def test_from_entwine_stays_backward_compatible(cache):
    """Two external callers still expect a single URL string or None."""
    _write(cache, _index("FL_Peninsular_FDEM_Brevard_2018"))
    url = dd.discover_ept_from_entwine(28.0, -81.0)
    assert isinstance(url, str) and url.endswith("/FL_Peninsular_FDEM_Brevard_2018/ept.json")
    _write(cache, _index())                       # nothing covers the point
    assert dd.discover_ept_from_entwine(28.0, -81.0) is None


# --------------------------------------------------------------------------
# 1. cache freshness
# --------------------------------------------------------------------------
def test_fresh_cache_is_not_redownloaded(cache, monkeypatch):
    _write(cache, _index("FL_A_2020"))
    def boom(*a, **k):
        raise AssertionError("should not re-download a fresh cache")
    monkeypatch.setattr("requests.get", boom)
    assert dd.discover_ept_candidates(28.0, -81.0)


def test_stale_cache_triggers_refresh(cache, monkeypatch):
    _write(cache, _index("FL_Old_2010"))
    old = time.time() - 40 * 86400                       # 40 days > 30-day TTL
    import os as _os
    _os.utime(cache, (old, old))
    calls = []

    class _R:
        content = json.dumps(_index("FL_Fresh_2024")).encode()
        def raise_for_status(self): pass
    def fake_get(url, timeout=120):
        calls.append(url)
        return _R()
    monkeypatch.setattr("requests.get", fake_get)

    names = [c["name"] for c in dd.discover_ept_candidates(28.0, -81.0)]
    assert calls, "stale cache must trigger a refresh"
    assert names == ["FL_Fresh_2024"], "refreshed index must be the one used"


def test_failed_refresh_falls_back_to_stale_cache(cache, monkeypatch):
    """A stale index still finds LiDAR; a hard failure would lose pitch entirely."""
    _write(cache, _index("FL_Stale_2015"))
    old = time.time() - 40 * 86400
    import os as _os
    _os.utime(cache, (old, old))
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr("requests.get", boom)
    assert [c["name"] for c in dd.discover_ept_candidates(28.0, -81.0)] == ["FL_Stale_2015"]


def test_no_cache_and_failed_download_returns_empty(cache, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr("requests.get", boom)
    assert dd.discover_ept_candidates(28.0, -81.0) == []


def test_env_var_forces_refresh(cache, monkeypatch):
    _write(cache, _index("FL_Cached_2020"))              # fresh on disk
    monkeypatch.setenv("MEASURE_IT_ENTWINE_REFRESH", "1")
    class _R:
        content = json.dumps(_index("FL_Forced_2025")).encode()
        def raise_for_status(self): pass
    monkeypatch.setattr("requests.get", lambda url, timeout=120: _R())
    assert [c["name"] for c in dd.discover_ept_candidates(28.0, -81.0)] == ["FL_Forced_2025"]


# --------------------------------------------------------------------------
# 5. fallback across covering datasets — the "always have the LiDAR" goal
# --------------------------------------------------------------------------
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
    def __init__(self, payload): self._p = payload
    def json(self):
        return json.loads(self._p) if isinstance(self._p, (bytes, str)) else self._p
    @property
    def content(self): return self._p


@pytest.fixture
def two_datasets(monkeypatch):
    """'bad' covers the point but holds no points over the building; 'good' does."""
    xs, ys = np.meshgrid(np.arange(499500.5, 499540.0, 0.5),
                         np.arange(3099500.5, 3099530.0, 0.5))
    x, y = xs.ravel(), ys.ravel()
    z = 0.5 * (x - 499500.0) + 10.0
    good = {
        "ept.json": _Resp({"bounds": ROOT, "dataType": "laszip",
                           "srs": {"authority": "EPSG", "horizontal": "32617"}}),
        "ept-hierarchy/0-0-0-0.json": _Resp({"0-0-0-0": len(x)}),
        "ept-data/0-0-0-0.laz": _Resp(_laz_bytes(x, y, z)),
    }
    # same CRS, but its octree sits far away -> no nodes intersect the footprint
    bad = {
        "ept.json": _Resp({"bounds": [400000.0, 3000000.0, 0.0,
                                      401024.0, 3001024.0, 1024.0],
                           "dataType": "laszip",
                           "srs": {"authority": "EPSG", "horizontal": "32617"}}),
        "ept-hierarchy/0-0-0-0.json": _Resp({"0-0-0-0": 10}),
    }
    seen = []

    def fake_get(url, timeout=60):
        seen.append(url)
        table = bad if "/bad/" in url else good
        for suffix, resp in table.items():
            if url.endswith(suffix):
                return resp
        raise AssertionError(f"unexpected URL {url}")
    monkeypatch.setattr(ef, "_get", fake_get)
    return seen


def test_falls_through_to_the_next_covering_dataset(two_datasets, monkeypatch):
    """The whole point: a covering dataset with nothing over the roof must not
    cost us the pitch when another survey has the building."""
    monkeypatch.setattr(dd, "discover_ept_candidates", lambda lat, lon, **k: [
        {"name": "BAD_2024", "url": "https://fake/bad/ept.json", "year": 2024, "density": 0.0},
        {"name": "GOOD_2023", "url": "https://fake/good/ept.json", "year": 2023, "density": 0.0},
    ])
    pts = ef.fetch_roof_points(28.0, -81.0, _fp_wgs84(), CRS_UTM)
    assert pts is not None and len(pts) > 0
    assert any("/bad/" in u for u in two_datasets), "the first candidate must be tried"
    assert any("/good/" in u for u in two_datasets), "and the second must rescue it"


def test_with_ground_contract_survives_the_fallback(two_datasets, monkeypatch):
    monkeypatch.setattr(dd, "discover_ept_candidates", lambda lat, lon, **k: [
        {"name": "BAD_2024", "url": "https://fake/bad/ept.json", "year": 2024, "density": 0.0},
        {"name": "GOOD_2023", "url": "https://fake/good/ept.json", "year": 2023, "density": 0.0},
    ])
    pts, ground_z = ef.fetch_roof_points(28.0, -81.0, _fp_wgs84(), CRS_UTM,
                                         with_ground=True)
    assert pts is not None and pts.shape[1] == 3
    assert ground_z is None or isinstance(ground_z, float)


def test_all_candidates_failing_returns_the_no_coverage_contract(two_datasets, monkeypatch):
    monkeypatch.setattr(dd, "discover_ept_candidates", lambda lat, lon, **k: [
        {"name": "BAD_2024", "url": "https://fake/bad/ept.json", "year": 2024, "density": 0.0},
    ])
    assert ef.fetch_roof_points(28.0, -81.0, _fp_wgs84(), CRS_UTM) is None
    assert ef.fetch_roof_points(28.0, -81.0, _fp_wgs84(), CRS_UTM,
                                with_ground=True) == (None, None)


def test_explicit_url_skips_discovery_entirely(two_datasets, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("explicit ept_url must not trigger discovery")
    monkeypatch.setattr(dd, "discover_ept_candidates", boom)
    pts = ef.fetch_roof_points(28.0, -81.0, _fp_wgs84(), CRS_UTM,
                               ept_url="https://fake/good/ept.json")
    assert pts is not None and len(pts) > 0


def test_max_candidates_caps_the_attempts(two_datasets, monkeypatch):
    monkeypatch.setattr(dd, "discover_ept_candidates", lambda lat, lon, **k: [
        {"name": "BAD_2024", "url": "https://fake/bad/ept.json", "year": 2024, "density": 0.0},
        {"name": "GOOD_2023", "url": "https://fake/good/ept.json", "year": 2023, "density": 0.0},
    ])
    assert ef.fetch_roof_points(28.0, -81.0, _fp_wgs84(), CRS_UTM,
                                max_candidates=1) is None      # never reached 'good'
    assert not any("/good/" in u for u in two_datasets)


def test_a_county_whose_name_ends_in_las_keeps_its_collection_year():
    """The release-marker test matched mid-word: "PINELLAS" ends with the
    letters "LAS", so FL_Peninsular_Pinellas_2018 scored 0 and ranked BELOW
    FL_PinellasCo_2007. Every Pinellas report therefore measured off
    eleven-year-old LiDAR at 2.8 pts/m2 — the density confound that broke the
    GIS-vs-NAIP comparison, in the one county with 3-inch public imagery."""
    from src.lidar.dataset_discovery import _collection_year

    assert _collection_year("FL_Peninsular_Pinellas_2018") == 2018
    assert _collection_year("FL_PinellasCo_2007") == 2007
    # the original bug this marker logic exists for must still be caught
    assert _collection_year("USGS_LPC_FL_Upper_Saint_Johns_2017_LAS_2019") == 2017
    assert _collection_year("FL_Peninsular_FDEM_Brevard_2018") == 2018
    assert _collection_year("FL_Elgin_2006_2008") == 2008


def test_release_marker_must_be_its_own_token():
    from src.lidar.dataset_discovery import _collection_year

    assert _collection_year("SOMEWHERE_LAS_2019") == 0        # real release marker
    assert _collection_year("SOMEWHERE_ATLAS_2019") == 2019   # word merely ending in las
    assert _collection_year("COUNTY_REL_2020") == 0
    assert _collection_year("LAUREL_2020") == 2020            # ends in "REL"
