"""Offline tests for the GIS chip fetcher's georeferencing (network mocked)."""
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image
from shapely.geometry import box

import src.ingestion.gis_chip as gc


def test_utm_epsg_florida():
    assert gc._utm_epsg(28.03, -80.70) == 32617      # zone 17N (Brevard)
    assert gc._utm_epsg(27.95, -82.46) == 32617      # zone 17N (Tampa)


def test_fetch_chip_gis_contract(monkeypatch, tmp_path):
    # a ~20x11 m footprint near Brevard, in lon/lat degrees
    fp = box(-80.6982, 28.0304, -80.6980, 28.0305)
    monkeypatch.setattr(
        "src.roofs.select_candidates.select_building",
        lambda lat, lon, buffer_meters=60: {"selected": SimpleNamespace(geometry=fp)})

    def fake_export(endpoint, bounds, sr, w, h, timeout=60):
        buf = io.BytesIO()
        Image.new("RGB", (w, h), (120, 120, 120)).save(buf, "PNG")
        return buf.getvalue()
    monkeypatch.setattr(gc, "_export_image", fake_export)

    chip, transform, png, anchor, meta = gc.fetch_chip_gis(
        28.0304, -80.6981, "FL", tmp_path)

    # contract: same shape family as the NAIP fetch_chip 5-tuple
    assert chip.ndim == 3 and chip.shape[2] == 3
    assert chip.shape[:2] == anchor.shape            # chip and anchor share the grid
    assert anchor.any()                              # footprint rasterized inside
    assert meta["crs"] == "EPSG:32617"
    assert meta["footprint_wgs84"].equals(fp)
    # METRIC transform (not degrees, not 3857-inflated): pixel size ~ gsd, north-up
    assert abs(transform.a - meta["gsd_m"]) < 0.05
    assert 0 < transform.a < 1.0 and transform.e < 0


# --- ArcGIS content types ---------------------------------------------------

def test_jpeg_from_an_imageserver_is_accepted(monkeypatch):
    """The Florida statewide FCDOP set answers exportImage with JPEG whenever
    the tile needs no transparency. _export_image demanded PNG magic bytes and
    raised "non-PNG response", so the ONLY GIS source covering most of Florida
    was discarded on its content type and the report silently degraded to 30 cm
    NAIP -- measured at 1250 Pineapple Ave, Melbourne."""
    import io

    from src.ingestion import gis_chip

    jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 64
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: io.BytesIO(jpeg))
    got = gis_chip._export_image("https://example/ImageServer",
                                 (0, 0, 10, 10), 26917, 8, 8)
    assert got == jpeg


def test_png_and_tiff_still_accepted(monkeypatch):
    import io

    from src.ingestion import gis_chip

    for blob in (b"\x89PNG\r\n\x1a\n" + b"\x00" * 32,
                 b"II*\x00" + b"\x00" * 32,
                 b"MM\x00*" + b"\x00" * 32):
        monkeypatch.setattr("urllib.request.urlopen",
                            lambda *a, _b=blob, **k: io.BytesIO(_b))
        assert gis_chip._export_image("https://e/ImageServer",
                                      (0, 0, 1, 1), 26917, 4, 4) == blob


def test_a_json_error_body_surfaces_the_servers_own_reason(monkeypatch):
    """ArcGIS reports failures as JSON. That IS worth refusing — but the message
    should carry what the server said, not just "not a PNG"."""
    import io

    import pytest

    from src.ingestion import gis_chip

    body = b'{"error":{"code":400,"message":"Invalid bounding box"}}'
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: io.BytesIO(body))
    with pytest.raises(RuntimeError, match="Invalid bounding box"):
        gis_chip._export_image("https://e/ImageServer", (0, 0, 1, 1), 26917, 4, 4)


def test_building_selection_is_not_repeated_per_imagery_attempt(monkeypatch):
    """A Melbourne report ran the full MS Buildings path TWICE in 71 seconds --
    index load, shard download, dedup, ranking -- because the GIS attempt
    selects the building, fails, and the NAIP fallback selects it again. Two
    identical "Closest building distance: 26.86m" blocks in one report. Inputs
    are identical every time, so the second call must not hit the network."""
    from src.roofs import select_candidates as sc

    sc._SELECT_CACHE.clear()
    calls = []

    def _fake_buildings(*a, **k):
        calls.append(1)
        raise RuntimeError("network reached")

    monkeypatch.setattr(sc, "get_buildings_in_buffer", _fake_buildings)
    for _ in range(2):
        try:
            sc.select_building(28.133, -80.627)
        except RuntimeError:
            pass
    assert len(calls) == 2, "failures must NOT be cached — only real results"

    # a successful result is cached, so the fallback chain reuses it
    sc._SELECT_CACHE.clear()
    sentinel = {"selected": "S", "candidates": "C", "dist_m": 26.86, "rank": 0}
    key = (round(28.133, 7), round(-80.627, 7), float(sc.DEFAULT_BUFFER_METERS), True)
    sc._cache_put(key, sentinel)
    assert sc.select_building(28.133, -80.627) is sentinel


def test_selection_cache_evicts_and_keys_on_location(monkeypatch):
    from src.roofs import select_candidates as sc

    sc._SELECT_CACHE.clear()
    for i in range(sc._SELECT_CACHE_MAX + 3):
        sc._cache_put((i, i, 100.0, True), {"n": i})
    assert len(sc._SELECT_CACHE) == sc._SELECT_CACHE_MAX
    assert (0, 0, 100.0, True) not in sc._SELECT_CACHE     # oldest evicted
