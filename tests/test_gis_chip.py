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
