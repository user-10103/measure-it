"""End-to-end service test: injected predictors + chip -> 6-page PDF + facts."""
import numpy as np
import pytest
from affine import Affine

from src.serve.report_service import generate_roof_report


def _rect(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


H = W = 100
ROOF = _rect(H, W, 20, 80, 20, 80)
QUADS = [_rect(H, W, 20, 50, 20, 50), _rect(H, W, 20, 50, 50, 80),
         _rect(H, W, 50, 80, 20, 50), _rect(H, W, 50, 80, 50, 80)]


def fake_facets(chip, concept):
    # emits a whole-roof mask too — the service must not collapse to 1 facet
    return np.asarray([ROOF] + QUADS), np.asarray([0.92, 0.7, 0.68, 0.66, 0.64])


def fake_outline(chip, concept):
    return np.asarray([ROOF]), np.asarray([0.95])


def no_outline(chip, concept):
    return np.zeros((0, H, W), bool), np.zeros((0,))


def fake_chip(lat, lon, state, out_dir, chip_buffer_m=18.0):
    from PIL import Image
    chip = np.full((H, W, 3), 120, np.uint8)
    png = out_dir / "chip.png"
    Image.fromarray(chip).save(png)
    # 0.6 m/px NAIP-like metric transform
    return chip, Affine(0.6, 0, 500000.0, 0, -0.6, 3100000.0), str(png)


def test_end_to_end_pdf_and_facts(tmp_path):
    res = generate_roof_report(
        (28.0303, -80.69809), "FL", fake_facets, fake_outline,
        out_dir=tmp_path, label="909 Spring Island Way",
        chip_fetcher=fake_chip,
    )
    assert res.num_facets == 4                       # whole-roof mask dropped
    assert res.outline_found
    with open(res.pdf_path, "rb") as fh:
        assert fh.read(5) == b"%PDF-"
    # metric CRS: 60x60 px roof at 0.6 m/px ~ 1296 m2
    assert 1000 < res.plan_area_m2 < 1600
    assert "eave" in res.edge_totals_m               # typed edge graph present
    assert set(res.edge_totals_m) & {"ridge", "hip", "valley"}


def test_coords_string_input(tmp_path):
    res = generate_roof_report(
        "28.0303, -80.69809", "FL", fake_facets, fake_outline,
        out_dir=tmp_path, chip_fetcher=fake_chip,
    )
    assert res.location_source == "coordinates"
    assert abs(res.lat - 28.0303) < 1e-6


def test_outline_fallback_from_facet_union(tmp_path):
    res = generate_roof_report(
        (28.0, -80.7), "FL", fake_facets, no_outline,
        out_dir=tmp_path, chip_fetcher=fake_chip,
    )
    assert not res.outline_found                     # zero-shot missed...
    assert res.edge_totals_m.get("eave", 0) > 0      # ...but eaves still exist
    with open(res.pdf_path, "rb") as fh:
        assert fh.read(5) == b"%PDF-"


def test_lidar_fusion_adds_pitch(tmp_path):
    # LiDAR grid over the roof in the fake chip's WORLD CRS (0.6 m/px,
    # origin 500000/3100000, y down); z = 0.5*(x-x0) -> every facet "6:12"
    xs, ys = np.meshgrid(np.arange(500010.0, 500050.0, 0.5),
                         np.arange(3099950.0, 3099990.0, 0.5))
    x, y = xs.ravel(), ys.ravel()
    pts = np.column_stack([x, y, 0.5 * (x - 500000.0)])
    res = generate_roof_report(
        (28.0303, -80.69809), "FL", fake_facets, fake_outline,
        out_dir=tmp_path, chip_fetcher=fake_chip, lidar_points=pts,
    )
    assert res.num_facets == 4
    assert res.num_pitched == 4                       # every facet got a pitch
    with open(res.pdf_path, "rb") as fh:
        assert fh.read(5) == b"%PDF-"


def test_without_lidar_num_pitched_zero(tmp_path):
    res = generate_roof_report(
        (28.0303, -80.69809), "FL", fake_facets, fake_outline,
        out_dir=tmp_path, chip_fetcher=fake_chip,
    )
    assert res.num_pitched == 0                       # imagery-only floor


def test_no_facets_still_produces_pdf(tmp_path):
    res = generate_roof_report(
        (28.0, -80.7), "FL", no_outline, no_outline,
        out_dir=tmp_path, chip_fetcher=fake_chip,
    )
    assert res.num_facets == 0
    with open(res.pdf_path, "rb") as fh:
        assert fh.read(5) == b"%PDF-"                # graceful, not a crash
