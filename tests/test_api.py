"""API tests — the full web layer with injected fakes (no GPU, no network)."""
import io

import numpy as np
import pytest
from affine import Affine
from fastapi.testclient import TestClient

import src.serve.api as api_mod
from src.serve.api import create_app


def _rect(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


H = W = 100
ROOF = _rect(H, W, 20, 80, 20, 80)
QUADS = [_rect(H, W, 20, 50, 20, 50), _rect(H, W, 20, 50, 50, 80),
         _rect(H, W, 50, 80, 20, 50), _rect(H, W, 50, 80, 50, 80)]


def fake_facets(chip, concept):
    h, w = chip.shape[:2]
    if (h, w) != (H, W):                      # uploaded images: scale masks
        def sc(m):
            ys, xs = np.where(m)
            out = np.zeros((h, w), bool)
            out[(ys * h // H), (xs * w // W)] = True
            return out
        return np.asarray([sc(m) for m in [ROOF] + QUADS]), \
            np.asarray([0.92, 0.7, 0.68, 0.66, 0.64])
    return np.asarray([ROOF] + QUADS), np.asarray([0.92, 0.7, 0.68, 0.66, 0.64])


def fake_outline(chip, concept):
    h, w = chip.shape[:2]
    m = np.zeros((h, w), bool)
    m[h // 5:4 * h // 5, w // 5:4 * w // 5] = True
    return np.asarray([m]), np.asarray([0.95])


def fake_chip(lat, lon, state, out_dir, chip_buffer_m=18.0):
    from PIL import Image
    chip = np.full((H, W, 3), 120, np.uint8)
    png = out_dir / "chip.png"
    Image.fromarray(chip).save(png)
    return chip, Affine(0.6, 0, 500000.0, 0, -0.6, 3100000.0), str(png)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_mod, "JOBS_ROOT", tmp_path)
    app = create_app(predict_facets=fake_facets, predict_outline=fake_outline,
                     chip_fetcher=fake_chip)
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["model_loaded"]


def test_index_serves_frontend(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Roof Measurement Report" in r.text


def test_report_by_coords(client):
    r = client.post("/api/report",
                    json={"location": "28.0303, -80.69809", "state": "FL"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["num_facets"] == 4
    assert d["plan_area_sqft"] > 0 and d["units_measured"]
    assert "eave" in d["edge_totals_ft"]
    # artifacts are fetchable
    pdf = client.get(d["pdf_url"])
    assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-"
    prev = client.get(d["preview_url"])
    assert prev.status_code == 200


def test_report_empty_location_422(client):
    r = client.post("/api/report", json={"location": "   ", "state": "FL"})
    assert r.status_code == 422
    assert "Google Maps" in r.json()["detail"]      # friendly guidance


def _png_bytes(h=120, w=120):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), 130, np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def test_image_upload_without_scale_no_pdf(client):
    r = client.post("/api/report/image",
                    files={"file": ("roof.png", _png_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["num_facets"] >= 1
    assert d["pdf_url"] is None and not d["units_measured"]   # honest units
    assert client.get(d["preview_url"]).status_code == 200


def test_image_upload_with_scale_full_pdf(client):
    r = client.post("/api/report/image",
                    files={"file": ("roof.png", _png_bytes(), "image/png")},
                    data={"scale_m_per_px": "0.6"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["pdf_url"] and d["units_measured"]
    assert client.get(d["pdf_url"]).content[:5] == b"%PDF-"


def test_image_upload_garbage_422(client):
    r = client.post("/api/report/image",
                    files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 422


def test_unknown_report_404(client):
    assert client.get("/api/report/deadbeef0000/pdf").status_code == 404


def test_no_models_503(tmp_path, monkeypatch):
    monkeypatch.setattr(api_mod, "JOBS_ROOT", tmp_path)
    monkeypatch.delenv("MEASURE_IT_CKPT", raising=False)
    app = create_app()                                # nothing injected, no env
    with TestClient(app) as c:
        r = c.post("/api/report",
                   json={"location": "28.0, -80.7", "state": "FL"})
        assert r.status_code == 503
