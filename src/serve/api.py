"""FastAPI wrapper around the report service — the deployable unit.

    uvicorn src.serve.api:app --host 0.0.0.0 --port 8000

Routes:
    GET  /                    the frontend (address box -> report card)
    GET  /healthz             liveness + model status
    POST /api/report          {"location": "...", "state": "FL"} -> facts + links
    POST /api/report/image    multipart upload (optional scale_m_per_px)
    GET  /api/report/{id}/pdf | /preview   the artifacts

Model loading: predictors are INJECTED. In production the app loads the real
SAM 3 pair at startup from $MEASURE_IT_CKPT (see sam3_predictors); in tests
``create_app(predict_facets=fake, ...)`` runs the whole API without a GPU.
A process-wide lock serializes report jobs — one GPU, one job at a time.
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from src.serve.report_service import (
    fetch_chip,
    generate_report_from_image,
    generate_roof_report,
)
from src.utils.resolve_location import LocationError, suggest_addresses

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
JOBS_ROOT = Path(os.getenv("MEASURE_IT_OUTPUT", "output/api_jobs"))


class ReportRequest(BaseModel):
    location: str = Field(..., description="address, 'lat, lon', or maps link")
    state: str = Field(..., min_length=2, max_length=2,
                       description="2-letter state for the NAIP archive")


def create_app(predict_facets=None, predict_outline=None,
               chip_fetcher=fetch_chip, suggester=suggest_addresses) -> FastAPI:
    state = {"predict_facets": predict_facets,
             "predict_outline": predict_outline}
    gpu_lock = threading.Lock()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(app):
        if state["predict_facets"] is None:          # not injected (production)
            ckpt = os.getenv("MEASURE_IT_CKPT")
            if not ckpt:
                logger.warning("MEASURE_IT_CKPT unset — /api/report will 503 "
                               "until models are provided")
            else:
                from src.roofs.sam3_predictors import load_sam3_predictors
                state["predict_facets"], state["predict_outline"] = \
                    load_sam3_predictors(
                        ckpt,
                        use_zeroshot_outline=os.getenv(
                            "MEASURE_IT_ZEROSHOT_OUTLINE", "1") == "1")
                logger.info("SAM 3 predictors ready")
        yield

    app = FastAPI(title="measure-it roof reports", version="0.1",
                  lifespan=_lifespan)

    def _require_models():
        if state["predict_facets"] is None:
            raise HTTPException(503, "model not loaded yet — try again shortly")

    def _job_dir() -> tuple[str, Path]:
        job_id = uuid.uuid4().hex[:12]
        d = JOBS_ROOT / job_id
        d.mkdir(parents=True, exist_ok=True)
        return job_id, d

    def _result_json(job_id: str, res) -> dict:
        SQFT = 10.7639
        return {
            "report_id": job_id,
            "num_facets": res.num_facets,
            "outline_found": res.outline_found,
            "plan_area_sqft": round(res.plan_area_m2 * SQFT, 1),
            "squares": round(res.plan_area_m2 * SQFT / 100.0, 1),
            "num_pitched": res.num_pitched,
            "location_source": res.location_source,
            "lat": res.lat, "lon": res.lon,
            "edge_totals_ft": {k: round(v * 3.28084, 1)
                               for k, v in res.edge_totals_m.items()},
            "units_measured": res.location_source != "image" or res.pdf_path is not None,
            "pdf_url": f"/api/report/{job_id}/pdf" if res.pdf_path else None,
            "preview_url": f"/api/report/{job_id}/preview",
        }

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "model_loaded": state["predict_facets"] is not None}

    @app.get("/api/suggest")
    def suggest(q: str = ""):
        """Address autocomplete — empty list when no geocoder is configured."""
        return {"suggestions": suggester(q)}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC_DIR / "index.html").read_text()

    @app.post("/api/report")
    def report(req: ReportRequest):
        _require_models()
        job_id, d = _job_dir()
        try:
            with gpu_lock:
                res = generate_roof_report(
                    req.location, req.state,
                    state["predict_facets"], state["predict_outline"],
                    out_dir=d, chip_fetcher=chip_fetcher,
                )
        except LocationError as e:
            raise HTTPException(422, str(e))
        except RuntimeError as e:                     # e.g. no NAIP coverage
            raise HTTPException(422, str(e))
        except Exception:
            logger.exception("report job %s failed", job_id)
            raise HTTPException(500, "report generation failed — "
                                     "the team has been notified")
        return _result_json(job_id, res)

    @app.post("/api/report/image")
    def report_image(file: UploadFile = File(...),
                     label: str = Form("uploaded roof"),
                     scale_m_per_px: Optional[float] = Form(None)):
        _require_models()
        from PIL import Image
        job_id, d = _job_dir()
        try:
            chip = np.array(Image.open(file.file).convert("RGB"))
        except Exception:
            raise HTTPException(422, "could not read that file as an image")
        if max(chip.shape[:2]) > 4096:
            raise HTTPException(422, "image too large (max 4096 px)")
        try:
            with gpu_lock:
                res = generate_report_from_image(
                    chip, state["predict_facets"], state["predict_outline"],
                    out_dir=d, label=label, scale_m_per_px=scale_m_per_px,
                )
        except Exception:
            logger.exception("image job %s failed", job_id)
            raise HTTPException(500, "report generation failed")
        out = _result_json(job_id, res)
        # remember artifact paths for the GET routes
        (d / ".preview").write_text(res.chip_path)
        return out

    def _artifact(job_id: str, name: str) -> Path:
        d = JOBS_ROOT / job_id
        if not d.is_dir() or not job_id.isalnum():
            raise HTTPException(404, "unknown report id")
        return d / name

    @app.get("/api/report/{job_id}/pdf")
    def report_pdf(job_id: str):
        p = _artifact(job_id, "roof_report.pdf")
        if not p.exists():
            raise HTTPException(404, "no PDF for this report")
        return FileResponse(p, media_type="application/pdf",
                            filename="roof_report.pdf")

    @app.get("/api/report/{job_id}/preview")
    def report_preview(job_id: str):
        marker = _artifact(job_id, ".preview")
        p = (Path(marker.read_text()) if marker.exists()
             else _artifact(job_id, "chip.png"))
        if not p.exists():
            raise HTTPException(404, "no preview for this report")
        return FileResponse(p, media_type="image/png")

    return app


app = create_app()
