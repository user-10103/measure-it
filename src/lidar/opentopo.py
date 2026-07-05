"""OpenTopography integrations (requires OPENTOPO_API_KEY).

Two capabilities, both best-effort (None/[] on any failure — the pipeline
never depends on OT being up):

1. ``get_ground_elevation`` — clip a tiny USGS 3DEP DTM (bare-earth) raster
   around the building via the synchronous ``/API/usgsdem`` endpoint and read
   the ground elevation. Used as the FALLBACK ground source for two-story
   detection when the EPT tiles carry no class-2 ground points.
2. ``list_point_datasets`` — the OT catalog for a point: which LiDAR surveys
   cover it and from what year. Diagnostic: lets us log when OT/3DEP has a
   NEWER survey than the entwine mirror served us.

Note: OT's point-cloud *downloads* are job-based (not request/response), so
real-time roof points stay on the direct EPT path; OT complements it.
"""
from __future__ import annotations

import io
import logging
import os
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

USGSDEM_URL = "https://portal.opentopography.org/API/usgsdem"
CATALOG_URL = "https://portal.opentopography.org/API/otCatalog"


def _api_key() -> Optional[str]:
    return os.getenv("OPENTOPO_API_KEY")


def get_ground_elevation(lat: float, lon: float,
                         dataset: str = "USGS1m",
                         half_width_deg: float = 0.0004) -> Optional[float]:
    """Bare-earth ground elevation (m) at a point from the 3DEP DTM, or None.

    ~40 m half-width clip; returns the median of valid cells (robust to the
    odd nodata pixel). USGS1m needs an OT account with 3DEP access; USGS10m
    is the open fallback dataset.
    """
    key = _api_key()
    if not key:
        return None
    import requests
    try:
        r = requests.get(USGSDEM_URL, params={
            "datasetName": dataset,
            "south": lat - half_width_deg, "north": lat + half_width_deg,
            "west": lon - half_width_deg, "east": lon + half_width_deg,
            "outputFormat": "GTiff", "API_Key": key,
        }, timeout=90)
        if r.status_code != 200 or not r.content[:200].startswith(b"II") \
                and not r.content[:200].startswith(b"MM"):
            if dataset == "USGS1m":          # 1m is gated; retry on open 10m
                logger.info("OT %s unavailable (%s) — trying USGS10m",
                            dataset, r.status_code)
                return get_ground_elevation(lat, lon, dataset="USGS10m",
                                            half_width_deg=half_width_deg)
            logger.warning("OT usgsdem failed (%s)", r.status_code)
            return None
        import numpy as np
        import rasterio
        with rasterio.open(io.BytesIO(r.content)) as src:
            band = src.read(1, masked=True)
        if band.count() == 0:
            return None
        val = float(np.ma.median(band))
        logger.info("OT ground elevation (%s): %.1f m", dataset, val)
        return val
    except Exception as e:  # noqa: BLE001 — fallback source, never fatal
        logger.warning("OT ground elevation failed (%s)", e)
        return None


def list_point_datasets(lat: float, lon: float,
                        pad_deg: float = 0.001) -> List[dict]:
    """LiDAR point datasets covering a location per the OT catalog.

    Returns [{"name", "year"}] sorted newest-first (year parsed from the
    name/dates when present). Diagnostic for survey recency.
    """
    key = _api_key()
    import requests
    try:
        r = requests.get(CATALOG_URL, params={
            "productFormat": "PointCloud",
            "minx": lon - pad_deg, "miny": lat - pad_deg,
            "maxx": lon + pad_deg, "maxy": lat + pad_deg,
            "detail": "false", "outputFormat": "json",
            "include_federated": "true",
            **({"API_Key": key} if key else {}),
        }, timeout=60)
        r.raise_for_status()
        out = []
        for ds in r.json().get("Datasets", []):
            meta = ds.get("Dataset", ds)
            name = meta.get("name") or meta.get("identifier", {}).get("value", "")
            years = [int(m.group(0)) for m in
                     re.finditer(r"(?:19|20)\d{2}",
                                 str(name) + str(meta.get("temporalCoverage", "")))] or [0]
            out.append({"name": name, "year": max(years)})
        out.sort(key=lambda d: -d["year"])
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("OT catalog failed (%s)", e)
        return []
