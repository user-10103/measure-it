"""County GIS aerial (≈7.6 cm / 15 cm) as a georeferenced report chip.

Drop-in for ``report_service.fetch_chip``: same
``(chip, Affine, png_path, anchor, meta)`` contract, but sourced from Florida
county / FCDOP ArcGIS ImageServers instead of 30 cm NAIP. The SAM3 facets were
fine-tuned on ~7 cm GIS chips, so inferring on this imagery (not 30 cm NAIP)
closes the train/serve resolution gap. Inject it with::

    generate_roof_report(..., chip_fetcher=fetch_chip_gis)

CRS: the chip is fetched and georeferenced in a **metric UTM** zone, NOT Web
Mercator (EPSG:3857). At FL latitude 3857 inflates ground distance ~1/cos(lat)
≈ 1.13× — a ~28% area error — which would corrupt every sqft in the report.
UTM is ~1:1, so plan areas stay honest.

The live ImageServer fetch needs network (validate in Colab); the georeferencing
math (UTM pick, Affine, anchor raster) is unit-tested offline.
"""
from __future__ import annotations

import io
import logging
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_UA = {"User-Agent": "Mozilla/5.0 (measure-it gis-chip fetch)"}


def _utm_epsg(lat: float, lon: float) -> int:
    """EPSG code of the UTM zone containing (lat, lon). Northern hemisphere only
    here (FL) -> 326xx; kept general for the southern case."""
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _export_image(endpoint: str, bounds, sr: int, w: int, h: int,
                  timeout: int = 60) -> bytes:
    """ArcGIS ImageServer exportImage -> PNG bytes, with bbox and image in one
    metric SR (so pixel<->world is a plain affine). Raises on a JSON error body."""
    west, south, east, north = bounds
    q = urllib.parse.urlencode({
        "bbox": f"{west:.3f},{south:.3f},{east:.3f},{north:.3f}",
        "bboxSR": sr, "imageSR": sr, "size": f"{w},{h}",
        "format": "png", "f": "image"})
    req = urllib.request.Request(f"{endpoint}/exportImage?{q}", headers=_UA)
    png = urllib.request.urlopen(req, timeout=timeout).read()
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"non-PNG response from {endpoint}: {png[:200]!r}")
    return png


def fetch_chip_gis(lat: float, lon: float, state: str, out_dir,
                   chip_buffer_m: Optional[float] = None,
                   endpoint: Optional[dict] = None):
    """GIS aerial chip for a location, matching ``fetch_chip``'s 5-tuple.

    Returns ``(chip HxWx3 uint8, rasterio Affine (UTM), png_path, anchor bool
    mask, meta)`` where ``meta`` carries ``crs`` and ``footprint_wgs84`` — the
    two fields the LiDAR fetch gate keys on.

    endpoint: an ImageServer descriptor ``{"url", "gsd_m", "year"}``; defaults to
    the statewide FCDOP 15 cm set (reachable, covers every FL county). County
    3-inch endpoints (e.g. Pinellas) can be passed explicitly.
    """
    import geopandas as gpd
    from PIL import Image
    from rasterio import features as rio_features
    from rasterio.transform import from_bounds

    from src.ingestion.county_imagery import FCDOP_FALLBACK
    from src.roofs.select_candidates import select_building
    from src.serve.report_service import CHIP_BUFFER_M, FOOTPRINT_BUFFER_M

    if chip_buffer_m is None:
        chip_buffer_m = CHIP_BUFFER_M
    ep = endpoint or FCDOP_FALLBACK
    gsd = float(ep["gsd_m"])

    # 1. target building footprint (same selection the NAIP path uses)
    sel = select_building(lat, lon, buffer_meters=FOOTPRINT_BUFFER_M)
    fp4326 = gpd.GeoDataFrame(geometry=[sel["selected"].geometry], crs="EPSG:4326")

    # 2. metric UTM frame; loose crop = footprint + chip_buffer_m
    utm = _utm_epsg(lat, lon)
    fp_utm = fp4326.to_crs(epsg=utm).geometry.iloc[0]
    bounds = fp_utm.buffer(chip_buffer_m).bounds          # (w, s, e, n) metres
    west, south, east, north = bounds
    w = max(8, round((east - west) / gsd))
    h = max(8, round((north - south) / gsd))

    # 3. fetch + decode; trust the requested grid (resize if the server rounded)
    png = _export_image(ep["url"], bounds, utm, w, h)
    img = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    if img.shape[:2] != (h, w):
        img = np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))

    # 4. georeference + rasterize the footprint anchor in the SAME grid
    transform = from_bounds(west, south, east, north, w, h)
    anchor = rio_features.rasterize(
        [(fp_utm, 1)], out_shape=(h, w), transform=transform, dtype="uint8"
    ).astype(bool)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "gis_chip.png"
    Image.fromarray(img).save(png_path)

    meta = {"crs": f"EPSG:{utm}", "footprint_wgs84": fp4326.geometry.iloc[0],
            "gsd_m": gsd, "source": ep["url"], "year": ep.get("year")}
    logger.info("GIS chip: %dx%d @ %.3f m/px (%s) UTM %d", w, h, gsd,
                ep["url"], utm)
    return img, transform, str(png_path), anchor, meta
