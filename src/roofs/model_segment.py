"""RF-DETR model facet segmentation for the LiDAR/measurement pipeline.

Runs the trained RF-DETR model on the clipped NAIP chip and converts its
pixel-space facet polygons into the LiDAR metric CRS, returning ``Facet``
objects that plug straight into the existing plane-fit + edge pipeline (same
contract as ``naip_guided_segmentation``). Downstream, LiDAR points are assigned
to each model facet and a real plane is fit — so the MODEL supplies the 2D facet
shape and LiDAR supplies the 3D pitch/aspect (fusion).

The heavy part (``segment_facets_model``) needs torch + rfdetr + a checkpoint and
runs on Colab/the pod. The pure-geometry conversion (``model_facets_to_metric``)
has no such deps and is unit-tested locally with a mock prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from shapely.geometry import Polygon

from src.roofs.segment import Facet

logger = logging.getLogger(__name__)


def model_facets_to_metric(pred: dict, transform, src_crs, dst_crs,
                           min_area_m2: float = 1.0) -> List[Facet]:
    """Convert an ``RFDETRBackend.predict()`` result into metric ``Facet``s.

    Parameters
    ----------
    pred : dict
        ``{"outline": [[x,y],...] | None, "facets": [{"polygon": [[x,y],...]}, ...]}``
        with polygon vertices in the chip's pixel space (col, row).
    transform : affine.Affine
        The NAIP raster's transform mapping pixel (col, row) -> world (x, y) in
        ``src_crs`` (i.e. ``rasterio`` dataset ``.transform``).
    src_crs, dst_crs : pyproj-compatible CRS
        Raster CRS and the LiDAR ``points_crs`` to emit polygons in.
    min_area_m2 : float
        Drop facets smaller than this after conversion.

    Returns
    -------
    list[Facet]
        Facets with ``.polygon`` set (metric, ``dst_crs``); ``.points`` is None
        until the pipeline attaches LiDAR points and fits a plane.
    """
    reproj = None
    if str(src_crs) != str(dst_crs):
        from pyproj import Transformer
        reproj = Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform

    def _ring_to_metric(ring):
        out = []
        for col, row in ring:
            wx, wy = transform * (col, row)          # pixel -> world (src_crs)
            if reproj is not None:
                wx, wy = reproj(wx, wy)              # -> dst_crs
            out.append((wx, wy))
        return out

    facets: List[Facet] = []
    fid = 1
    for f in pred.get("facets", []) or []:
        ring = f.get("polygon")
        if not ring or len(ring) < 3:
            continue
        poly = Polygon(_ring_to_metric(ring))
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.geom_type != "Polygon" or poly.is_empty or poly.area < min_area_m2:
            continue
        facets.append(Facet(facet_id=fid, polygon=poly, label=fid))
        fid += 1
    return facets


def preprocess_like_training(bands_chw: np.ndarray,
                             dtype_max: float = 255.0) -> np.ndarray:
    """Exact replica of the training chip pipeline (adresses/pipeline
    preprocess.process_chip): normalize to [0,1] -> per-band bilateral denoise
    -> unsharp mask (RGB only) -> per-band percentile stretch (2/98) -> NIR
    shadow lift. Matching the model's training distribution is essential — a
    mismatched preprocessing shifts it the *other* way and suppresses detections.

    Parameters
    ----------
    bands_chw : (C, H, W) raw raster array (>=3 bands; band 4 = NIR used for the
        shadow lift when present).
    dtype_max : max value of the source dtype (255 for NAIP uint8).

    Returns
    -------
    (H, W, 3) uint8 RGB, ready for RFDETRBackend.predict().
    """
    try:
        import cv2
    except Exception:                                  # pragma: no cover
        return np.transpose(np.clip(bands_chw[:3], 0, 255).astype(np.uint8), (1, 2, 0))

    b = bands_chw.astype(np.float32) / float(dtype_max)     # -> [0,1]

    def _u8(x):
        return np.clip(x * 255.0, 0, 255).astype(np.uint8)

    def _f(u):
        return u.astype(np.float32) / 255.0

    # 1. bilateral denoise, per band
    for i in range(b.shape[0]):
        b[i] = _f(cv2.bilateralFilter(_u8(b[i]), 9, 75, 75))
    # 2. unsharp mask, RGB only  (u*(1+amount) - blur*amount, amount=1.5, k=5)
    for i in range(min(3, b.shape[0])):
        u = _u8(b[i])
        blur = cv2.GaussianBlur(u, (5, 5), 0)
        b[i] = _f(cv2.addWeighted(u, 1.0 + 1.5, blur, -1.5, 0))
    # 3. per-band percentile contrast stretch (2 / 98)
    for i in range(b.shape[0]):
        lo, hi = np.percentile(b[i], (2, 98))
        if hi > lo:
            b[i] = np.clip((b[i] - lo) / (hi - lo), 0, 1)
    # 4. NIR-guided shadow lift (gamma 0.5 where NIR < 0.3), RGB channels
    if b.shape[0] >= 4:
        mask = b[3] < 0.3
        for i in range(3):
            ch = b[i]
            ch[mask] = np.power(ch[mask], 0.5)
            b[i] = np.clip(ch, 0, 1)

    return np.stack([_u8(b[i]) for i in range(3)], axis=-1)


def assign_lidar_points(facets: List[Facet], lidar_points,
                        min_points: int = 3) -> List[Facet]:
    """Attach roof LiDAR points to each model facet (point-in-polygon), so the
    downstream plane fit gives real pitch/aspect. Facets with fewer than
    ``min_points`` inside are dropped (no LiDAR support -> no reliable plane).
    This is the fusion step: model shape + LiDAR 3D."""
    if lidar_points is None or len(lidar_points) == 0:
        return facets
    import shapely
    xs = np.asarray(lidar_points['x'], dtype="float64")
    ys = np.asarray(lidar_points['y'], dtype="float64")
    kept: List[Facet] = []
    for fac in facets:
        try:
            mask = shapely.contains_xy(fac.polygon, xs, ys)
        except Exception:
            continue
        if int(mask.sum()) >= min_points:
            fac.points = lidar_points[mask]
            kept.append(fac)
    # renumber so facet_ids stay contiguous after drops
    for i, fac in enumerate(kept, start=1):
        fac.facet_id = i
        fac.label = i
    return kept


def segment_facets_model(naip_clipped_path, points_crs: str, checkpoint: str,
                         lidar_points=None, threshold: float = 0.40,
                         preprocess: bool = True) -> Optional[List[Facet]]:
    """Run RF-DETR on the clipped NAIP chip, convert to metric facets, and
    attach LiDAR points to each (for the downstream plane fit).

    Requires torch + rfdetr + the checkpoint (Colab/pod). Returns None on any
    failure so the pipeline falls back to NAIP-guided / k-means segmentation.
    """
    try:
        import rasterio
        from src.roofs.rfdetr_backend import RFDETRBackend
    except Exception as e:  # noqa: BLE001
        logger.warning("model segmentation unavailable (%s) — fallback", e)
        return None

    try:
        with rasterio.open(str(naip_clipped_path)) as src:
            arr = src.read()                          # (bands, H, W)
            transform = src.transform
            src_crs = src.crs
        if arr.shape[0] < 3:
            logger.warning("NAIP chip has <3 bands — cannot run model")
            return None
        if preprocess:                                # match training distribution
            dmax = (np.iinfo(arr.dtype).max
                    if np.issubdtype(arr.dtype, np.integer) else 1.0)
            rgb = preprocess_like_training(arr, dtype_max=float(dmax))
        else:
            rgb = np.transpose(arr[:3], (1, 2, 0))
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        backend = RFDETRBackend(checkpoint, threshold=threshold)
        pred = backend.predict(rgb)
        facets = model_facets_to_metric(pred, transform, src_crs, points_crs)
        facets = assign_lidar_points(facets, lidar_points)
        logger.info("Model segmentation: %d facet(s) with LiDAR support "
                    "from RF-DETR checkpoint", len(facets))
        return facets or None
    except Exception as e:  # noqa: BLE001
        logger.warning("model segmentation failed (%s) — fallback", e)
        return None
