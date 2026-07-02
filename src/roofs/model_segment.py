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


_ESRGAN_UPSAMPLER = None
_ESRGAN_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.1.0/RealESRGAN_x4plus.pth")


def _get_esrgan(tile: int = 256):
    """Load & cache a RealESRGAN_x4plus upsampler matching the training config
    (adresses/pipeline/upscale.py). Requires torch + basicsr + realesrgan."""
    global _ESRGAN_UPSAMPLER
    if _ESRGAN_UPSAMPLER is not None:
        return _ESRGAN_UPSAMPLER
    import os
    from urllib.request import urlretrieve
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    ckpt_dir = "/tmp/realesrgan_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, "RealESRGAN_x4plus.pth")
    if not os.path.exists(ckpt):
        logger.info("Downloading RealESRGAN_x4plus checkpoint…")
        urlretrieve(_ESRGAN_URL, ckpt)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                    num_grow_ch=32, scale=4)
    on_cuda = torch.cuda.is_available()
    _ESRGAN_UPSAMPLER = RealESRGANer(
        scale=4, model_path=ckpt, model=model, tile=tile, tile_pad=10,
        pre_pad=0, half=on_cuda,
    )
    return _ESRGAN_UPSAMPLER


def upscale_esrgan(rgb: np.ndarray, scale: int = 4) -> np.ndarray:
    """Real-ESRGAN x4 upscale, matching the training chip pipeline (the model
    trained on ESRGAN-upscaled chips). Returns uint8 RGB. Falls back to the
    ORIGINAL chip on any failure — but then detections won't match training, so
    the failure is logged loudly."""
    try:
        up = _get_esrgan()
        out, _ = up.enhance(rgb, outscale=scale)
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("ESRGAN upscale unavailable (%s) — feeding un-upscaled "
                       "chip; model input will NOT match training", e)
        return rgb


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


def gap_fill_facets(model_facets: List[Facet], outline_native,
                    lidar_points, min_gap_area_m2: float = 10.0,
                    min_points: int = 10) -> List[Facet]:
    """Phase C fusion: fill the part of the roof outline the model DIDN'T cover
    with facets segmented from the LiDAR points there, so the facet set tiles the
    full outline.

    Why: a weak model (0.19 mAP) detects only some facets; if those few are then
    stretched to fill the whole roof (the pipeline's remainder-absorption), their
    boundaries distort and the perimeter mis-types (eave<->rake). Adding real
    LiDAR facets for the uncovered region keeps the model's crisp interiors AND
    a correct outline-following perimeter.

    Returns model_facets + new LiDAR gap facets (ids renumbered to not collide).
    """
    from shapely.ops import unary_union
    import shapely

    valid = [f for f in model_facets
             if f.polygon is not None and not f.polygon.is_empty]
    if not valid or outline_native is None or outline_native.is_empty:
        return model_facets
    try:
        covered = unary_union([f.polygon for f in valid]).intersection(outline_native)
        gap = outline_native.difference(covered)
    except Exception:
        return model_facets
    if gap.is_empty or gap.area < min_gap_area_m2:
        logger.info("gap_fill: model covers the roof (uncovered %.1f m2) — no fill",
                    0.0 if gap.is_empty else gap.area)
        return model_facets
    if lidar_points is None or len(lidar_points) == 0:
        return model_facets

    xs = np.asarray(lidar_points['x'], dtype="float64")
    ys = np.asarray(lidar_points['y'], dtype="float64")
    in_gap = shapely.contains_xy(gap, xs, ys)
    if int(in_gap.sum()) < min_points:
        logger.info("gap_fill: %.0f m2 uncovered but <%d LiDAR pts there — skip",
                    gap.area, min_points)
        return model_facets

    from src.roofs.segment import segment_facets
    gap_facets = segment_facets(lidar_points[in_gap], method="kmeans")
    nid = max((f.facet_id for f in model_facets), default=0)
    out = list(model_facets)
    for f in gap_facets:
        nid += 1
        f.facet_id = nid
        f.label = nid
        out.append(f)
    logger.info("gap_fill: %.0f m2 uncovered -> +%d LiDAR facet(s) (%d pts); "
                "%d facet(s) total", gap.area, len(gap_facets),
                int(in_gap.sum()), len(out))
    return out


def segment_facets_model(naip_clipped_path, points_crs: str, checkpoint: str,
                         lidar_points=None, threshold: float = 0.40,
                         preprocess: bool = True,
                         upscale: bool = True) -> Optional[List[Facet]]:
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

        # Real-ESRGAN x4 to match the training chip pipeline. The model then
        # predicts in the UPSCALED pixel space, so divide the raster transform
        # by the actual upscale factor before mapping polygons to world.
        w0 = rgb.shape[1]
        if upscale:
            rgb = upscale_esrgan(rgb, scale=4)
        eff_scale = rgb.shape[1] / float(w0)          # 1.0 if upscale was skipped
        if eff_scale != 1.0:
            from affine import Affine
            transform = transform * Affine.scale(1.0 / eff_scale, 1.0 / eff_scale)

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
