"""
SAMRefiner post-processing for RF-DETR facet masks.

Takes the rough binary masks produced by RF-DETR and refines them to
pixel-accurate boundaries using Meta's Segment Anything Model (SAM).
Uses distance-guided point prompts auto-generated from the coarse mask
— no manual prompting needed.

Based on: https://github.com/linyq2117/samrefiner (MIT)

Usage
-----
from src.roofs.sam_refiner import build_sam_refiner, refine_prediction

refiner = build_sam_refiner(checkpoint="/path/to/sam_vit_h_4b8939.pth")

# prediction is the dict from RFDETRBackend.predict()
refined = refine_prediction(image_rgb, prediction, refiner)
# refined["facets"] now has pixel-accurate polygon boundaries
"""

import logging
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)

# SAM checkpoint download URL (vit_h is the best quality model)
SAM_VIT_H_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)
SAM_DEFAULT_CACHE = "/tmp/sam_vit_h_4b8939.pth"


def download_sam_checkpoint(dest: str = SAM_DEFAULT_CACHE) -> str:
    """Download sam_vit_h if not already cached. Returns path."""
    import urllib.request
    from pathlib import Path
    p = Path(dest)
    if not p.exists():
        log.info("Downloading SAM vit_h checkpoint (~2.4 GB)…")
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAM_VIT_H_URL, dest)
        log.info(f"SAM checkpoint saved to {dest}")
    return dest


def build_sam_refiner(
    checkpoint: Optional[str] = None,
    model_type: str = "vit_h",
    device: str = "cuda",
):
    """
    Load SAM and return a SamPredictor ready for mask refinement.

    Parameters
    ----------
    checkpoint : path to sam_vit_h_4b8939.pth. Downloads automatically if None.
    model_type : "vit_h" (best), "vit_l", or "vit_b" (fastest).
    device     : "cuda" or "cpu".

    Returns
    -------
    sam_predictor — a segment_anything.SamPredictor instance.
    """
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        raise ImportError(
            "segment_anything not installed. Run:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    if checkpoint is None:
        checkpoint = download_sam_checkpoint(SAM_DEFAULT_CACHE)

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam = sam.to(device)
    sam.eval()
    return SamPredictor(sam)


def _mask_to_points(mask: np.ndarray, n_pos: int = 5, n_neg: int = 3):
    """
    Generate positive (inside) and negative (outside) point prompts
    from a coarse binary mask using distance transform.

    Positive points: local maxima of the distance transform (furthest
    from the boundary — most confidently inside the mask).
    Negative points: sampled from the background near the mask edge.
    """
    import cv2
    from scipy.ndimage import maximum_filter

    mask_u8 = mask.astype(np.uint8)

    # Positive points — distance transform maxima
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    local_max = (dist == maximum_filter(dist, size=15)) & (dist > 2)
    ys, xs = np.where(local_max)
    if len(xs) == 0:
        ys, xs = np.where(mask_u8 > 0)
    # Sort by distance descending and take top n_pos
    dists = dist[ys, xs]
    order = np.argsort(dists)[::-1]
    ys, xs = ys[order[:n_pos]], xs[order[:n_pos]]
    pos_points = np.stack([xs, ys], axis=1)

    # Negative points — dilated background near mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dilated = cv2.dilate(mask_u8, kernel)
    bg_near = (dilated > 0) & (mask_u8 == 0)
    by, bx = np.where(bg_near)
    if len(bx) >= n_neg:
        idx = np.random.choice(len(bx), n_neg, replace=False)
        neg_points = np.stack([bx[idx], by[idx]], axis=1)
    else:
        neg_points = np.zeros((0, 2), dtype=int)

    points = np.concatenate([pos_points, neg_points], axis=0)
    labels = np.array([1] * len(pos_points) + [0] * len(neg_points))
    return points, labels


def _mask_to_polygon(mask: np.ndarray, simplify_eps: float = 1.0):
    """Largest contour of a binary mask → list of [x, y] vertices."""
    import cv2
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    eps = simplify_eps * cv2.arcLength(c, True) / 100
    approx = cv2.approxPolyDP(c, eps, True)
    pts = approx.reshape(-1, 2).tolist()
    return pts if len(pts) >= 3 else None


def refine_prediction(
    image_rgb: np.ndarray,
    prediction: dict,
    sam_predictor,
    min_area_px: int = 100,
) -> dict:
    """
    Refine RF-DETR facet masks using SAM.

    Parameters
    ----------
    image_rgb   : H×W×3 uint8 numpy array (RGB).
    prediction  : dict from RFDETRBackend.predict(), containing
                  "facets": list of {"polygon": [...], "mask": np.ndarray, ...}
    sam_predictor : SamPredictor from build_sam_refiner().
    min_area_px : drop refined masks smaller than this (noise guard).

    Returns
    -------
    New prediction dict with refined polygons in "facets".
    Facets without a "mask" key fall back to their original polygon.
    """
    import copy

    facets = prediction.get("facets", [])
    if not facets:
        return prediction

    # Set the image once — SAM encodes it internally
    sam_predictor.set_image(image_rgb)

    refined_facets = []
    for facet in facets:
        mask = facet.get("mask")  # np.ndarray [H, W] bool/uint8 or None

        if mask is None:
            # No mask available — use polygon to rasterise one
            poly = facet.get("polygon")
            if poly is None:
                refined_facets.append(facet)
                continue
            import cv2
            h, w = image_rgb.shape[:2]
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)

        mask_np = np.array(mask, dtype=np.uint8)
        if mask_np.sum() < min_area_px:
            refined_facets.append(facet)
            continue

        try:
            points, labels = _mask_to_points(mask_np)
            if len(points) == 0:
                refined_facets.append(facet)
                continue

            masks_sam, scores, _ = sam_predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )
            # Pick highest-scoring mask
            best_idx  = int(np.argmax(scores))
            best_mask = masks_sam[best_idx].astype(np.uint8)

            poly_refined = _mask_to_polygon(best_mask)
            if poly_refined is None or len(poly_refined) < 3:
                refined_facets.append(facet)
                continue

            new_facet = copy.copy(facet)
            new_facet["polygon"]          = poly_refined
            new_facet["mask"]             = best_mask
            new_facet["refined_by_sam"]   = True
            new_facet["sam_score"]        = float(scores[best_idx])
            refined_facets.append(new_facet)

        except Exception as e:
            log.warning(f"SAM refinement failed for one facet: {e}")
            refined_facets.append(facet)

    result = dict(prediction)
    result["facets"] = refined_facets
    return result
