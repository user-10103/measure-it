"""
RFDETRBackend — RF-DETR segmentation model wired as a RoofModelBackend.

Loads a checkpoint produced by training/train_rfdetr.py and runs inference
on a single chip (np.ndarray, HWC uint8 RGB).  Returns pixel-coord dicts
that drop straight into segment_facets_ml / process_chip_rgb.

Class mapping (from training/roof_dataset train/_annotations.coco.json):
  COCO category id 1 = "roof_polygon"  →  outline   (rfdetr 0-indexes → 0)
  COCO category id 2 = "facet"         →  facets     (rfdetr 0-indexes → 1)

Usage:
    from src.roofs.rfdetr_backend import RFDETRBackend
    backend = RFDETRBackend("/path/to/checkpoint_best_ema.pth")
    pred = backend.predict(chip_rgb)   # H×W×3 uint8 RGB ndarray
    # pred = {"outline": [[x,y], ...] | None,
    #          "facets": [{"polygon": [[x,y], ...]}, ...]}

Hard deps (inference only, not needed at import time):
    rfdetr, supervision, opencv-python (cv2), numpy
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Module-level model cache: resolved_path_str → loaded model object.
# cv2 / rfdetr / supervision are imported lazily inside functions so that
# the rest of measure-it can import this module without those heavy deps.
_model_cache: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mask_to_polygon(
    mask: np.ndarray,
    scale_xy: Optional[tuple] = None,
    epsilon_frac: float = 0.005,
) -> Optional[List[List[float]]]:
    """Convert a binary H×W bool/uint8 mask to a simplified [x,y] polygon.

    Args:
        mask:         H×W bool or uint8 array (any non-zero = foreground).
        scale_xy:     Optional (sx, sy) tuple.  Every point is multiplied by
                      (sx, sy) after contour extraction.  Pass this when the
                      mask was produced at a different resolution than the
                      original chip (e.g. model-internal 512 vs chip 768×780).
        epsilon_frac: Douglas-Peucker ε as a fraction of the contour perimeter.

    Returns:
        [[x, y], ...] in the chip's pixel coordinate space, or None if the
        mask is empty / produces a degenerate contour (< 3 pts).
    """
    import cv2  # lazy — only needed at inference time

    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 4:
        return None

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon_frac * peri, True)
    if len(approx) < 3:
        return None

    # approx shape: (N, 1, 2) → [[x, y], ...]
    pts = [[float(pt[0][0]), float(pt[0][1])] for pt in approx]

    if scale_xy is not None:
        sx, sy = scale_xy
        pts = [[x * sx, y * sy] for x, y in pts]

    return pts


def _load_model(checkpoint_path: str) -> Any:
    """Load (and cache) an RF-DETR model from a .pth checkpoint.

    Uses rfdetr.from_checkpoint which auto-detects model size from the
    weight header — no need to know Small/Medium/Large at call time.
    """
    key = str(Path(checkpoint_path).resolve())
    if key in _model_cache:
        return _model_cache[key]

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    # rfdetr.from_checkpoint is the stable public API (rfdetr ≥ 1.7).
    # Do NOT import RFDETRSegSmall/etc. "for side-effects" — that breaks
    # on version changes and is unnecessary.
    from rfdetr import from_checkpoint  # noqa

    logger.info(f"Loading RF-DETR checkpoint: {checkpoint_path}")
    model = from_checkpoint(checkpoint_path)
    _model_cache[key] = model
    logger.info("RF-DETR model loaded and cached.")
    return model


# ---------------------------------------------------------------------------
# Public backend class
# ---------------------------------------------------------------------------

class RFDETRBackend:
    """RoofModelBackend using the trained RF-DETR-Seg checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to checkpoint_best_ema.pth (or any .pth from train_rfdetr.py).
    threshold:
        Confidence threshold for model.predict(). Default 0.35 — deliberately
        lower than rfdetr's default 0.5 so borderline facets survive for the
        tiling/snap step to resolve.
    resolution:
        Documented training resolution (default 512). Not passed to the model
        (rfdetr handles resizing internally); stored for __repr__ only.
    """

    # 0-indexed category ids as rfdetr sees them (COCO 1-indexed → 0-indexed).
    # Verified against training/roof_dataset/train/_annotations.coco.json:
    #   {"id": 1, "name": "roof_polygon"} → rfdetr class 0
    #   {"id": 2, "name": "facet"}        → rfdetr class 1
    CAT_ROOF_POLYGON = 0
    CAT_FACET        = 1

    def __init__(
        self,
        checkpoint_path: str,
        threshold: float = 0.35,
        resolution: int = 512,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.threshold = threshold
        self.resolution = resolution
        # Eagerly load so startup errors surface immediately, not at first call.
        self._model = _load_model(self.checkpoint_path)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Run the model on one chip image.

        Parameters
        ----------
        image:
            H×W×3 uint8 numpy array in RGB order.

        Returns
        -------
        {"outline": [[x,y], ...] | None, "facets": [{"polygon": [[x,y], ...]}, ...]}

        All coordinates are in the chip's own pixel space (same H×W as the
        input image), regardless of the model's internal processing resolution.
        """
        import warnings
        warnings.filterwarnings("ignore")

        img_h, img_w = image.shape[:2]

        detections = self._model.predict(image, threshold=self.threshold)

        # sv.Detections attributes we use:
        #   .mask       (N, mask_H, mask_W) bool array, or None
        #   .class_id   (N,) int  (0-indexed)
        #   .confidence (N,) float
        if detections is None or len(detections) == 0:
            logger.warning("RF-DETR returned no detections.")
            return {"outline": None, "facets": []}

        masks      = detections.mask
        class_ids  = detections.class_id
        confidence = getattr(detections, "confidence", None)

        if masks is None:
            logger.warning("RF-DETR detections carry no masks — wrong model type?")
            return {"outline": None, "facets": []}

        # ----------------------------------------------------------------
        # RESOLUTION CHECK — detect and correct mask/chip size mismatch.
        # rfdetr may return masks at its internal resolution (e.g. 512×512)
        # rather than the chip's native size.  If they differ, every polygon
        # would be in the wrong coordinate space.  We compute a scale factor
        # and pass it through to _mask_to_polygon.
        # ----------------------------------------------------------------
        _, mask_h, mask_w = masks.shape
        if mask_h != img_h or mask_w != img_w:
            scale_xy = (img_w / mask_w, img_h / mask_h)
            logger.warning(
                f"Mask size ({mask_w}×{mask_h}) ≠ chip size ({img_w}×{img_h}). "
                f"Rescaling polygons by {scale_xy[0]:.4f}×{scale_xy[1]:.4f}."
            )
        else:
            scale_xy = None

        # ----------------------------------------------------------------
        # roof_polygon (cat 0) → outline
        # ----------------------------------------------------------------
        outline: Optional[List[List[float]]] = None
        roof_idx = np.where(class_ids == self.CAT_ROOF_POLYGON)[0]
        if len(roof_idx) > 0:
            # Prefer highest confidence; fall back to largest mask area.
            if confidence is not None:
                best = roof_idx[np.argmax(confidence[roof_idx])]
            else:
                best = roof_idx[np.argmax([masks[i].sum() for i in roof_idx])]
            outline = _mask_to_polygon(masks[best], scale_xy=scale_xy)
            if outline is None:
                logger.warning("roof_polygon mask produced degenerate contour.")

        # ----------------------------------------------------------------
        # facet (cat 1) → facets list
        # ----------------------------------------------------------------
        facets: List[Dict[str, Any]] = []
        facet_idx = np.where(class_ids == self.CAT_FACET)[0]
        for i in facet_idx:
            poly = _mask_to_polygon(masks[i], scale_xy=scale_xy)
            if poly is not None:
                facets.append({"polygon": poly})

        logger.debug(
            f"RF-DETR: chip {img_w}×{img_h}, masks {mask_w}×{mask_h}, "
            f"outline={'yes' if outline else 'no'}, {len(facets)} facet(s)"
        )
        return {"outline": outline, "facets": facets}

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Predict on a list of chips sequentially (model requires bs=1)."""
        return [self.predict(img) for img in images]

    def __repr__(self) -> str:
        return (
            f"RFDETRBackend(checkpoint={self.checkpoint_path!r}, "
            f"threshold={self.threshold}, resolution={self.resolution})"
        )


# ---------------------------------------------------------------------------
# Smoke-test — run directly:
#   cd /workspace/measure-it
#   python src/roofs/rfdetr_backend.py output/checkpoint_best_ema.pth \
#          training/roof_dataset/val/some_chip.png
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys
    import cv2  # fine here — this block only runs when invoked directly

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Smoke-test RFDETRBackend on a chip.")
    ap.add_argument("checkpoint", help="Path to checkpoint_best_ema.pth")
    ap.add_argument("image",      help="Path to a chip PNG/JPG")
    ap.add_argument("--threshold", type=float, default=0.35)
    args = ap.parse_args()

    backend = RFDETRBackend(args.checkpoint, threshold=args.threshold)

    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"ERROR: cannot read {args.image}", file=sys.stderr)
        sys.exit(1)
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_rgb.shape[:2]

    # Access raw detections to verify mask resolution before polygon conversion.
    import warnings, numpy as np
    warnings.filterwarnings("ignore")
    raw = backend._model.predict(img_rgb, threshold=args.threshold)

    print(f"\n=== Raw detection check ===")
    print(f"Chip size : {img_w} × {img_h}")
    if raw is not None and len(raw) > 0 and raw.mask is not None:
        _, mh, mw = raw.mask.shape
        match = "✓ MATCH" if (mh == img_h and mw == img_w) else f"✗ MISMATCH — will rescale by ({img_w/mw:.4f}, {img_h/mh:.4f})"
        print(f"Mask size : {mw} × {mh}  {match}")
        print(f"N detections: {len(raw)}")
        for det_i in range(len(raw)):
            cid  = raw.class_id[det_i]
            conf = raw.confidence[det_i] if raw.confidence is not None else float('nan')
            name = {0: "roof_polygon", 1: "facet"}.get(int(cid), f"cat{cid}")
            ys, xs = np.where(raw.mask[det_i])
            if len(xs):
                bbox = f"x:[{xs.min()}-{xs.max()}] y:[{ys.min()}-{ys.max()}]"
            else:
                bbox = "empty mask"
            print(f"  [{det_i}] class={name}({cid}) conf={conf:.3f}  mask_bbox={bbox}")
    else:
        print("No detections or no masks.")

    # Now run the full predict (with rescaling applied).
    pred    = backend.predict(img_rgb)
    outline = pred["outline"]
    facets  = pred["facets"]

    print(f"\n=== Prediction output ===")
    if outline:
        xs = [p[0] for p in outline]
        ys = [p[1] for p in outline]
        print(f"outline : {len(outline)} pts  bbox x:[{min(xs):.1f}-{max(xs):.1f}] y:[{min(ys):.1f}-{max(ys):.1f}]")
        # Sanity: outline bbox should span most of the chip
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        if x_span < img_w * 0.3 or y_span < img_h * 0.3:
            print(f"  ⚠ WARNING: outline spans only {x_span/img_w:.0%} × {y_span/img_h:.0%} of chip — may be swapped with facet")
    else:
        print("outline : None")

    print(f"facets  : {len(facets)} facet(s)")
    for i, f in enumerate(facets):
        xs = [p[0] for p in f["polygon"]]
        ys = [p[1] for p in f["polygon"]]
        print(f"  [{i}] {len(f['polygon'])} pts  bbox x:[{min(xs):.1f}-{max(xs):.1f}] y:[{min(ys):.1f}-{max(ys):.1f}]")

    # Final verdict
    print()
    issues = []
    if outline and (max(xs := [p[0] for p in outline]) - min(xs)) < img_w * 0.3:
        issues.append("outline too small (category swap?)")
    if not facets:
        issues.append("no facets detected (lower --threshold or check category mapping)")
    if issues:
        print("⚠ ISSUES:", "; ".join(issues))
    else:
        print("✓ Looks good — outline spans chip, facets present")
