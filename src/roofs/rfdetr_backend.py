"""
RFDETRBackend — RF-DETR segmentation model wired as a RoofModelBackend.

Loads a checkpoint produced by training/train_rfdetr.py and runs inference
on a single chip (np.ndarray, HWC BGR or RGB uint8).  Returns pixel-coord
dicts that drop straight into segment_facets_ml / process_chip_rgb.

Class mapping (from training/roof_dataset train/_annotations.coco.json):
  category 0 = "roof_polygon"  →  outline
  category 1 = "facet"         →  facets[*].polygon

Usage:
    from src.roofs.rfdetr_backend import RFDETRBackend
    backend = RFDETRBackend("/path/to/checkpoint_best_ema.pth")
    pred = backend.predict(chip_rgb)   # chip is H×W×3 uint8 ndarray
    # pred = {"outline": [[x,y], ...] | None,
    #          "facets": [{"polygon": [[x,y], ...]}, ...]}
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports so the rest of the measure-it package loads without torch.
_model_cache: Dict[str, Any] = {}


def _mask_to_polygon(mask: np.ndarray, epsilon_frac: float = 0.005) -> Optional[List[List[float]]]:
    """Convert a binary HxW bool/uint8 mask to an [x,y] polygon.

    Uses the largest external contour and applies Douglas-Peucker
    simplification so the polygon is a manageable size.

    Returns None if the mask is empty or produces a degenerate contour.
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # largest contour by area
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 4:
        return None
    # simplify
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon_frac * peri, True)
    if len(approx) < 3:
        return None
    # approx shape is (N,1,2) → [[x,y], ...]
    return [[float(pt[0][0]), float(pt[0][1])] for pt in approx]


def _load_model(checkpoint_path: str):
    """Load (and cache) an RFDETRSeg model from a .pth checkpoint."""
    key = str(Path(checkpoint_path).resolve())
    if key in _model_cache:
        return _model_cache[key]

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    # rfdetr 1.7+ uses from_checkpoint; it auto-detects model size from weights.
    from rfdetr import RFDETRSegSmall  # noqa: imported for side-effects on first call
    from rfdetr import from_checkpoint  # top-level API

    logger.info(f"Loading RF-DETR checkpoint: {checkpoint_path}")
    model = from_checkpoint(checkpoint_path)
    _model_cache[key] = model
    logger.info("RF-DETR model loaded and cached.")
    return model


class RFDETRBackend:
    """RoofModelBackend that runs the trained RF-DETR-Seg checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to checkpoint_best_ema.pth (or any .pth produced by
        training/train_rfdetr.py).
    threshold:
        Confidence threshold passed to model.predict(). Default 0.35
        (lower than the rfdetr default of 0.5 so lower-confidence facets
        are still returned for the tiling/snap step to resolve).
    resolution:
        Input resolution the model was trained at (default 512, matching
        train_rfdetr.py default).  Used only for documentation; rfdetr
        resizes internally.
    """

    # Categories (must match training COCO categories order, 0-indexed)
    CAT_ROOF_POLYGON = 0  # "roof_polygon" → outline
    CAT_FACET = 1         # "facet"        → facets

    def __init__(
        self,
        checkpoint_path: str,
        threshold: float = 0.35,
        resolution: int = 512,
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.threshold = threshold
        self.resolution = resolution
        # Eagerly load so startup errors surface immediately.
        self._model = _load_model(self.checkpoint_path)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Run the model on a chip and return the pipeline prediction dict.

        Parameters
        ----------
        image:
            H×W×3 uint8 numpy array (RGB or BGR — rfdetr accepts both via
            PIL conversion internally when passed as ndarray).

        Returns
        -------
        dict with keys:
            "outline":  [[x, y], ...] in pixel coords, or None
            "facets":   [{"polygon": [[x, y], ...]}, ...]
        """
        import warnings
        warnings.filterwarnings("ignore")

        h, w = image.shape[:2]

        # rfdetr.predict accepts np.ndarray (HWC uint8 RGB)
        detections = self._model.predict(
            image,
            threshold=self.threshold,
        )

        # detections is sv.Detections with:
        #   .xyxy           (N,4) bboxes
        #   .mask           (N,H,W) bool array, or None
        #   .class_id       (N,) int
        #   .confidence     (N,)

        if detections is None or len(detections) == 0:
            logger.warning("RF-DETR returned no detections.")
            return {"outline": None, "facets": []}

        masks = detections.mask          # (N,H,W) bool | None
        class_ids = detections.class_id  # (N,) int
        confidences = getattr(detections, "confidence", None)

        if masks is None:
            logger.warning("RF-DETR detections have no masks — check model type.")
            return {"outline": None, "facets": []}

        # ---- roof_polygon → outline ----
        outline: Optional[List[List[float]]] = None
        roof_indices = np.where(class_ids == self.CAT_ROOF_POLYGON)[0]
        if len(roof_indices) > 0:
            # Pick highest-confidence roof polygon (or largest mask area)
            if confidences is not None:
                best_idx = roof_indices[np.argmax(confidences[roof_indices])]
            else:
                best_idx = roof_indices[
                    np.argmax([masks[i].sum() for i in roof_indices])
                ]
            outline = _mask_to_polygon(masks[best_idx])
            if outline is None:
                logger.warning("roof_polygon mask produced degenerate contour.")

        # ---- facet → facets ----
        facets: List[Dict[str, Any]] = []
        facet_indices = np.where(class_ids == self.CAT_FACET)[0]
        for idx in facet_indices:
            poly = _mask_to_polygon(masks[idx])
            if poly is not None:
                facets.append({"polygon": poly})

        logger.debug(
            f"RF-DETR predict: outline={'yes' if outline else 'no'}, "
            f"{len(facets)} facet(s) from {image.shape[:2]} chip"
        )
        return {"outline": outline, "facets": facets}

    # ------------------------------------------------------------------
    # Convenience: bulk predict on a list of chips
    # ------------------------------------------------------------------

    def predict_batch(
        self, images: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Predict on a list of chips, one at a time (model requires bs=1)."""
        return [self.predict(img) for img in images]

    def __repr__(self) -> str:
        return (
            f"RFDETRBackend(checkpoint={self.checkpoint_path!r}, "
            f"threshold={self.threshold}, resolution={self.resolution})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test (run this file directly: python -m src.roofs.rfdetr_backend)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Smoke-test RFDETRBackend on a chip image.")
    ap.add_argument("checkpoint", help="Path to checkpoint_best_ema.pth")
    ap.add_argument("image", help="Path to a chip PNG/JPG")
    ap.add_argument("--threshold", type=float, default=0.35)
    args = ap.parse_args()

    backend = RFDETRBackend(args.checkpoint, threshold=args.threshold)
    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: cannot read {args.image}", file=sys.stderr)
        sys.exit(1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred = backend.predict(img_rgb)

    print("=== Prediction ===")
    outline = pred["outline"]
    facets  = pred["facets"]
    print(f"outline: {len(outline)} pts" if outline else "outline: None")
    print(f"facets:  {len(facets)} facet(s)")
    for i, f in enumerate(facets):
        print(f"  facet[{i}]: {len(f['polygon'])} pts")
