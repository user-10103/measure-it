"""
Monocular per-facet pitch estimation (RGB-only path).

Wraps Depth Anything V2 (MIT licensed) to estimate facet slope from a single
aerial chip, with graceful fallbacks so the seam works before the model/weights
are present:
    depth_anything  ->  lidar_fallback (if a LiDAR slope is supplied)  ->  default

The model is lazy-loaded on first use; torch/transformers are NOT required to
import this module or to exercise the fallback paths.
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Standard roof pitches (rise:12), shared with metrics.py
STANDARD_PITCHES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24]


@dataclass
class PitchEstimate:
    """One facet's estimated pitch and where it came from."""
    facet_id: int
    slope_deg: float
    source: str          # "depth_anything" | "lidar_fallback" | "default"
    confidence: float     # 0-1


def slope_to_pitch_string(slope_deg: float) -> str:
    """Slope angle -> 'X:12' rounded to the nearest standard pitch."""
    rise_12 = math.tan(math.radians(slope_deg)) * 12.0
    pitch = min(STANDARD_PITCHES, key=lambda p: abs(p - rise_12))
    return f"{pitch}:12"


def _plane_slope_from_depth_patch(
    depth_patch: np.ndarray, mask: np.ndarray, gsd_m_per_px: float
) -> float:
    """Least-squares plane fit over masked depth pixels -> slope in degrees.

    depth_patch is metric depth (metres) per pixel. Fits depth = a*col + b*row,
    converts the per-pixel gradient to per-metre using the ground sample
    distance, and returns degrees(atan(hypot(a_m, b_m))).
    """
    rows, cols = np.nonzero(mask)
    if len(rows) < 3:
        return 0.0
    z = depth_patch[rows, cols].astype(float)
    A = np.column_stack([cols.astype(float), rows.astype(float), np.ones(len(cols))])
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    a_px, b_px = coeffs[0], coeffs[1]
    gsd = gsd_m_per_px or 1.0
    a_m, b_m = a_px / gsd, b_px / gsd               # metres of depth per metre of ground
    return math.degrees(math.atan(math.hypot(a_m, b_m)))


class MonocularPitchEstimator:
    """Estimate facet pitch from a single RGB chip via Depth Anything V2."""

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 device: str = "cpu", default_slope_deg: float = 18.0):
        self.model_name = model_name
        self.device = device
        self.default_slope_deg = default_slope_deg
        self._pipe = None
        self._load_failed = False

    def _ensure_model(self) -> bool:
        """Lazy-load the depth model. Returns False (never raises) if unavailable."""
        if self._pipe is not None:
            return True
        if self._load_failed:
            return False
        try:  # pragma: no cover - exercised only where torch is installed
            from transformers import pipeline
            self._pipe = pipeline(task="depth-estimation", model=self.model_name,
                                  device=self.device)
            return True
        except Exception as e:
            logger.warning("Depth Anything unavailable (%s); using fallback pitch.", e)
            self._load_failed = True
            return False

    def _depth_map(self, image: np.ndarray) -> Optional[np.ndarray]:  # pragma: no cover
        from PIL import Image as PILImage
        pil = PILImage.fromarray(image.astype(np.uint8))
        out = self._pipe(pil)
        depth = np.array(out["depth"], dtype=float)
        return depth

    def estimate_facet_pitch(
        self, image: np.ndarray, facet_polygon, gsd_m_per_px: float,
        lidar_slope_deg: Optional[float] = None, facet_id: int = 0,
    ) -> PitchEstimate:
        """Estimate one facet's slope, falling back as needed."""
        if self._ensure_model():  # pragma: no cover - needs torch
            try:
                depth = self._depth_map(image)
                mask = _polygon_mask(facet_polygon, depth.shape)
                slope = _plane_slope_from_depth_patch(depth, mask, gsd_m_per_px)
                conf = 0.7 if mask.sum() >= 9 else 0.3
                return PitchEstimate(facet_id, slope, "depth_anything", conf)
            except Exception as e:
                logger.warning("Depth inference failed (%s); falling back.", e)

        if lidar_slope_deg is not None:
            return PitchEstimate(facet_id, float(lidar_slope_deg), "lidar_fallback", 0.9)
        return PitchEstimate(facet_id, self.default_slope_deg, "default", 0.1)

    def estimate_all(
        self, image: np.ndarray, facet_polygons: List, gsd_m_per_px: float,
        lidar_slopes: Optional[List[Optional[float]]] = None,
    ) -> List[PitchEstimate]:
        out = []
        for i, poly in enumerate(facet_polygons):
            ls = lidar_slopes[i] if lidar_slopes is not None and i < len(lidar_slopes) else None
            out.append(self.estimate_facet_pitch(image, poly, gsd_m_per_px,
                                                 lidar_slope_deg=ls, facet_id=i))
        return out


def _polygon_mask(polygon, shape) -> np.ndarray:  # pragma: no cover - used with model
    """Rasterize a shapely polygon (pixel coords) to a boolean mask of `shape`."""
    from PIL import Image as PILImage, ImageDraw
    h, w = shape[:2]
    img = PILImage.new("L", (w, h), 0)
    coords = [(float(x), float(y)) for x, y in polygon.exterior.coords]
    ImageDraw.Draw(img).polygon(coords, fill=1)
    return np.array(img, dtype=bool)
