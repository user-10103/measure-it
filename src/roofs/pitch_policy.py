"""
Settled pitch policy for the roof-measurement map.

EMPIRICAL BASIS (Tampa Bay dataset, 2,991 cleaned roofs, 2026-06-06): per-facet
pitch MAGNITUDE is neither recoverable nor variable enough to chase as a
measurement:
  - True pitch is near-uniform ~4/12 across tract housing (3 oblique passes,
    incl. a LiDAR steep-claim roof, all read 4/12).
  - LiDAR per-facet slope is NOISE: same-roof hip facets (which must share one
    pitch) scattered 0-88 deg; within-roof range median 12.7 deg; 47% of roofs
    span >15 deg internally.
  - Human pitch bins are over-spread noise (21% agreement vs an oblique gold;
    three near-identical roofs labelled steep/flat/medium).

So the MAP reports pitch as an ANNOTATION, not a precision measurement:
  1. flat vs pitched     -- the one robust, valuable distinction
  2. pitched facets       -- the roof's common-pitch prior (default 4:12, or the
                             roof's own modal pitch IFF its facets are internally
                             consistent enough to trust)
  3. genuinely-steep      -- flagged needs_review (verify in oblique), never
                             silently asserted

Aspect (downslope direction) IS geometrically real and is kept (from the plane /
ridge geometry); only pitch MAGNITUDE is degraded to a prior.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

# Tampa tract-home prior. Override per-region if a better modal pitch is known.
DEFAULT_PITCH_X12 = 4

# A facet is "flat" below this slope. Equivalent surface/plan area ratio cutoff
# (1/cos) is ~1.02 at ~11 deg, which cleanly separates true-flat (~1.00) from a
# 4:12 pitch (~1.054) even when the slope number itself is noisy.
FLAT_SLOPE_DEG = 5.0
FLAT_AREA_RATIO_MAX = 1.02

# Below this within-roof slope spread, the facets agree well enough that we trust
# their median pitch instead of the prior (lets a genuinely-consistent steep roof
# report its real pitch). Above it, the per-facet slopes are noise -> use prior.
CONSISTENT_SPREAD_DEG = 8.0

# A pitched facet claiming to be steeper than this is outside the tract-home norm
# and sits where the slope data is least trustworthy -> flag rather than assert.
STEEP_REVIEW_DEG = 33.0

STANDARD_PITCHES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24]


@dataclass
class PitchAnnotation:
    """The map's pitch annotation for one facet."""
    is_flat: bool
    pitch_string: str          # "X:12"
    needs_review: bool         # steep/uncertain -> verify in oblique
    source: str                # "flat_detect" | "prior" | "roof_modal"
    confidence: float          # 0-1


def slope_to_x12(slope_deg: float) -> int:
    """Slope angle -> rise-in-12, snapped to the nearest standard pitch."""
    rise_12 = math.tan(math.radians(slope_deg)) * 12.0
    return min(STANDARD_PITCHES, key=lambda p: abs(p - rise_12))


def is_flat(
    slope_deg: Optional[float] = None,
    is_flat_hint: Optional[bool] = None,
    surface_area_m2: Optional[float] = None,
    plan_area_m2: Optional[float] = None,
) -> bool:
    """Flat detection, by reliability order: independent area-ratio > explicit
    model hint > slope threshold. (The area ratio is only meaningful when
    surface_area is an INDEPENDENT measurement, e.g. LiDAR 3D vs plan area.)"""
    if surface_area_m2 and plan_area_m2 and plan_area_m2 > 0:
        return (surface_area_m2 / plan_area_m2) <= FLAT_AREA_RATIO_MAX
    if is_flat_hint is not None:
        return bool(is_flat_hint)
    if slope_deg is not None:
        return slope_deg < FLAT_SLOPE_DEG
    return False


def roof_pitch_prior(facet_slopes: List[Optional[float]]) -> int:
    """Pick the roof's pitch prior (rise-in-12).

    Per-facet slopes are noise, so we only trust them when the roof's pitched
    facets AGREE (spread <= CONSISTENT_SPREAD_DEG) -- then use their median.
    Otherwise fall back to the tract-home default. This lets a genuinely
    consistent steep roof report its real pitch while the noisy majority gets the
    sane 4:12 prior.
    """
    vals = [s for s in facet_slopes if s is not None and s >= FLAT_SLOPE_DEG]
    if len(vals) >= 2 and (max(vals) - min(vals)) <= CONSISTENT_SPREAD_DEG:
        median = sorted(vals)[len(vals) // 2]
        return slope_to_x12(median)
    return DEFAULT_PITCH_X12


def classify_pitch(
    slope_deg: Optional[float] = None,
    is_flat_hint: Optional[bool] = None,
    surface_area_m2: Optional[float] = None,
    plan_area_m2: Optional[float] = None,
    roof_pitch_x12: int = DEFAULT_PITCH_X12,
    roof_modal: bool = False,
) -> PitchAnnotation:
    """Produce one facet's map pitch annotation under the settled policy."""
    if is_flat(slope_deg, is_flat_hint, surface_area_m2, plan_area_m2):
        return PitchAnnotation(True, "0:12", False, "flat_detect", 0.9)

    needs_review = slope_deg is not None and slope_deg >= STEEP_REVIEW_DEG
    return PitchAnnotation(
        is_flat=False,
        pitch_string=f"{roof_pitch_x12}:12",
        needs_review=needs_review,
        source="roof_modal" if roof_modal else "prior",
        confidence=0.2 if needs_review else 0.5,
    )
