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

# ---------------------------------------------------------------------------
# Bayesian MAP pitch snapping (Fix 2)
#
# Standard pitches with Florida tract-home probability mass.  The prior
# encodes that 4:12 dominates FL residential, 5:12 and 3:12 are common,
# steeper pitches are rare.  Combining this with a Gaussian measurement
# likelihood lets us snap a noisy slope to the most probable standard pitch
# rather than blindly taking the nearest or the default 4:12.
# ---------------------------------------------------------------------------
_BAYES_PITCHES_X12 = [2,    3,    4,    5,    6,    7,    8,   10,   12]
_BAYES_PRIOR       = [0.03, 0.08, 0.45, 0.20, 0.12, 0.06, 0.04, 0.01, 0.01]

# ---------------------------------------------------------------------------
# Symmetric pitch pairs (Fix 3)
#
# A hip roof's N and S facets share one pitch by construction (same rise,
# same run). Likewise for E/W, NE/SW, NW/SE. When opposite-aspect facets
# disagree by more than CONSISTENT_SPREAD_DEG we pool their measurements
# and replace both with the mean — halves per-facet variance on hip roofs.
# ---------------------------------------------------------------------------
_OPPOSITE_ASPECTS = {
    "N": "S",  "S": "N",
    "E": "W",  "W": "E",
    "NE": "SW", "SW": "NE",
    "NW": "SE", "SE": "NW",
}


@dataclass
class PitchAnnotation:
    """The map's pitch annotation for one facet."""
    is_flat: bool
    pitch_string: str          # "X:12"
    needs_review: bool         # steep/uncertain -> verify in oblique
    source: str                # "flat_detect" | "prior" | "roof_modal" | "bayes"
    confidence: float          # 0-1


def slope_to_x12(slope_deg: float) -> int:
    """Slope angle -> rise-in-12, snapped to the nearest standard pitch."""
    rise_12 = math.tan(math.radians(slope_deg)) * 12.0
    return min(STANDARD_PITCHES, key=lambda p: abs(p - rise_12))


def bayesian_snap_pitch(slope_deg: float, sigma_deg: float = 2.5) -> int:
    """MAP estimate: snap a measured slope to the most probable standard pitch.

    Combines a Gaussian measurement likelihood (centred on ``slope_deg`` with
    width ``sigma_deg``) with the Florida tract-home pitch prior.  Returns the
    rise-in-12 integer whose posterior is highest.

    ``sigma_deg`` should reflect the actual measurement uncertainty:
      - Raw LiDAR (Fix 1 path): ~1.5 deg  → tighter snapping
      - Raster-derived slope:   ~4-5 deg  → wider snapping, prior dominates

    Args:
        slope_deg: Measured slope in degrees (0–90).
        sigma_deg: Measurement uncertainty in degrees.

    Returns:
        Rise-in-12 integer (a value from ``_BAYES_PITCHES_X12``).
    """
    best_pitch = _BAYES_PITCHES_X12[0]
    best_posterior = -1.0
    for p_x12, prior in zip(_BAYES_PITCHES_X12, _BAYES_PRIOR):
        pitch_deg = math.degrees(math.atan(p_x12 / 12.0))
        diff = slope_deg - pitch_deg
        likelihood = math.exp(-0.5 * (diff / sigma_deg) ** 2)
        posterior = likelihood * prior
        if posterior > best_posterior:
            best_posterior = posterior
            best_pitch = p_x12
    return best_pitch


def enforce_symmetric_pitch(facets: list) -> None:
    """Pool slope estimates from geometrically opposite-aspect facets in-place.

    A hip roof's N and S facets share one pitch by construction (identical
    rise and run).  When a symmetric pair's slopes differ by more than
    ``CONSISTENT_SPREAD_DEG`` the individual measurements are noise: we
    replace both with their mean, halving per-facet variance without
    discarding signal.

    Operates on any objects with ``slope_deg`` (float|None) and
    ``aspect_bin`` (str|None) attributes — works for both the LiDAR ``Facet``
    and the RGB/ML path facet dicts.

    Flat facets (``slope_deg < FLAT_SLOPE_DEG``) are left untouched.

    Args:
        facets: List of Facet-like objects.  Modified in-place.
    """
    from collections import defaultdict

    by_aspect: dict = defaultdict(list)
    for f in facets:
        ab = (getattr(f, "aspect_bin", None) or "").upper().strip()
        sd = getattr(f, "slope_deg", None)
        if ab and sd is not None and sd >= FLAT_SLOPE_DEG:
            by_aspect[ab].append(f)

    visited: set = set()
    for aspect, fs in list(by_aspect.items()):
        if aspect in visited:
            continue
        opp = _OPPOSITE_ASPECTS.get(aspect)
        if not opp or opp not in by_aspect:
            continue
        visited.add(aspect)
        visited.add(opp)

        paired = [f for f in fs + by_aspect[opp]
                  if getattr(f, "slope_deg", None) is not None]
        if len(paired) < 2:
            continue

        slopes = [f.slope_deg for f in paired]
        spread = max(slopes) - min(slopes)
        if spread <= CONSISTENT_SPREAD_DEG:
            continue  # Already consistent — no correction needed.

        mean_slope = sum(slopes) / len(slopes)
        for f in paired:
            f.slope_deg = mean_slope


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


def roof_pitch_prior(
    facet_slopes: List[Optional[float]],
    sigma_deg: float = 2.5,
) -> int:
    """Pick the roof's pitch prior (rise-in-12).

    When pitched facets agree (spread <= CONSISTENT_SPREAD_DEG) we trust their
    median and snap it to the most probable standard pitch via Bayesian MAP
    (``bayesian_snap_pitch``).  The MAP estimator is strictly better than a
    nearest-neighbour snap: it accounts for the FL prior so a measured 15.5 deg
    correctly snaps to 4:12 (14.0 deg) rather than 5:12 (22.6 deg) even though
    5:12 is geometrically equidistant.

    When facets disagree we fall back to the DEFAULT_PITCH_X12 (4:12) prior.

    Args:
        facet_slopes: Per-facet slope_deg values (None entries ignored).
        sigma_deg: Measurement uncertainty forwarded to ``bayesian_snap_pitch``.
            Use ~1.5 for raw-LiDAR fits, ~4.0 for raster-derived fits.

    Returns:
        Rise-in-12 integer.
    """
    vals = [s for s in facet_slopes if s is not None and s >= FLAT_SLOPE_DEG]
    if len(vals) >= 2 and (max(vals) - min(vals)) <= CONSISTENT_SPREAD_DEG:
        median = sorted(vals)[len(vals) // 2]
        return bayesian_snap_pitch(median, sigma_deg=sigma_deg)
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
    source = "roof_modal" if roof_modal else "prior"
    return PitchAnnotation(
        is_flat=False,
        pitch_string=f"{roof_pitch_x12}:12",
        needs_review=needs_review,
        source=source,
        confidence=0.2 if needs_review else 0.5,
    )
