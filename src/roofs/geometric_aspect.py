"""
Geometric per-facet aspect (downslope compass direction) from facet + outline.

Human/LS aspect labels are noise (18% agreement vs an oblique gold; opposite
facets on a hip roof get the same label, which blocks ridge detection). But
aspect is recoverable from GEOMETRY: a roof facet slopes DOWN from the roof
interior (ridge/hip) toward its EAVE -- the part of its boundary that lies on the
roof outline. So the downslope direction points from the facet centroid toward
its eave edge.

This makes opposite facets come out opposite (one N, one S), which is what the
ridge-vs-hip classifier needs, and it needs no trained model.
"""
import math
from typing import Optional

from shapely.geometry import Polygon

EAVE_TOL_M = 1.0          # facet-boundary within this of the outline = eave
MIN_EAVE_LEN_M = 0.5


def facet_aspect_deg(facet_poly: Polygon, outline: Optional[Polygon],
                     tol: float = EAVE_TOL_M) -> Optional[float]:
    """Downslope bearing (deg, 0=N, 90=E) from facet geometry, or None.

    Returns None when the eave can't be located (interior facet, no outline) so
    the caller can fall back to the labelled aspect.
    """
    if outline is None or facet_poly is None or facet_poly.is_empty:
        return None
    try:
        eave = facet_poly.boundary.intersection(outline.boundary.buffer(tol))
    except Exception:
        return None
    if eave.is_empty or eave.length < MIN_EAVE_LEN_M:
        return None
    ec = eave.centroid                          # low side of the facet
    c = facet_poly.centroid                     # interior-ish
    dx, dy = ec.x - c.x, ec.y - c.y             # facet -> eave = downslope
    if math.hypot(dx, dy) < 1e-6:
        return None
    return (90.0 - math.degrees(math.atan2(dy, dx))) % 360.0
