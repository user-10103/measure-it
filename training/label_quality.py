"""Quality-gate for LiDAR auto-generated facet labels.

The label factory produces good labels on some roofs and garbage on others (56
speckle-facets, jagged boundaries, poor coverage). Training on the garbage would
degrade the model, so this gate scores each roof's labels on geometric quality and
returns KEEP / REJECT. Only KEPT roofs feed the training set. Running it over many
roofs also reports the *fraction* that pass — the number that decides whether the
auto-label plan is viable at all.

Pure geometry (polygons + footprint), no LiDAR/network — testable anywhere.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def score_labels(polygons: list, footprint, *,
                 min_facets: int = 3, max_facets: int = 14,
                 min_coverage: float = 0.55, min_facet_area: float = 2.0,
                 max_sliver_frac: float = 0.30, max_avg_vertices: float = 18.0,
                 max_overlap_frac: float = 0.15) -> dict:
    """Score one roof's facet labels. Returns metrics + ``passed`` + ``reasons``.

    A roof is KEPT only if all checks pass:
      - facet count in [min_facets, max_facets]      (not over/under-segmented)
      - facets cover >= min_coverage of the footprint (no big gaps)
      - few slivers                                   (no noise fragments)
      - boundaries not jagged (avg vertices low)      (clean geometry)
      - facets don't overlap much                     (clean partition)
    """
    from shapely.ops import unary_union

    polys = [p for p in polygons if p is not None and not p.is_empty and p.is_valid]
    n = len(polys)
    m: dict = {"n_facets": n}
    reasons: List[str] = []

    if n == 0:
        m.update(coverage=0.0, sliver_frac=1.0, avg_vertices=0.0, overlap_frac=0.0,
                 passed=False, reasons=["no facets"])
        return m

    fp_area = float(footprint.area) if footprint is not None else 0.0
    union = unary_union(polys)
    coverage = (union.intersection(footprint).area / fp_area) if fp_area > 0 else 0.0
    areas = np.array([p.area for p in polys], dtype=float)
    sliver_frac = float((areas < min_facet_area).mean())
    avg_vertices = float(np.mean([len(p.exterior.coords) - 1 for p in polys]))
    overlap_frac = float(max(0.0, (areas.sum() - union.area) / areas.sum()))

    m.update(coverage=round(coverage, 3), sliver_frac=round(sliver_frac, 3),
             avg_vertices=round(avg_vertices, 1), overlap_frac=round(overlap_frac, 3))

    if not (min_facets <= n <= max_facets):
        reasons.append(f"n_facets={n}")
    if coverage < min_coverage:
        reasons.append(f"coverage={coverage:.2f}")
    if sliver_frac > max_sliver_frac:
        reasons.append(f"slivers={sliver_frac:.2f}")
    if avg_vertices > max_avg_vertices:
        reasons.append(f"jagged_avg_v={avg_vertices:.0f}")
    if overlap_frac > max_overlap_frac:
        reasons.append(f"overlap={overlap_frac:.2f}")

    m["passed"] = len(reasons) == 0
    m["reasons"] = reasons
    return m
