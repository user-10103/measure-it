"""Empirical roof grammar — what a real roof report's numbers look like.

The constraint audit found no roof grammar anywhere in this codebase: no
primitive library, no topology catalogue, no statistical prior beyond one
orphaned Florida pitch distribution. Every bound the gate applied was invented
from a single address, which is how several of them ended up wrong.

These are different. They are derived from SIX EagleView Premium reports —
real contractor deliverables with the answers on them — and the useful property
is that they need NO ground truth to apply. `eave segments per facet` is
computable on any address we run, so six measured roofs become a scorecard for
every roof.

WHAT THE SAMPLE IS, because it governs how hard these may be pushed:
  n = 6, one contractor, one Tampa-metro quarter in 2023, two pairs almost
  neighbours -> effective n is nearer 4. All hip-dominant asphalt-shingle FL
  tract homes, 4,947-8,615 sqft, predominant pitch 5/12-7/12, ZERO parapets and
  ZERO "Simple" roofs. Nothing here describes a gable-dominant Midwest roof or
  a flat commercial deck, and this pipeline is scoped USA-wide.

So every bound below is PADDED >=20% beyond the observed range, and every check
is a WARNING. A breach means look, not fail. Tightening these on this sample
would reproduce exactly the mistake they were built to replace.

ALSO: the complexity metrics are three views of one latent variable (pairwise
|r| > 0.8). Gating all three triple-counts the same evidence and inflates the
false-warning rate, so only the tightest is shipped as a check.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

M2_PER_SQFT = 0.09290304
FT_PER_M = 3.280839895

# (observed_lo, observed_hi, padded_lo, padded_hi, CV, what it means)
INVARIANTS: Dict[str, Tuple[float, float, float, float, float, str]] = {
    # The strongest rule in the data by a distance: almost every facet has
    # exactly ONE eave run. 20 facets against 6 eave segments is broken, and you
    # can say so without knowing the right answer.
    "eaves_per_facet": (0.93, 1.13, 0.74, 1.36, 0.060,
                        "eave segments per facet"),
    # Near-Euler consistency on the facet graph.
    "edges_per_facet": (2.37, 3.25, 1.90, 3.90, 0.092,
                        "total edge segments per facet"),
    # On a hip-dominant roof the top-line length tracks the eave length.
    "ridgehip_over_eave": (0.83, 1.11, 0.66, 1.33, 0.087,
                           "(ridge + hip) / eave length"),
    # Area <-> perimeter consistency. Tightest of the length ratios.
    "perimeter_per_ksqft": (69.9, 88.3, 56.0, 106.0, 0.077,
                            "perimeter ft per 1000 sqft"),
    # eave/rake itself is unusable — 2.05 to 215.5, CV 1.78, because rake goes to
    # nearly zero on a pure hip roof and explodes the ratio. The bounded
    # reformulation is monotone and still encodes "these roofs are hip-dominant".
    "eave_share_of_perimeter": (0.672, 0.995, 0.54, 1.0, 0.13,
                                "eave / (eave + rake)"),
}


def _edge_counts(edges) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for e in edges or []:
        out[e.get("edge_type", "unspecified")] = out.get(
            e.get("edge_type", "unspecified"), 0) + 1
    return out


def grammar_findings(report_input: dict, model) -> List[dict]:
    """Return [{id, ok, detail}] for each invariant that can be computed.

    An invariant whose inputs are missing is OMITTED, never passed vacuously —
    a check that silently disappears reads exactly like a check that passed.
    """
    findings: List[dict] = []
    edges = report_input.get("edges") or []
    counts = _edge_counts(edges)
    ft = getattr(model, "edge_totals_ft", {}) or {}
    n_facets = getattr(model, "num_facets", 0) or 0
    area_sqft = getattr(model, "total_area_sqft", 0.0) or 0.0

    def add(key, value):
        lo_o, hi_o, lo, hi, cv, what = INVARIANTS[key]
        ok = lo <= value <= hi
        findings.append({
            "id": f"grammar_{key}", "ok": ok,
            "detail": (f"{what} = {value:.2f} "
                       f"({'within' if ok else 'OUTSIDE'} {lo:.2f}-{hi:.2f}; "
                       f"6 EagleView roofs observed {lo_o:.2f}-{hi_o:.2f}, "
                       f"CV {cv:.3f}, padded 20%. FL hip-dominant sample, "
                       f"effective n~4 — investigate, do not treat as failure)")})

    if n_facets:
        if counts.get("eave"):
            add("eaves_per_facet", counts["eave"] / n_facets)
        if edges:
            add("edges_per_facet", len(edges) / n_facets)

    eave_ft = ft.get("eave", 0.0)
    if eave_ft > 0.5:
        add("ridgehip_over_eave", (ft.get("ridge", 0.0) + ft.get("hip", 0.0)) / eave_ft)
        rake_ft = ft.get("rake", 0.0)
        if eave_ft + rake_ft > 0.5:
            add("eave_share_of_perimeter", eave_ft / (eave_ft + rake_ft))
        if area_sqft > 0:
            perim = eave_ft + rake_ft
            add("perimeter_per_ksqft", perim / (area_sqft / 1000.0))

    # An INEQUALITY, not a ratio: hip exceeded ridge on all six, minimum 1.58x,
    # but the magnitude ranges 1.58-3.55 and is not worth bounding. On a
    # hip-dominant roof the direction is the signal.
    if ft.get("ridge", 0.0) > 0.5 or ft.get("hip", 0.0) > 0.5:
        hip, ridge = ft.get("hip", 0.0), ft.get("ridge", 0.0)
        findings.append({
            "id": "grammar_hip_exceeds_ridge", "ok": hip > ridge,
            "detail": (f"hip {hip:.0f} ft vs ridge {ridge:.0f} ft — "
                       + ("hip-dominant as expected" if hip > ridge else
                          "ridge EXCEEDS hip, which did not occur on any of the "
                          "six reference roofs (min ratio 1.58). Expected on a "
                          "gable-dominant roof, so read with the roof type.")) })
    return findings
