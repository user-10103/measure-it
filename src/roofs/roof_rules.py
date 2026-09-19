"""Roof geometry, stated as rules over PLANES rather than heuristics over plans.

WHY THIS EXISTS
---------------
Edge typing today never reads a facet's plane. ``geom_edges`` contains zero
occurrences of ``plane_abc`` or ``slope_deg``; it types seams from 2-D corner
convexity and then, where LiDAR exists, refines using only the compass
``aspect_deg``, ``is_flat`` and ``median_z``. A 2:12 and a 12:12 facet meeting
are indistinguishable to it. Meanwhile ``annotate_facets_with_lidar`` puts the
full plane ``(a, b, c)`` on every measured facet and hands it to the classifier,
which ignores it.

A roof is not a set of regions that happen to touch. It is a piecewise-planar
surface, and that constrains what a seam between two facets can BE:

  * two facets on the same plane are ONE facet, and the line between them is
    not an edge at all — it is a segmentation artefact;
  * two facets on parallel planes at different heights meet at a STEP, not a
    fold — a parapet or a wall;
  * two facets on intersecting planes fold along their intersection line, and
    that line must pass through the seam we drew. If it does not, the seam is
    fictional: we cut a single surface in half and called the halves facets.

That last rule is the one nothing currently tests, and it is the one that
catches 601 Gulf Way — four parallel strips reported at 1/12, 4/12 and 7/12
with zero ridge and zero hip between them. Three different planes were fitted
to what the drawing shows as one continuous surface, and each seam was labelled
"transition" because the classifier's only question was whether the compass
aspects were within 45 degrees.

Everything here is pure geometry over (a, b, c) and a segment. No model, no
imagery, no network.

THE SIGNED QUANTITIES
---------------------
For planes P1, P2 (z = a*x + b*y + c) and a seam segment with midpoint m:

  step        = |P1(m) - P2(m)|          how far apart the surfaces are AT the
                                         seam. Zero for any real fold.
  fold_deg    = angle between normals    zero when coplanar or parallel.
  rise        = dz/ds along the segment  zero for a level edge (ridge, eave).
  convex      = both planes fall away    ridge/hip up; valley down.

``step`` and ``fold_deg`` are independent, and the pair identifies the seam:

    fold ~ 0, step ~ 0   -> same plane. FALSE SEAM, the facets should be merged.
    fold ~ 0, step > 0   -> parallel surfaces at different heights. STEP.
    fold > 0, step ~ 0   -> a genuine fold. Ridge / hip / valley by rise+convex.
    fold > 0, step > 0   -> the planes DO intersect, but not here. The seam was
                            drawn somewhere the surface does not actually
                            crease. FICTIONAL.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# A fold shallower than this is not a crease. Matches merge_coplanar_facets'
# angle_tol_deg, so the two stages cannot disagree about what "same plane" means.
COPLANAR_DEG = 5.0

# Surfaces this close at the seam are meeting, not stepping. The 3D exporter
# measured independently-fitted planes on a clean 4:12 gable disagreeing at the
# ridge by a median 0.9 cm and a p90 of 2.0 cm, so 10 cm is several times the
# honest noise floor and still far below any real parapet.
SEAM_STEP_M = 0.10

# |dz/ds| along an edge, in metres per metre. A ridge and an eave are LEVEL; a
# hip and a rake follow the slope. 1/12 pitch is 0.083, so 0.05 sits below the
# shallowest pitch anyone builds and will not call a genuine hip a ridge.
LEVEL_RISE = 0.05


def plane_z(abc: Sequence[float], x: float, y: float) -> float:
    return abc[0] * x + abc[1] * y + abc[2]


def normal(abc: Sequence[float]) -> Tuple[float, float, float]:
    """Unit upward normal of z = a*x + b*y + c."""
    a, b = float(abc[0]), float(abc[1])
    n = math.sqrt(a * a + b * b + 1.0)
    return (-a / n, -b / n, 1.0 / n)


def fold_angle_deg(p1: Sequence[float], p2: Sequence[float]) -> float:
    n1, n2 = normal(p1), normal(p2)
    d = max(-1.0, min(1.0, sum(u * v for u, v in zip(n1, n2))))
    return math.degrees(math.acos(d))


def rise_along(abc: Sequence[float], seg: Sequence[Sequence[float]]) -> float:
    """|dz/ds| along a segment, on that facet's plane. 0 = level."""
    (x0, y0), (x1, y1) = seg[0][:2], seg[-1][:2]
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 1e-9:
        return 0.0
    return abs((abc[0] * (x1 - x0) + abc[1] * (y1 - y0)) / L)


@dataclass
class SeamVerdict:
    """What the two planes say the seam between them is."""
    edge_type: str                 # ridge|hip|valley|step|false_seam|fictional
    step_m: float
    fold_deg: float
    rise: float
    convex: Optional[bool]
    real: bool                     # False => the partition is wrong here
    detail: str = ""


def classify_seam(p1: Optional[Sequence[float]],
                  p2: Optional[Sequence[float]],
                  seg: Sequence[Sequence[float]],
                  coplanar_deg: float = COPLANAR_DEG,
                  step_m: float = SEAM_STEP_M,
                  level_rise: float = LEVEL_RISE) -> SeamVerdict:
    """Type one interior seam from the two facet planes that meet at it."""
    if p1 is None or p2 is None:
        return SeamVerdict("unspecified", 0.0, 0.0, 0.0, None, True,
                           "a flank has no measured plane")
    (x0, y0), (x1, y1) = seg[0][:2], seg[-1][:2]
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    step = abs(plane_z(p1, mx, my) - plane_z(p2, mx, my))
    fold = fold_angle_deg(p1, p2)
    rise = (rise_along(p1, seg) + rise_along(p2, seg)) / 2.0

    if fold <= coplanar_deg:
        if step <= step_m:
            return SeamVerdict(
                "false_seam", step, fold, rise, None, False,
                f"same plane either side (fold {fold:.1f} deg, step {step*100:.0f} cm) "
                f"— these are one facet, not two")
        return SeamVerdict(
            "step", step, fold, rise, None, True,
            f"parallel surfaces {step:.2f} m apart — a step, not a fold")

    if step > step_m:
        return SeamVerdict(
            "fictional", step, fold, rise, None, False,
            f"the planes fold at {fold:.1f} deg but are {step*100:.0f} cm apart "
            f"HERE — their intersection line does not pass through this seam, so "
            f"the seam is a cut through one surface, not a crease in it")

    # A genuine fold. Step perpendicular to the seam into each side and ask
    # whether the surface falls away (convex: ridge/hip) or rises (valley).
    L = math.hypot(x1 - x0, y1 - y0) or 1.0
    px, py = -(y1 - y0) / L, (x1 - x0) / L          # unit normal in plan
    d = 0.5
    z_mid = plane_z(p1, mx, my)
    drop1 = plane_z(p1, mx - px * d, my - py * d) - z_mid
    drop2 = plane_z(p2, mx + px * d, my + py * d) - z_mid
    convex = drop1 < 0 and drop2 < 0
    concave = drop1 > 0 and drop2 > 0

    if concave:
        return SeamVerdict("valley", step, fold, rise, False, True,
                           f"surface rises on both sides (fold {fold:.1f} deg)")
    if convex:
        if rise <= level_rise:
            return SeamVerdict("ridge", step, fold, rise, True, True,
                               f"level crease, fold {fold:.1f} deg")
        return SeamVerdict("hip", step, fold, rise, True, True,
                           f"sloping crease {rise:.2f} m/m, fold {fold:.1f} deg")
    # One side up, one down: the surface passes through — a pitch change along
    # the same downhill direction, e.g. a gambrel or a shed dormer shoulder.
    return SeamVerdict("transition", step, fold, rise, None, True,
                       f"pitch change without a crease (fold {fold:.1f} deg)")


def classify_boundary(abc: Optional[Sequence[float]],
                      seg: Sequence[Sequence[float]],
                      level_rise: float = LEVEL_RISE) -> Tuple[str, float]:
    """Perimeter edge -> (eave|rake|unspecified, rise).

    An eave is LEVEL — it is where water leaves the roof, so it runs across the
    slope. A rake climbs with the slope up a gable end. This is the same
    geometry the current azimuth rule approximates, but read off the plane
    itself and reported in m/m, so a breach is a measurement rather than a
    cosine against a 45-degree window.
    """
    if abc is None:
        return "unspecified", 0.0
    rise = rise_along(abc, seg)
    return ("eave" if rise <= level_rise else "rake"), rise


@dataclass
class PartitionReport:
    false_seams: int = 0
    fictional_seams: int = 0
    total_seams: int = 0
    false_len_m: float = 0.0
    fictional_len_m: float = 0.0
    details: List[str] = field(default_factory=list)

    @property
    def unreal_fraction(self) -> float:
        return ((self.false_seams + self.fictional_seams) / self.total_seams
                if self.total_seams else 0.0)

    def to_dict(self) -> dict:
        return {"seams": self.total_seams,
                "false_seams": self.false_seams,
                "fictional_seams": self.fictional_seams,
                "false_seam_m": round(self.false_len_m, 2),
                "fictional_seam_m": round(self.fictional_len_m, 2),
                "unreal_seam_fraction": round(self.unreal_fraction, 3)}


def audit_partition(seams: Sequence[dict],
                    planes: Dict[int, Sequence[float]]) -> PartitionReport:
    """Check a facet partition against the planes it claims to represent.

    ``seams`` is [{"geometry_xy": [[x,y],...], "facets": (id1, id2)}, ...].
    A partition where many seams are false or fictional is not a roof — it is
    a picture cut into pieces, and every area, pitch and edge total derived
    from it inherits that.
    """
    rep = PartitionReport()
    for s in seams:
        ids = s.get("facets") or ()
        if len(ids) != 2:
            continue
        geom = s.get("geometry_xy") or []
        if len(geom) < 2:
            continue
        v = classify_seam(planes.get(ids[0]), planes.get(ids[1]), geom)
        rep.total_seams += 1
        if v.edge_type == "false_seam":
            rep.false_seams += 1
            rep.false_len_m += _seg_len(geom)
            rep.details.append(f"facets {ids[0]}/{ids[1]}: {v.detail}")
        elif v.edge_type == "fictional":
            rep.fictional_seams += 1
            rep.fictional_len_m += _seg_len(geom)
            rep.details.append(f"facets {ids[0]}/{ids[1]}: {v.detail}")
    return rep


def _seg_len(geom: Sequence[Sequence[float]]) -> float:
    return sum(math.dist(geom[i][:2], geom[i + 1][:2])
               for i in range(len(geom) - 1))


# --- wiring: apply the rules to a built edge list ---------------------------
INTERIOR_TYPES = {"ridge", "hip", "valley", "transition", "parapet",
                  "wall_flashing", "step_flashing", "unspecified"}
PERIMETER_TYPES = {"eave", "rake"}


def apply_plane_rules(edges: List[dict], facets, annotations,
                      touch_tol: float = 1.5) -> Tuple[List[dict], PartitionReport]:
    """Re-type edges from the facet PLANES, and audit the partition doing it.

    A refinement pass, not a replacement: an edge whose flanking facets have no
    measured plane keeps whatever the geometric classifier decided. Only edges
    where a plane actually exists are re-typed, so this can add evidence but
    never removes it.

    The planes were already being handed to the classifier — ``annotations``
    carries ``plane_abc`` for every measured facet — and ignored in favour of
    the compass ``aspect_deg`` alone. Reading them costs nothing and answers
    questions aspect cannot: whether the two surfaces actually MEET at the seam
    we drew, and whether a perimeter edge is level.
    """
    from shapely.geometry import Point

    lookup = [(f.facet_id, f.polygon) for f in facets
              if getattr(f, "polygon", None) is not None and not f.polygon.is_empty]
    planes = {fid: a["plane_abc"] for fid, a in (annotations or {}).items()
              if a.get("plane_abc") is not None}
    rep = PartitionReport()
    if not planes:
        return edges, rep

    for e in edges:
        geom = e.get("geometry_xy") or []
        if len(geom) < 2:
            continue
        (x0, y0), (x1, y1) = geom[0][:2], geom[-1][:2]
        mid = Point((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        near = _nearest_facets(mid, lookup, touch_tol)
        etype = e.get("edge_type")

        if etype in PERIMETER_TYPES and near:
            abc = planes.get(near[0])
            if abc is not None:
                new, rise = classify_boundary(abc, geom)
                e["rise_m_per_m"] = round(rise, 4)
                if new != etype:
                    e["edge_type"] = new
                    e["retyped_by"] = "plane"
            continue

        if etype in INTERIOR_TYPES and len(near) >= 2:
            p1, p2 = planes.get(near[0]), planes.get(near[1])
            if p1 is None or p2 is None:
                continue
            v = classify_seam(p1, p2, geom)
            e["seam_step_m"] = round(v.step_m, 4)
            e["seam_fold_deg"] = round(v.fold_deg, 2)
            rep.total_seams += 1
            if v.edge_type == "false_seam":
                rep.false_seams += 1
                rep.false_len_m += _seg_len(geom)
                rep.details.append(f"facets {near[0]}/{near[1]}: {v.detail}")
                # It is not an edge. Say unspecified rather than invent a crease
                # the surface does not have — a fabricated ridge is worse than a
                # declared gap, because it is billable footage.
                e["edge_type"] = "unspecified"
                e["retyped_by"] = "plane:false_seam"
            elif v.edge_type == "fictional":
                rep.fictional_seams += 1
                rep.fictional_len_m += _seg_len(geom)
                rep.details.append(f"facets {near[0]}/{near[1]}: {v.detail}")
                e["edge_type"] = "unspecified"
                e["retyped_by"] = "plane:fictional"
            elif v.edge_type != "unspecified" and v.edge_type != etype:
                e["edge_type"] = v.edge_type
                e["retyped_by"] = "plane"
    return edges, rep


def _nearest_facets(mid, lookup, touch_tol):
    """Facet ids whose boundary passes within touch_tol of a point, nearest first.

    Same rule the existing classifier uses, duplicated here rather than imported
    so this module stays free of geom_edges (which imports nothing from here —
    keeping the dependency one-way).
    """
    hits = sorted(((poly.boundary.distance(mid), fid) for fid, poly in lookup),
                  key=lambda t: t[0])
    return [fid for d, fid in hits if d <= touch_tol]
