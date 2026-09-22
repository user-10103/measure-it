"""Lift the measured roof into a 3D model — from LiDAR, not from obliques.

THE 45-DEGREE QUESTION
----------------------
EagleView flies N/S/E/W obliques at ~45 deg and ships a 3D model, and our
imagery is nadir-only, so "no obliques" has been carried as the reason we have
no 3D. It is not. Obliques are needed for the oblique PHOTO PAGES, for facades
and storey counts, and to eyeball a steep pitch the policy flagged. The roof
GEOMETRY needs three things, and we already have all three:

  1. Facet polygons in plan that share their edges EXACTLY. src/roofs/tiling.py
     nodes the combined linework and polygonizes it, so adjacent facets come
     out of one arrangement and their shared boundary is the same coordinates
     on both sides — not two nearly-equal boundaries.
  2. A plane per facet. PlaneModel is z = a*x + b*y + c, and fuse_sam_lidar
     records ``plane_abc`` — the intercept included, which is what places the
     plane in space.
  3. Those planes to AGREE where facets meet. Measured on 200 synthetic 4:12
     gables with a realistic 8 cm per-point residual, two independently fitted
     planes disagree at the ridge by a median of 0.9 cm and a p90 of 2.0 cm.
     At report scale that is watertight.

So the missing piece was never the imagery. It was that nothing wrote the file.

WHAT THIS REFUSES TO DO
-----------------------
A facet whose plane was DECLINED (too few LiDAR points, canopy, no coverage —
see fuse_sam_lidar.decline) has no plane. The tempting thing is to lay it flat
at the mean roof height so the model looks complete. That produces a model that
renders beautifully and is wrong in a way no viewer can see, which is the exact
failure this codebase keeps finding. Declined facets are emitted into their own
named group so a viewer can hide them, counted in the manifest, and never given
a fabricated pitch.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Two facets meeting at a ridge are welded when their vertices land within this
# distance in plan. The seams come from ONE noded arrangement, so true shared
# vertices are bit-identical and this only absorbs float round-trips; it is not
# a way to pull sloppy geometry together.
WELD_TOL_M = 0.02

# A z disagreement above this at a welded vertex is not float noise — it is two
# planes that genuinely do not meet there, which means a seam in the wrong
# place or a plane fitted through something that is not a roof.
CRACK_WARN_M = 0.15


@dataclass
class Roof3D:
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    faces: List[Tuple[List[int], str]] = field(default_factory=list)  # (idx, group)
    n_facets: int = 0
    n_declined: int = 0
    max_crack_m: float = 0.0
    median_crack_m: float = 0.0
    n_cracks_over_tol: int = 0
    notes: List[str] = field(default_factory=list)

    def manifest(self) -> dict:
        """What a reader needs to judge the model without opening it."""
        return {"facets": self.n_facets, "declined_facets": self.n_declined,
                "vertices": len(self.vertices), "faces": len(self.faces),
                "max_seam_gap_m": round(self.max_crack_m, 4),
                "median_seam_gap_m": round(self.median_crack_m, 4),
                "seams_over_tol": self.n_cracks_over_tol,
                "seam_tol_m": CRACK_WARN_M, "notes": list(self.notes)}


def _plane_z(abc, x: float, y: float) -> float:
    a, b, c = abc
    return a * x + b * y + c


def _coords_of(f: dict) -> List[Tuple[float, float]]:
    """Plan ring of a facet, from either representation.

    The measurement code passes shapely under ``polygon``; the report's own
    records pass ``polygon_xy`` as a plain coordinate list (they are JSON, and
    a shapely object is not). Accepting only one of those is what made the
    first wiring of this module a silent no-op.
    """
    poly = f.get("polygon")
    if poly is not None and getattr(poly, "exterior", None) is not None:
        if poly.is_empty:
            return []
        return [(float(x), float(y)) for x, y in list(poly.exterior.coords)[:-1]]
    xy = f.get("polygon_xy")
    if xy:
        pts = [(float(p[0]), float(p[1])) for p in xy]
        if len(pts) > 2 and pts[0] == pts[-1]:
            pts = pts[:-1]
        return pts
    return []


def _key(x: float, y: float, tol: float) -> Tuple[int, int]:
    return (int(round(x / tol)), int(round(y / tol)))


def build_roof_3d(facets: Sequence[dict], weld_tol_m: float = WELD_TOL_M) -> Roof3D:
    """Facet records -> a welded 3D surface.

    Each facet needs ``polygon`` (shapely, planar metric CRS) and ``plane_abc``.
    A facet without ``plane_abc`` is DECLINED: its outline is still emitted, at
    the height of whatever neighbours it shares vertices with, in the
    ``declined`` group — never with an invented pitch.
    """
    import numpy as np

    out = Roof3D()
    placed: Dict[Tuple[int, int], List[float]] = {}   # plan key -> z samples

    # Pass 1: every vertex of every PLANED facet contributes a z from its own
    # plane. A vertex shared by two facets therefore collects two samples, and
    # their spread is the seam gap.
    rings: List[Tuple[List[Tuple[float, float]], Optional[tuple], str]] = []
    skipped_no_geom = 0
    for f in facets:
        coords = _coords_of(f)
        if not coords:
            skipped_no_geom += 1
            continue
        abc = f.get("plane_abc")
        if abc is None:
            out.n_declined += 1
            rings.append((coords, None, "declined"))
            continue
        out.n_facets += 1
        grp = "flat" if f.get("is_flat") else "roof"
        rings.append((coords, tuple(float(v) for v in abc), grp))
        for x, y in coords:
            placed.setdefault(_key(x, y, weld_tol_m), []).append(_plane_z(abc, x, y))

    cracks = [max(v) - min(v) for v in placed.values() if len(v) > 1]
    if cracks:
        out.max_crack_m = float(max(cracks))
        out.median_crack_m = float(np.median(cracks))
        out.n_cracks_over_tol = int(sum(c > CRACK_WARN_M for c in cracks))
        if out.n_cracks_over_tol:
            logger.warning(
                "3D: %d shared vertex/vertices disagree by more than %.0f cm "
                "(max %.2f m). Two planes that do not meet at their shared seam "
                "means the seam is in the wrong place or a plane was fitted "
                "through canopy — the model will look solid regardless.",
                out.n_cracks_over_tol, 100 * CRACK_WARN_M, out.max_crack_m)
            out.notes.append(f"{out.n_cracks_over_tol} seam(s) over "
                             f"{CRACK_WARN_M} m")

    # Pass 2: emit. A welded vertex takes the MEAN of its samples so the two
    # sides of a ridge close exactly; declined facets take whatever their
    # neighbours established, and sit flat only where nothing else touches them.
    index: Dict[Tuple[int, int], int] = {}
    fallback_z = (float(np.median([z for v in placed.values() for z in v]))
                  if placed else 0.0)
    for coords, abc, grp in rings:
        idx: List[int] = []
        for x, y in coords:
            k = _key(x, y, weld_tol_m)
            if k not in index:
                if k in placed:
                    z = float(np.mean(placed[k]))
                elif abc is not None:
                    z = _plane_z(abc, x, y)
                else:
                    z = fallback_z
                index[k] = len(out.vertices)
                out.vertices.append((float(x), float(y), z))
            idx.append(index[k])
        if len(set(idx)) >= 3:
            out.faces.append((idx, grp))
    if out.n_declined:
        out.notes.append(f"{out.n_declined} facet(s) had no measured plane and "
                         f"carry NO pitch — group 'declined'")
    # REFUSE TO RETURN A QUIET NOTHING. The report's facet records carry
    # `polygon_xy`, not a shapely `polygon` — so the first wiring of this
    # function read f["polygon"], found None on every facet, skipped them all,
    # and returned an empty model that the caller silently declined to write.
    # No error, no file, no sign. An empty model is only legitimate when there
    # was nothing to build from.
    if facets and not out.faces:
        raise ValueError(
            f"{len(facets)} facet(s) supplied but none produced geometry "
            f"({skipped_no_geom} had no usable polygon). Expected 'polygon' "
            f"(shapely) or 'polygon_xy' ([[x, y], ...]) on each facet.")
    if skipped_no_geom:
        out.notes.append(f"{skipped_no_geom} facet(s) had no usable polygon")
    return out


def to_obj(model: Roof3D, name: str = "roof") -> str:
    """Wavefront OBJ. Chosen because every viewer opens it with no toolchain.

    Groups are preserved (``roof`` / ``flat`` / ``declined``) so a declined
    facet can be hidden rather than mistaken for measured geometry.
    """
    lines = [f"# {name} — measure-it roof model",
             f"# {model.manifest()}",
             "# Z is orthometric height in the chip's planar metric CRS.",
             ""]
    for x, y, z in model.vertices:
        lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
    cur = None
    for idx, grp in model.faces:
        if grp != cur:
            lines.append(f"g {grp}")
            cur = grp
        lines.append("f " + " ".join(str(i + 1) for i in idx))   # OBJ is 1-based
    return "\n".join(lines) + "\n"
