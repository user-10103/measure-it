"""
Edge Classification Module

Classifies roof edges as ridge, hip, valley, eave, or rake.
Computes edge lengths and intersection lines between plane pairs.

Reference: agent.md §8, CLAUDE.md §4.1
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict

import math

import numpy as np
from shapely.geometry import (
    LineString, Polygon, Point, MultiPolygon, MultiLineString, GeometryCollection,
)
from shapely.ops import nearest_points, unary_union

from src.roofs.plane_fit import PlaneModel

logger = logging.getLogger(__name__)

# --- geometric edge typing (RGB path, ported from adresses cell38_p7v1) ---
COPLANAR_ANGLE_TOL_DEG = 8.0   # facets with normals within this are the same plane
COPLANAR_ZGAP_M = 0.3          # ...and within this vertical gap at the shared point
OUTLINE_SIMPLIFY_M = 1.0       # Douglas-Peucker on the outline -> straight sides
MIN_EDGE_LEN_M = 0.5           # drop slivers shorter than this
# ridge vs hip (among convex edges): a ridge joins facets facing ~OPPOSITE ways
# (downslopes ~180deg apart -> cos ~ -1); a hip joins facets ~90deg apart
# (cos ~ 0). cos(angle between gradients) below this => ridge. This is robust to
# the approximate snapped line direction and tolerates 45deg-binned aspects,
# unlike the old absolute along-line cutoff that misread ridges as hips.
RIDGE_OPPOSE_COS_MAX = -0.5    # gradients > 120deg apart => ridge


class EdgeType(Enum):
    """Roof edge classification types."""
    RIDGE = "ridge"      # Both planes slope downward away from line
    HIP = "hip"          # Planes slope downward away from each other
    VALLEY = "valley"    # Planes slope downward towards line
    EAVE = "eave"        # Lower edge at wall line
    RAKE = "rake"        # Gable end edge
    UNKNOWN = "unknown"


@dataclass
class RoofEdge:
    """
    Represents a classified roof edge.

    Attributes:
        edge_id: Unique identifier
        edge_type: Classification (ridge, hip, valley, etc.)
        geometry: LineString geometry
        length_m: Edge length in meters
        facet_ids: Tuple of adjacent facet IDs (None for boundary edges)
        dihedral_angle_deg: Angle between adjacent planes (if applicable)
    """
    edge_id: int
    edge_type: EdgeType
    geometry: LineString
    length_m: float
    facet_ids: Tuple[Optional[int], Optional[int]]
    dihedral_angle_deg: Optional[float] = None

    def to_dict(self) -> dict:
        """Export as dict for JSON serialization."""
        return {
            "edge_id": self.edge_id,
            "edge_type": self.edge_type.value,
            "length_m": round(self.length_m, 2),
            "facet_ids": list(self.facet_ids),
            "dihedral_angle_deg": (
                round(self.dihedral_angle_deg, 1)
                if self.dihedral_angle_deg is not None
                else None
            ),
            "geometry_wkt": self.geometry.wkt
        }


def compute_plane_intersection(
    plane1: PlaneModel,
    plane2: PlaneModel,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float]
) -> Optional[LineString]:
    """
    Compute intersection line of two planes.

    Reference: agent.md §8.1

    For planes:
        z1 = a1*x + b1*y + c1
        z2 = a2*x + b2*y + c2

    Intersection where z1 = z2:
        (a1-a2)*x + (b1-b2)*y + (c1-c2) = 0

    Args:
        plane1, plane2: PlaneModel objects
        x_range: (x_min, x_max) bounds
        y_range: (y_min, y_max) bounds

    Returns:
        LineString of intersection clipped to bounds, or None if parallel
    """
    # Coefficients of intersection line equation
    da = plane1.a - plane2.a
    db = plane1.b - plane2.b
    dc = plane1.c - plane2.c

    # Check if planes are parallel (coefficients nearly equal)
    if abs(da) < 1e-10 and abs(db) < 1e-10:
        logger.debug("Planes are parallel - no intersection line")
        return None

    # Find two points on the intersection line within bounds
    x_min, x_max = x_range
    y_min, y_max = y_range

    points = []

    # Try y boundaries
    for y in [y_min, y_max]:
        if abs(da) > 1e-10:
            x = -(db * y + dc) / da
            if x_min <= x <= x_max:
                z = plane1.predict(np.array([x]), np.array([y]))[0]
                points.append((x, y, z))

    # Try x boundaries
    for x in [x_min, x_max]:
        if abs(db) > 1e-10:
            y = -(da * x + dc) / db
            if y_min <= y <= y_max:
                z = plane1.predict(np.array([x]), np.array([y]))[0]
                points.append((x, y, z))

    if len(points) < 2:
        return None

    # Use first two distinct points
    points = list(set(points))[:2]

    if len(points) < 2:
        return None

    return LineString([(p[0], p[1]) for p in points])


def compute_dihedral_angle(plane1: PlaneModel, plane2: PlaneModel) -> float:
    """
    Compute dihedral angle between two planes.

    The dihedral angle is the angle between the plane normals.

    Args:
        plane1, plane2: PlaneModel objects

    Returns:
        Angle in degrees (0-180)
    """
    n1 = np.array(plane1.normal)
    n2 = np.array(plane2.normal)

    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)

    return np.degrees(angle_rad)


def classify_intersection_edge(
    plane1: PlaneModel,
    plane2: PlaneModel,
    intersection_line: LineString
) -> EdgeType:
    """
    Classify intersection edge as ridge, hip, or valley.

    Reference: agent.md §8.3

    - Ridge: both planes slope downward away from line
    - Hip: planes slope downward away from each other
    - Valley: planes slope downward towards line

    Args:
        plane1, plane2: PlaneModel objects
        intersection_line: LineString geometry

    Returns:
        EdgeType classification
    """
    # Get midpoint of intersection
    mid = intersection_line.interpolate(0.5, normalized=True)
    mx, my = mid.x, mid.y

    # Direction perpendicular to intersection line
    coords = list(intersection_line.coords)
    if len(coords) < 2:
        return EdgeType.UNKNOWN

    dx = coords[1][0] - coords[0][0]
    dy = coords[1][1] - coords[0][1]
    length = np.sqrt(dx**2 + dy**2)

    if length < 1e-10:
        return EdgeType.UNKNOWN

    # Perpendicular direction (normalized)
    perp_x = -dy / length
    perp_y = dx / length

    # Check gradient direction for each plane relative to perpendicular
    # Gradient direction is (a, b) for plane z = ax + by + c

    # Dot product of gradient with perpendicular direction
    dot1 = plane1.a * perp_x + plane1.b * perp_y
    dot2 = plane2.a * perp_x + plane2.b * perp_y

    # If both gradients point same direction relative to perp: ridge or valley
    # If gradients point opposite directions: hip

    if dot1 * dot2 > 0:
        # Same direction - both slope towards or away from line
        # Check if slopes point toward line (valley) or away (ridge)
        # Use Z values: if moving perpendicular increases Z, it's a ridge
        z_mid = plane1.predict(np.array([mx]), np.array([my]))[0]
        z_perp = plane1.predict(
            np.array([mx + perp_x]),
            np.array([my + perp_y])
        )[0]

        if z_perp < z_mid:
            return EdgeType.RIDGE
        else:
            return EdgeType.VALLEY
    else:
        # Opposite directions - hip edge
        return EdgeType.HIP


def extract_boundary_edges(
    polygon: Polygon,
    facet_id: int
) -> List[RoofEdge]:
    """
    Extract boundary edges from a facet polygon.

    Boundary edges are classified as eave (lower edges) or rake (gable ends).

    Args:
        polygon: Facet boundary polygon
        facet_id: Facet identifier

    Returns:
        List of RoofEdge objects
    """
    edges = []
    coords = list(polygon.exterior.coords)

    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]

        line = LineString([p1, p2])
        length = line.length

        # Simple classification: horizontal edges are eaves, others are rakes
        # More sophisticated would use building footprint comparison
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])

        # If edge is more horizontal than vertical, likely eave
        if dx > dy:
            edge_type = EdgeType.EAVE
        else:
            edge_type = EdgeType.RAKE

        edge = RoofEdge(
            edge_id=len(edges),
            edge_type=edge_type,
            geometry=line,
            length_m=length,
            facet_ids=(facet_id, None)
        )
        edges.append(edge)

    return edges


def classify_all_edges(
    facet_polygons: List[Tuple[int, Polygon]],
    planes: List[PlaneModel],
    bounds: Tuple[float, float, float, float]
) -> List[RoofEdge]:
    """
    Classify all edges for a multi-facet roof.

    Finds intersection edges between facet pairs and boundary edges.

    Args:
        facet_polygons: List of (facet_id, polygon) tuples
        planes: List of PlaneModel objects (one per facet)
        bounds: (x_min, y_min, x_max, y_max) bounds

    Returns:
        List of all RoofEdge objects
    """
    x_min, y_min, x_max, y_max = bounds
    all_edges = []
    edge_counter = 0

    n_facets = len(facet_polygons)

    # Find intersection edges between facet pairs
    for i in range(n_facets):
        for j in range(i + 1, n_facets):
            facet_id_i, poly_i = facet_polygons[i]
            facet_id_j, poly_j = facet_polygons[j]
            plane_i = planes[i]
            plane_j = planes[j]

            # Check if polygons are adjacent (touch or overlap)
            if not poly_i.touches(poly_j) and not poly_i.intersects(poly_j):
                continue

            # Compute intersection line
            intersection = compute_plane_intersection(
                plane_i, plane_j,
                (x_min, x_max),
                (y_min, y_max)
            )

            if intersection is None:
                continue

            # Clip to polygon intersection area
            combined = poly_i.union(poly_j)
            try:
                clipped = intersection.intersection(combined.buffer(0.5))
                if clipped.is_empty or not isinstance(clipped, LineString):
                    continue
            except Exception:
                clipped = intersection

            # Classify edge
            edge_type = classify_intersection_edge(plane_i, plane_j, clipped)
            dihedral = compute_dihedral_angle(plane_i, plane_j)

            edge = RoofEdge(
                edge_id=edge_counter,
                edge_type=edge_type,
                geometry=clipped,
                length_m=clipped.length,
                facet_ids=(facet_id_i, facet_id_j),
                dihedral_angle_deg=dihedral
            )
            all_edges.append(edge)
            edge_counter += 1

    # Add boundary edges for each facet
    for facet_id, polygon in facet_polygons:
        boundary_edges = extract_boundary_edges(polygon, facet_id)
        for edge in boundary_edges:
            edge.edge_id = edge_counter
            all_edges.append(edge)
            edge_counter += 1

    logger.info(
        f"Edge classification: {len(all_edges)} edges "
        f"({sum(1 for e in all_edges if e.edge_type == EdgeType.RIDGE)} ridges, "
        f"{sum(1 for e in all_edges if e.edge_type == EdgeType.HIP)} hips, "
        f"{sum(1 for e in all_edges if e.edge_type == EdgeType.VALLEY)} valleys)"
    )

    return all_edges


def compute_edge_lengths_by_type(edges: List[RoofEdge]) -> Dict[str, float]:
    """
    Compute total length by edge type.

    Reference: CLAUDE.md §4.1

    Args:
        edges: List of RoofEdge objects

    Returns:
        Dict mapping edge type to total length in meters
    """
    lengths = {}

    for edge in edges:
        type_name = edge.edge_type.value
        if type_name not in lengths:
            lengths[type_name] = 0.0
        lengths[type_name] += edge.length_m

    # Round values
    return {k: round(v, 2) for k, v in lengths.items()}


def edges_to_geojson(edges: List[RoofEdge]) -> dict:
    """
    Export edges as GeoJSON FeatureCollection.

    Args:
        edges: List of RoofEdge objects

    Returns:
        GeoJSON dict
    """
    features = []

    for edge in edges:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": list(edge.geometry.coords)
            },
            "properties": {
                "edge_id": edge.edge_id,
                "edge_type": edge.edge_type.value,
                "length_m": round(edge.length_m, 2),
                "facet_ids": list(edge.facet_ids),
                "dihedral_angle_deg": edge.dihedral_angle_deg
            }
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }


# ---------------------------------------------------------------------------
# Geometric edge typing for the RGB path (ported from adresses cell38_p7v1).
#
# Interior ridge/hip/valley come from intersections of CO-PLANAR-MERGED facets
# (adjacent facets whose plane normals agree < 8 deg and vertical gap < 0.3 m
# are unioned first, so same-plane fragments stop emitting fake hip seams).
# Perimeter eave/rake come from the CLEAN roof OUTLINE, simplified into straight
# sides and classified by the adjacent facet's downslope direction.
# ---------------------------------------------------------------------------

def _plane_normal_angle(p1: PlaneModel, p2: PlaneModel) -> float:
    """Angle (deg) between two plane normals."""
    n1 = np.array(p1.normal)
    n2 = np.array(p2.normal)
    return math.degrees(math.acos(float(np.clip(np.dot(n1, n2), -1.0, 1.0))))


def _are_coplanar(f1: dict, f2: dict) -> bool:
    """True if two facet dicts {poly, plane} are the same plane (angle + z-gap)."""
    if _plane_normal_angle(f1["plane"], f2["plane"]) >= COPLANAR_ANGLE_TOL_DEG:
        return False
    mx = (f1["poly"].centroid.x + f2["poly"].centroid.x) / 2
    my = (f1["poly"].centroid.y + f2["poly"].centroid.y) / 2
    z1 = f1["plane"].predict(np.array([mx]), np.array([my]))[0]
    z2 = f2["plane"].predict(np.array([mx]), np.array([my]))[0]
    return abs(z1 - z2) < COPLANAR_ZGAP_M


def merge_coplanar_facets(facets: List[dict]) -> List[dict]:
    """Union-find merge of adjacent + co-planar facets.

    Args:
        facets: list of {"facet_id", "poly": Polygon, "plane": PlaneModel}.

    Returns:
        merged list of {"facet_id", "poly", "plane"} with area-weighted planes.
    """
    n = len(facets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _are_coplanar(facets[i], facets[j]) and \
                    facets[i]["poly"].buffer(0.15).intersects(facets[j]["poly"]):
                parent[find(i)] = find(j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: List[dict] = []
    for k, members in enumerate(groups.values()):
        polys = [facets[m]["poly"] for m in members]
        u = unary_union(polys) if len(polys) > 1 else polys[0]
        if isinstance(u, MultiPolygon):
            u = max(u.geoms, key=lambda g: g.area)
        if not u.is_valid:
            u = u.buffer(0)
        areas = np.array([facets[m]["poly"].area for m in members], dtype=float)
        w = areas / areas.sum() if areas.sum() > 0 else np.ones(len(members)) / len(members)
        plane = PlaneModel(
            a=float(sum(w[t] * facets[m]["plane"].a for t, m in enumerate(members))),
            b=float(sum(w[t] * facets[m]["plane"].b for t, m in enumerate(members))),
            c=float(sum(w[t] * facets[m]["plane"].c for t, m in enumerate(members))),
            inlier_mask=np.zeros(0, dtype=bool), inlier_count=0,
            residual_median=0.0, success=True,
        )
        merged.append({"facet_id": k, "poly": u, "plane": plane})
    return merged


def _line_components(geom) -> List[List[tuple]]:
    """Flatten a (Multi)LineString / GeometryCollection into coord lists."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [list(geom.coords)]
    if isinstance(geom, (MultiLineString, GeometryCollection)):
        out = []
        for g in geom.geoms:
            out.extend(_line_components(g))
        return out
    return []


def classify_interior_edge(
    plane_i: PlaneModel, plane_j: PlaneModel, line: LineString,
    centroid_i: tuple, centroid_j: tuple,
) -> EdgeType:
    """Classify a shared interior edge as ridge / valley / hip.

    Intercept-independent (the synthesized planes have arbitrary `c`). Two cues:
      1. Perpendicular: does each facet DESCEND moving from the line into it?
           both descend -> ridge-or-hip;  both ascend -> valley;  mixed -> hip.
      2. Along the line: a ridge line is LEVEL (no fall along its length) while a
         hip line SLOPES (apex -> eave). So a "both-descend" line that runs
         downhill is a hip, not a ridge.
    """
    mid = line.interpolate(0.5, normalized=True)
    mx, my = mid.x, mid.y
    coords = list(line.coords)
    dx, dy = coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1]
    L = math.hypot(dx, dy) or 1.0
    px, py = -dy / L, dx / L                        # unit perpendicular to the line
    lx, ly = dx / L, dy / L                         # unit along the line

    def slope_into(plane, centroid):
        # +1/-1 = which side of the line this facet sits on
        s = 1.0 if ((centroid[0] - mx) * px + (centroid[1] - my) * py) >= 0 else -1.0
        # rate of height change moving from the line into the facet
        return s * (plane.a * px + plane.b * py)

    into_i = slope_into(plane_i, centroid_i)
    into_j = slope_into(plane_j, centroid_j)

    # ridge vs hip: angle between the two facets' downslope gradients. Opposite
    # (~180deg, cos ~ -1) => ridge; perpendicular (~90deg, cos ~ 0) => hip.
    # Robust to the approximate snapped line direction and to binned aspects.
    gi = math.hypot(plane_i.a, plane_i.b)
    gj = math.hypot(plane_j.a, plane_j.b)
    cos_grad = ((plane_i.a * plane_j.a + plane_i.b * plane_j.b) / (gi * gj)
                if gi > 1e-9 and gj > 1e-9 else 0.0)
    facets_oppose = cos_grad < RIDGE_OPPOSE_COS_MAX

    tol = 1e-9
    if into_i < -tol and into_j < -tol:
        return EdgeType.RIDGE if facets_oppose else EdgeType.HIP
    if into_i > tol and into_j > tol:
        return EdgeType.VALLEY
    return EdgeType.HIP


def classify_perimeter_edge(plane: PlaneModel, p0: tuple, p1: tuple) -> EdgeType:
    """Eave vs rake for a perimeter segment, from the adjacent facet's slope.

    Eave = the down-slope direction is ~perpendicular to the edge (contour /
    bottom edge). Rake = down-slope runs ~parallel to the edge (gable end).
    A near-flat facet defaults to eave.
    """
    g = math.hypot(plane.a, plane.b)
    if g < 1e-6:
        return EdgeType.EAVE
    dsx, dsy = -plane.a / g, -plane.b / g          # downslope unit vector
    ex, ey = p1[0] - p0[0], p1[1] - p0[1]
    el = math.hypot(ex, ey)
    if el < 1e-9:
        return EdgeType.EAVE
    ex, ey = ex / el, ey / el
    parallelism = abs(dsx * ex + dsy * ey)         # 0 = perpendicular, 1 = parallel
    return EdgeType.RAKE if parallelism >= 0.5 else EdgeType.EAVE


def _adjacent_facet(merged: List[dict], outline: Polygon, p0: tuple, p1: tuple) -> Optional[dict]:
    """Find the merged facet just inside an outline segment (for eave/rake)."""
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy) or 1.0
    px, py = -dy / L, dx / L
    for s in (0.8, -0.8, 1.6, -1.6):
        cand = Point(mx + s * px, my + s * py)
        if outline.contains(cand):
            for f in merged:
                if f["poly"].buffer(0.3).contains(cand):
                    return f
            if merged:
                return min(merged, key=lambda f: (f["poly"].centroid.x - cand.x) ** 2
                           + (f["poly"].centroid.y - cand.y) ** 2)
            return None
    return None


def classify_edges_from_facets(
    facet_polygons: List[Tuple[int, Polygon]],
    planes: List[PlaneModel],
    outline: Optional[Polygon] = None,
) -> List[RoofEdge]:
    """Derive typed roof edges from facet polygons + planes (RGB path).

    Args:
        facet_polygons: list of (facet_id, Polygon) in a planar metric CRS.
        planes: PlaneModel per facet (same order); synthesize via
            plane_fit.plane_from_pitch_aspect when only pitch/aspect are known.
        outline: clean roof outline polygon for perimeter eave/rake. If None,
            perimeter edges are taken from the merged-facet boundary instead.

    Returns:
        List[RoofEdge] (ridge/hip/valley interior, eave/rake perimeter).
    """
    facets = [{"facet_id": fid, "poly": poly, "plane": planes[i]}
              for i, (fid, poly) in enumerate(facet_polygons) if poly is not None]
    merged = merge_coplanar_facets(facets)
    edges: List[RoofEdge] = []
    eid = 0

    # INTERIOR ridge/hip/valley from merged-facet intersections
    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            fi, fj = merged[i], merged[j]
            shared = fi["poly"].boundary.intersection(fj["poly"].boundary)
            for comp in _line_components(shared):
                if len(comp) < 2:
                    continue
                line = LineString(comp)
                if line.length < MIN_EDGE_LEN_M:
                    continue
                ci = (fi["poly"].centroid.x, fi["poly"].centroid.y)
                cj = (fj["poly"].centroid.x, fj["poly"].centroid.y)
                etype = classify_interior_edge(fi["plane"], fj["plane"], line, ci, cj)
                dihedral = compute_dihedral_angle(fi["plane"], fj["plane"])
                edges.append(RoofEdge(eid, etype, line, line.length,
                                      (fi["facet_id"], fj["facet_id"]), dihedral))
                eid += 1

    # PERIMETER eave/rake from the clean outline (fallback: merged boundary)
    if outline is not None and not outline.is_empty:
        coords = list(outline.simplify(OUTLINE_SIMPLIFY_M).exterior.coords)
        for k in range(len(coords) - 1):
            p0, p1 = coords[k], coords[k + 1]
            seg = LineString([p0, p1])
            if seg.length < MIN_EDGE_LEN_M:
                continue
            fac = _adjacent_facet(merged, outline, p0, p1)
            etype = classify_perimeter_edge(fac["plane"], p0, p1) if fac else EdgeType.EAVE
            edges.append(RoofEdge(eid, etype, seg, seg.length,
                                  (fac["facet_id"] if fac else None, None)))
            eid += 1
    else:
        for fi in merged:
            others = [fj["poly"].boundary for fj in merged if fj["facet_id"] != fi["facet_id"]]
            boundary = fi["poly"].boundary
            lone = boundary.difference(unary_union(others).buffer(0.05)) if others else boundary
            for comp in _line_components(lone):
                if len(comp) < 2:
                    continue
                seg = LineString(comp)
                if seg.length < MIN_EDGE_LEN_M:
                    continue
                etype = classify_perimeter_edge(fi["plane"], comp[0], comp[-1])
                edges.append(RoofEdge(eid, etype, seg, seg.length, (fi["facet_id"], None)))
                eid += 1

    logger.info(
        "Geometric edges: %d total (%d ridge, %d hip, %d valley, %d eave, %d rake)",
        len(edges),
        sum(1 for e in edges if e.edge_type == EdgeType.RIDGE),
        sum(1 for e in edges if e.edge_type == EdgeType.HIP),
        sum(1 for e in edges if e.edge_type == EdgeType.VALLEY),
        sum(1 for e in edges if e.edge_type == EdgeType.EAVE),
        sum(1 for e in edges if e.edge_type == EdgeType.RAKE),
    )
    return edges
