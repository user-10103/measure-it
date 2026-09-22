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
RAKE_PARALLELISM_THRESH = 0.75  # |cos(angle)| >= this -> rake; < this -> eave.
                               # 0.75 ~ 41 deg from parallel. Eave: 0-41 deg,
                               # rake: 41-90 deg to the downslope direction.
# ridge vs hip (among convex edges): a ridge joins facets facing ~OPPOSITE ways
# (downslopes ~180deg apart -> cos ~ -1); a hip joins facets ~90deg apart
# (cos ~ 0). cos(angle between gradients) below this => ridge. This is robust to
# the approximate snapped line direction and tolerates 45deg-binned aspects,
# unlike the old absolute along-line cutoff that misread ridges as hips.
RIDGE_OPPOSE_COS_MAX = -0.5    # gradients > 120deg apart => ridge


class EdgeType(Enum):
    """Roof edge classification types."""
    RIDGE = "ridge"               # Both planes slope downward away from line
    HIP = "hip"                   # Planes slope downward away from each other
    VALLEY = "valley"             # Planes slope downward towards line
    EAVE = "eave"                 # Lower perimeter edge (facet drains outward)
    RAKE = "rake"                 # Gable-end perimeter edge (slope-parallel)
    STEP_FLASHING = "step_flashing"  # Sloped facet's HIGH-side perimeter edge (meets wall)
    WALL_FLASHING = "wall_flashing"  # Roof meets a taller wall head-on (height step at junction)
    PARAPET = "parapet"           # Flat facet perimeter edge (terminates at raised wall)
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


def merge_collinear_edges(
    edges: "List[RoofEdge]",
    angle_tol_deg: float = 10.0,
    snap_tol_m: float = 0.3,
) -> "List[RoofEdge]":
    """Merge same-type edges that form a continuous physical run.

    The tiling step produces one RoofEdge per shared polygon boundary segment;
    a single physical ridge becomes several short entries.  The previous
    sequential-scan approach only merged *consecutive* same-type entries in list
    order, which missed segments of the same physical run that were interleaved
    with edges of other types.

    This replacement uses a proximity graph:
      1. Group edges by type.
      2. Within each type, build an undirected graph: two segments are linked
         if (a) an endpoint of one is within *snap_tol_m* of an endpoint of the
         other AND (b) their bearing difference is <= angle_tol_deg.
      3. Merge each connected component with shapely.ops.linemerge.

    Length totals are invariant (sum before == sum after).
    """
    import math
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge

    if not edges:
        return edges

    def _bearing(line: LineString) -> float:
        coords = list(line.coords)
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]
        return math.degrees(math.atan2(dx, dy)) % 180  # undirected bearing

    def _angle_diff(a: float, b: float) -> float:
        d = abs(a - b) % 180
        return min(d, 180 - d)

    def _endpoints(line: LineString):
        coords = list(line.coords)
        return (coords[0][0], coords[0][1]), (coords[-1][0], coords[-1][1])

    def _pt_dist(p, q):
        return math.hypot(p[0] - q[0], p[1] - q[1])

    # Union-Find for grouping
    parent = list(range(len(edges)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    # Pre-compute bearings and endpoints once
    bearings = [_bearing(e.geometry) for e in edges]
    endpoints = [_endpoints(e.geometry) for e in edges]

    # For each pair of same-type edges, check proximity + bearing
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if edges[i].edge_type != edges[j].edge_type:
                continue
            if _angle_diff(bearings[i], bearings[j]) > angle_tol_deg:
                continue
            # Check if any endpoints are close
            ep_i = endpoints[i]
            ep_j = endpoints[j]
            close = False
            for pi in ep_i:
                for pj in ep_j:
                    if _pt_dist(pi, pj) <= snap_tol_m:
                        close = True
                        break
                if close:
                    break
            if close:
                union(i, j)

    # Group by component
    from collections import defaultdict
    components = defaultdict(list)
    for i, edge in enumerate(edges):
        components[find(i)].append(edge)

    merged = []
    for group in components.values():
        if len(group) == 1:
            merged.append(group[0])
            continue
        # Merge geometries
        multi = MultiLineString([e.geometry for e in group])
        merged_geom = linemerge(multi)
        if merged_geom.geom_type != "LineString":
            # linemerge couldn't join (disconnected fragments) — take longest
            if merged_geom.geom_type == "MultiLineString":
                merged_geom = max(merged_geom.geoms, key=lambda g: g.length)
            else:
                merged_geom = group[0].geometry
        total_len = sum(e.length_m for e in group)
        rep = group[0]
        all_facet_ids = set()
        for e in group:
            all_facet_ids.update(e.facet_ids or ())
        merged.append(RoofEdge(
            edge_id=rep.edge_id,
            edge_type=rep.edge_type,
            geometry=merged_geom,
            length_m=round(total_len, 2),
            facet_ids=tuple(sorted(x for x in all_facet_ids if x is not None)),
            dihedral_angle_deg=rep.dihedral_angle_deg,
        ))

    return merged


def correct_edge_lengths_3d(
    edges: List["RoofEdge"],
    plane_by_facet_id: Dict[int, "PlaneModel"],
) -> None:
    """Correct hip/rake/valley lengths from plan-projection to true 3D slope distance.

    Ridges and eaves run horizontally along contour lines — no correction needed.
    Hips, rakes, and valleys lie on sloped surfaces; their true length is:
        length_3d = length_2d × √(1 + (a·dx + b·dy)²)
    where (a, b) is the gradient of the adjacent plane and (dx, dy) is the edge's
    unit direction in plan. Modifies edge.length_m in-place.
    """
    NEEDS_CORRECTION = {EdgeType.HIP, EdgeType.RAKE, EdgeType.VALLEY}
    for edge in edges:
        if edge.edge_type not in NEEDS_CORRECTION:
            continue
        plane = None
        for fid in edge.facet_ids:
            if fid is not None and fid in plane_by_facet_id:
                plane = plane_by_facet_id[fid]
                break
        if plane is None:
            continue
        coords = list(edge.geometry.coords)
        if len(coords) < 2:
            continue
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]
        plan_len = math.hypot(dx, dy)
        if plan_len < 1e-9:
            continue
        dx /= plan_len
        dy /= plan_len
        dz_per_unit = plane.a * dx + plane.b * dy
        edge.length_m = round(edge.length_m * math.sqrt(1.0 + dz_per_unit ** 2), 4)


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


STEP_FLASHING_INWARD_THRESH = 0.35  # downslope·inward > this → edge is on the HIGH side

def classify_perimeter_edge(
    plane: PlaneModel,
    p0: tuple,
    p1: tuple,
    interior_point: Optional[tuple] = None,
) -> EdgeType:
    """Classify a perimeter segment as eave, rake, step_flashing, or parapet.

    Rules (applied in priority order):
      - Flat plane (gradient ≈ 0)    → PARAPET (raised wall at edge of flat section)
      - Slope ∥ edge direction        → RAKE (gable end)
      - Downslope toward interior     → STEP_FLASHING (edge is on facet's high side)
      - Otherwise                     → EAVE (edge is on facet's low side)

    `interior_point` is any point inside the roof (e.g. the adjacent facet's centroid).
    Without it, step flashing cannot be distinguished from eave.
    """
    g = math.hypot(plane.a, plane.b)
    if g < math.tan(math.radians(5.0)):
        # Flat facet perimeter is a parapet wall. Threshold matches the metrics
        # flat snap (2.5 deg) — fitted membrane-roof planes are never exactly
        # gradient-zero, so an epsilon test would classify them as eaves.
        return EdgeType.PARAPET

    dsx, dsy = -plane.a / g, -plane.b / g   # unit downslope vector
    ex, ey = p1[0] - p0[0], p1[1] - p0[1]
    el = math.hypot(ex, ey)
    if el < 1e-9:
        return EdgeType.EAVE
    ex, ey = ex / el, ey / el

    parallelism = abs(dsx * ex + dsy * ey)  # 0 = perp to edge, 1 = parallel
    if parallelism >= RAKE_PARALLELISM_THRESH:
        return EdgeType.RAKE

    if interior_point is not None:
        mx = (p0[0] + p1[0]) / 2
        my = (p0[1] + p1[1]) / 2
        ix = interior_point[0] - mx
        iy = interior_point[1] - my
        il = math.hypot(ix, iy)
        if il > 1e-9:
            # Positive → downslope is toward the roof interior → edge is on the HIGH
            # side (facet drains away from this perimeter segment toward the interior).
            # That geometry matches step flashing: a sloped roof plane meeting a wall
            # at its upper edge.
            if (dsx * ix / il + dsy * iy / il) > STEP_FLASHING_INWARD_THRESH:
                return EdgeType.STEP_FLASHING

    return EdgeType.EAVE


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


def _classify_step_junction(
    plane_i: PlaneModel,
    plane_j: PlaneModel,
    line: LineString,
    step_height_m: float,
) -> Optional[EdgeType]:
    """Detect a roof-meets-wall junction between two facets.

    Evaluates both planes at the shared edge's midpoint. A height gap larger
    than `step_height_m` means the facets are at different stories — the edge
    is flashing on the LOWER roof, not a ridge/hip/valley:
      - lower facet flat (near-zero gradient)      → WALL_FLASHING
      - lower facet pitched, edge ∥ its downslope  → STEP_FLASHING
      - lower facet pitched, edge ⊥ its downslope  → WALL_FLASHING

    Returns None when the planes meet within tolerance (normal interior edge).
    """
    mid = line.interpolate(0.5, normalized=True)
    mx, my = mid.x, mid.y
    z_i = plane_i.predict(np.array([mx]), np.array([my]))[0]
    z_j = plane_j.predict(np.array([mx]), np.array([my]))[0]

    lower = plane_i if z_i < z_j else plane_j
    g = math.hypot(lower.a, lower.b)
    lower_is_flat = g < math.tan(math.radians(5.0))

    # A flat membrane meeting a pitched section is NEVER a ridge/hip/valley —
    # any real gap there is roof-meets-wall. Use a tighter trigger for that
    # case; pitched-pitched junctions keep the caller's threshold so ridge
    # fits that disagree by a few decimeters aren't misread as steps.
    trigger = min(0.3, step_height_m) if lower_is_flat else step_height_m
    if abs(z_i - z_j) <= trigger:
        if lower_is_flat:
            # Flush flat-pitched junction: the pitched side drains onto the
            # membrane — that is its eave, never a ridge/hip/valley.
            return EdgeType.EAVE
        return None

    if lower_is_flat:
        return EdgeType.WALL_FLASHING

    coords = list(line.coords)
    ex, ey = coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1]
    el = math.hypot(ex, ey)
    if el < 1e-9:
        return EdgeType.WALL_FLASHING
    parallelism = abs((-lower.a / g) * (ex / el) + (-lower.b / g) * (ey / el))
    if parallelism >= RAKE_PARALLELISM_THRESH:
        return EdgeType.STEP_FLASHING       # edge runs up the lower roof's slope
    return EdgeType.WALL_FLASHING           # edge runs across the lower roof's slope


def _split_at_corners(coords, angle_thresh_deg: float = 25.0):
    """Split a polyline into maximal straight pieces, cutting at any vertex
    whose turn angle exceeds angle_thresh_deg. Keeps an eave and a rake that
    share a facet from being merged into one mis-typed perimeter edge."""
    if len(coords) <= 2:
        return [list(coords)]
    pieces, cur = [], [coords[0]]
    for i in range(1, len(coords) - 1):
        cur.append(coords[i])
        ax, ay = coords[i][0] - coords[i - 1][0], coords[i][1] - coords[i - 1][1]
        bx, by = coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1]
        la, lb = math.hypot(ax, ay) or 1.0, math.hypot(bx, by) or 1.0
        cosang = max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb)))
        if math.degrees(math.acos(cosang)) > angle_thresh_deg:
            pieces.append(cur)
            cur = [coords[i]]
    cur.append(coords[-1])
    pieces.append(cur)
    return pieces


def classify_edges_from_facets(
    facet_polygons: List[Tuple[int, Polygon]],
    planes: List[PlaneModel],
    outline: Optional[Polygon] = None,
    step_height_m: Optional[float] = None,
) -> List[RoofEdge]:
    """Derive typed roof edges from facet polygons + planes (RGB path).

    Args:
        facet_polygons: list of (facet_id, Polygon) in a planar metric CRS.
        planes: PlaneModel per facet (same order); synthesize via
            plane_fit.plane_from_pitch_aspect when only pitch/aspect are known.
        outline: clean roof outline polygon for perimeter eave/rake. If None,
            perimeter edges are taken from the merged-facet boundary instead.
        step_height_m: if set, interior junctions where the two planes' heights
            differ by more than this are classified as wall/step flashing (roof
            meets a taller wall) instead of ridge/hip/valley. Requires planes
            with REAL intercepts (LiDAR fits) — leave None for the RGB path,
            whose synthesized planes have arbitrary `c`.

    Returns:
        List[RoofEdge] (ridge/hip/valley/flashing interior, eave/rake perimeter).
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
                etype = None
                if step_height_m is not None:
                    etype = _classify_step_junction(
                        fi["plane"], fj["plane"], line, step_height_m
                    )
                dihedral = compute_dihedral_angle(fi["plane"], fj["plane"])
                if etype is None:
                    _aspect_diff = abs(
                        (math.degrees(math.atan2(-fi["plane"].b, -fi["plane"].a))
                         - math.degrees(math.atan2(-fj["plane"].b, -fj["plane"].a)) + 180.0)
                        % 360.0 - 180.0
                    )
                    if dihedral < 12.0 and _aspect_diff < 90.0:
                        # Near-coplanar junction draining the SAME way with no
                        # height step: fragments of one physical plane (adjacent
                        # aspect bins at 3:12 differ ~10.6° dihedral). Not a
                        # structural edge — emitting it fakes valleys/hips along
                        # mid-slope fragment seams. Opposite-draining pairs are
                        # kept even at low dihedral: a shallow-fitted ridge is
                        # still a ridge.
                        continue
                    etype = classify_interior_edge(fi["plane"], fj["plane"], line, ci, cj)
                    if etype == EdgeType.VALLEY and step_height_m is not None:
                        # Height sanity check (real LiDAR intercepts): a valley
                        # line sits BELOW both facets' interiors. Gradient-sign
                        # noise on shallow fits fakes valleys at ridge seams.
                        _mid = line.interpolate(0.5, normalized=True)
                        _zl = float(np.mean([
                            fi["plane"].predict(np.array([_mid.x]), np.array([_mid.y]))[0],
                            fj["plane"].predict(np.array([_mid.x]), np.array([_mid.y]))[0],
                        ]))
                        _zi = fi["plane"].predict(np.array([ci[0]]), np.array([ci[1]]))[0]
                        _zj = fj["plane"].predict(np.array([cj[0]]), np.array([cj[1]]))[0]
                        if _zl >= min(_zi, _zj) - 0.05:
                            # line is not below both interiors -> ridge family
                            _gi = math.hypot(fi["plane"].a, fi["plane"].b)
                            _gj = math.hypot(fj["plane"].a, fj["plane"].b)
                            _cosg = ((fi["plane"].a * fj["plane"].a
                                      + fi["plane"].b * fj["plane"].b)
                                     / max(_gi * _gj, 1e-12))
                            etype = EdgeType.RIDGE if _cosg < -0.5 else EdgeType.HIP
                edges.append(RoofEdge(eid, etype, line, line.length,
                                      (fi["facet_id"], fj["facet_id"]), dihedral))
                eid += 1
                if etype in (EdgeType.WALL_FLASHING, EdgeType.STEP_FLASHING):
                    # A story step is TWO edges in roofing terms: flashing on
                    # the lower roof AND the upper roof's own perimeter edge
                    # (its eave/rake — water drains off it onto the lower roof).
                    mid = line.interpolate(0.5, normalized=True)
                    z_i = fi["plane"].predict(np.array([mid.x]), np.array([mid.y]))[0]
                    z_j = fj["plane"].predict(np.array([mid.x]), np.array([mid.y]))[0]
                    upper = fi if z_i >= z_j else fj
                    coords2 = list(line.coords)
                    up_type = classify_perimeter_edge(
                        upper["plane"], coords2[0], coords2[-1],
                        (upper["poly"].centroid.x, upper["poly"].centroid.y),
                    )
                    if up_type in (EdgeType.EAVE, EdgeType.RAKE):
                        edges.append(RoofEdge(eid, up_type, line, line.length,
                                              (upper["facet_id"], None)))
                        eid += 1

    # PERIMETER: intersect each merged facet's boundary with the outline directly.
    # This correctly handles outlines that span multiple facet types (e.g. a 30m
    # edge that is partly hip-eave and partly flat-parapet) without relying on a
    # midpoint probe that can misidentify the adjacent facet on long segments.
    if outline is not None and not outline.is_empty:
        # Walk the outline and probe INWARD to find each segment's owner facet.
        # Cell geometry at the rim is unreliable (absorbed unmodeled strips,
        # 1 m LiDAR noise at parapet lips), so intersecting cell boundaries
        # with the outline misattributes the perimeter; a 1.5 m inward probe
        # lands past the noisy strip in the facet that actually owns the rim.
        from shapely.geometry import Point as _Pt
        from shapely.ops import substring as _substring

        ring = outline.exterior
        L = ring.length
        n = max(int(L / 0.5), 16)

        def _owner_at(dist):
            p = ring.interpolate(dist)
            p2 = ring.interpolate(min(dist + 0.25, L))
            tx, ty = p2.x - p.x, p2.y - p.y
            tl = math.hypot(tx, ty) or 1.0
            nx, ny = -ty / tl, tx / tl
            for probe_m in (1.5, 0.75):
                for sgn in (1.0, -1.0):
                    q = _Pt(p.x + sgn * nx * probe_m, p.y + sgn * ny * probe_m)
                    if outline.contains(q):
                        for f in merged:
                            if f["poly"].contains(q):
                                return f
                        break   # inside outline but in no cell: try shorter probe
            return None

        samples = [(k * L / n, _owner_at((k + 0.5) * L / n)) for k in range(n)]
        # group consecutive same-owner samples into outline runs
        runs = []
        for k, (d, owner) in enumerate(samples):
            if runs and runs[-1][2] is owner:
                runs[-1][1] = (k + 1) * L / n
            else:
                runs.append([d, (k + 1) * L / n, owner])
        # a run cut in two by the ring start/end wraps around — merge ends
        if len(runs) > 1 and runs[0][2] is runs[-1][2]:
            runs[0][0] = runs[-1][0] - L
            runs.pop()

        for d0, d1, owner in runs:
            if owner is None:
                continue
            if d0 >= 0:
                seg = _substring(ring, d0, d1)
            else:        # wrapped run: stitch the two outline pieces
                part_a = _substring(ring, d0 + L, L)
                part_b = _substring(ring, 0, d1)
                coords = list(part_a.coords) + list(part_b.coords)[1:]
                seg = LineString(coords) if len(coords) >= 2 else None
            if seg is None or seg.length < MIN_EDGE_LEN_M:
                continue
            interior_pt = (owner["poly"].centroid.x, owner["poly"].centroid.y)
            # An owner-run can turn a corner (e.g. an eave meeting a rake at a
            # gable end, both owned by the same facet). Classifying the whole
            # run by its two endpoints loses the rake, so split the run into
            # straight pieces at corners and type each piece on its own.
            for piece in _split_at_corners(list(seg.coords)):
                pseg = LineString(piece)
                if pseg.length < MIN_EDGE_LEN_M:
                    continue
                etype = classify_perimeter_edge(
                    owner["plane"], piece[0], piece[-1], interior_pt)
                if etype == EdgeType.STEP_FLASHING:
                    # This pass walks the BUILDING outline — nothing is taller
                    # beyond it, so a high-side run here is a drip edge that
                    # report conventions count as eave. True step flashing
                    # arises at interior story steps (step-junction pass).
                    etype = EdgeType.EAVE
                edges.append(RoofEdge(eid, etype, pseg, pseg.length,
                                      (owner["facet_id"], None)))
                eid += 1
            logger.debug(
                "perimeter run: facet %s slope=%.1f° %.1f m -> %s",
                owner["facet_id"],
                math.degrees(math.atan(math.hypot(owner["plane"].a, owner["plane"].b))),
                seg.length, etype.value,
            )
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
                interior_pt = (fi["poly"].centroid.x, fi["poly"].centroid.y)
                etype = classify_perimeter_edge(fi["plane"], comp[0], comp[-1], interior_pt)
                edges.append(RoofEdge(eid, etype, seg, seg.length, (fi["facet_id"], None)))
                eid += 1

    correct_edge_lengths_3d(edges, {f["facet_id"]: f["plane"] for f in merged})

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
