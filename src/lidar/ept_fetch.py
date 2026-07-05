"""Pure-Python EPT point fetch — no PDAL, pip-only deps (requests + laspy).

EPT (Entwine Point Tiles) is just static files over HTTPS: an ``ept.json``
header, a JSON octree hierarchy, and LAZ tiles per octree node. The USGS
archive (``usgs-lidar-public``) serves them publicly with no credentials. So
the roof-point fetch the LiDAR fusion needs is: read the header, walk the
hierarchy for nodes intersecting the footprint, download + decode those LAZ
tiles, clip to the footprint, reproject to the chip CRS.

This exists because the original ``ept_client.fetch_lidar_points`` requires
PDAL — a conda-first system dependency deliberately absent from the service
container. This module needs only wheels.

Entry point: ``fetch_roof_points(lat, lon, footprint_wgs84, target_crs)`` ->
(N, 3) float array in ``target_crs`` (the NAIP chip CRS), ready for
``fuse_sam_lidar.annotate_facets_with_lidar``.
"""
from __future__ import annotations

import io
import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MAX_NODES = 400          # octree tiles per query — far above a house's need
VEG_GROUND_CLASSES = {2, 3, 4, 5, 7, 9}   # ground / vegetation / noise / water


def _get(url: str, timeout: int = 60):
    """Single HTTP entry point (patchable in tests)."""
    import requests
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r


def _node_bbox(key: str, root: Tuple[float, float, float, float, float, float]):
    """EPT node key 'D-X-Y-Z' -> (minx, miny, maxx, maxy) in the cubic bounds."""
    d, x, y, _z = (int(p) for p in key.split("-"))
    size_x = (root[3] - root[0]) / (1 << d)
    size_y = (root[4] - root[1]) / (1 << d)
    return (root[0] + x * size_x, root[1] + y * size_y,
            root[0] + (x + 1) * size_x, root[1] + (y + 1) * size_y)


def _walk_hierarchy(base_url: str, root_bounds, qbox,
                    max_nodes: int = MAX_NODES) -> List[str]:
    """Collect data-node keys whose box intersects the query box."""
    minx, miny, maxx, maxy = qbox
    keys: List[str] = []
    pending = ["0-0-0-0"]           # hierarchy files to load
    while pending and len(keys) < max_nodes:
        h = _get(f"{base_url}/ept-hierarchy/{pending.pop()}.json").json()
        for key, count in h.items():
            if count == 0:
                continue
            bx0, by0, bx1, by1 = _node_bbox(key, root_bounds)
            if bx1 < minx or bx0 > maxx or by1 < miny or by0 > maxy:
                continue
            if count == -1:          # subtree in its own hierarchy file
                pending.append(key)
            else:
                keys.append(key)
                if len(keys) >= max_nodes:
                    logger.warning("EPT node cap (%d) hit — truncating", max_nodes)
                    break
    return keys


def fetch_roof_points(
    lat: float,
    lon: float,
    footprint_wgs84,
    target_crs: str,
    ept_url: Optional[str] = None,
    buffer_m: float = 3.0,
    max_nodes: int = MAX_NODES,
    with_ground: bool = False,
):
    """Roof point cloud for a building -> (N, 3) xyz in ``target_crs``.

    Args:
        footprint_wgs84: shapely Polygon of the building (EPSG:4326).
        target_crs: CRS the SAM facets live in (the NAIP chip CRS).
        ept_url: EPT root URL; discovered via the entwine index when None.
        with_ground: also return the median GROUND elevation (class-2 points
            near the building) -> returns (points, ground_z | None). Enables
            building-height / two-story detection downstream.

    Returns None when there is no coverage (callers degrade to the
    imagery-only report — pitch "unspecified").
    """
    def _nothing():
        return (None, None) if with_ground else None
    from pyproj import CRS, Transformer
    from shapely import contains_xy
    from shapely.ops import transform as shp_transform

    if ept_url is None:
        from src.lidar.dataset_discovery import discover_ept_from_entwine
        ept_url = discover_ept_from_entwine(lat, lon)
        if ept_url is None:
            logger.info("no EPT LiDAR coverage at (%.5f, %.5f)", lat, lon)
            return _nothing()
    base = ept_url[: ept_url.rfind("/")] if ept_url.endswith("ept.json") else ept_url.rstrip("/")

    header = _get(f"{base}/ept.json").json()
    if header.get("dataType") not in (None, "laszip", "binary"):
        logger.warning("unsupported EPT dataType %r", header.get("dataType"))
        return _nothing()
    srs = header.get("srs", {})
    if srs.get("authority") == "EPSG" and srs.get("horizontal"):
        crs_ept = CRS.from_epsg(int(srs["horizontal"]))
    elif srs.get("wkt"):
        crs_ept = CRS.from_wkt(srs["wkt"])
    else:
        logger.warning("EPT header has no usable SRS")
        return _nothing()
    root_bounds = header["bounds"]                       # cubic

    # footprint -> EPT CRS; buffer in native units
    unit = crs_ept.axis_info[0].unit_conversion_factor or 1.0   # 1.0 = meters
    to_ept = Transformer.from_crs("EPSG:4326", crs_ept, always_xy=True).transform
    fp_ept = shp_transform(to_ept, footprint_wgs84).buffer(buffer_m / unit)
    qbox = fp_ept.bounds

    keys = _walk_hierarchy(base, root_bounds, qbox, max_nodes)
    if not keys:
        logger.info("EPT: no octree nodes intersect the footprint")
        return _nothing()
    logger.info("EPT: reading %d node(s) from %s", len(keys), base)

    import laspy
    xs, ys, zs, cls = [], [], [], []
    for key in keys:
        try:
            las = laspy.read(io.BytesIO(_get(f"{base}/ept-data/{key}.laz").content))
        except Exception as e:  # noqa: BLE001 — a missing tile shouldn't kill the roof
            logger.warning("EPT node %s unreadable (%s)", key, e)
            continue
        x, y = np.asarray(las.x, float), np.asarray(las.y, float)
        keep = ((x >= qbox[0]) & (x <= qbox[2]) &
                (y >= qbox[1]) & (y <= qbox[3]))
        if not keep.any():
            continue
        xs.append(x[keep]); ys.append(y[keep])
        zs.append(np.asarray(las.z, float)[keep])
        c = getattr(las, "classification", None)
        cls.append(np.asarray(c, int)[keep] if c is not None
                   else np.zeros(int(keep.sum()), int))
    if not xs:
        return _nothing()
    x = np.concatenate(xs); y = np.concatenate(ys)
    z = np.concatenate(zs); c = np.concatenate(cls)

    # GROUND elevation first (class 2 near the building) — building height
    ground_z = None
    if with_ground and (c == 2).any():
        gz = z[c == 2]
        ground_z = float(np.median(gz))

    # drop ground/vegetation/noise when the dataset is classified at all
    if (c > 0).any():
        keep = ~np.isin(c, list(VEG_GROUND_CLASSES))
        x, y, z = x[keep], y[keep], z[keep]

    # clip to the buffered footprint, then to the chip CRS (meters)
    inside = contains_xy(fp_ept, x, y)
    x, y, z = x[inside], y[inside], z[inside]
    if len(x) == 0:
        return _nothing()
    if unit != 1.0:                     # e.g. US-feet state plane: z follows xy
        z = z * unit
        if ground_z is not None:
            ground_z = ground_z * unit
        logger.info("EPT native units factor %.4f applied to z", unit)
    to_target = Transformer.from_crs(crs_ept, CRS.from_user_input(target_crs),
                                     always_xy=True).transform
    tx, ty = to_target(x, y)
    pts = np.column_stack([tx, ty, z])
    logger.info("EPT: %d roof point(s) for the footprint%s", len(pts),
                f", ground z={ground_z:.1f} m" if ground_z is not None else "")
    return (pts, ground_z) if with_ground else pts
