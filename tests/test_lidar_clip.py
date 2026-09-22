"""The LiDAR clip must cover the facets, not just the footprint."""
import types
from shapely.geometry import box
from src.serve.report_service import _lidar_clip_geometry

# a footprint in WGS84 and a detected outline 22 m away in a metric CRS (UTM 17N)
FP = box(-80.6470, 28.1215, -80.6465, 28.1219)

def _roof(outline, georef=True):
    return types.SimpleNamespace(outline=outline, georeferenced=georef)

def test_union_covers_an_outline_the_footprint_misses():
    # outline offset ~40 m east of the footprint in UTM metres
    meta = {"footprint_wgs84": FP, "crs": "EPSG:32617"}
    import pyproj
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
    x, y = to_utm.transform(-80.6467, 28.1217)
    outline = box(x + 40, y + 40, x + 70, y + 70)
    merged = _lidar_clip_geometry(meta, _roof(outline))
    assert merged.area > FP.area                    # widened
    assert merged.contains(FP.centroid)             # still covers the footprint
    # and now covers the outline region the old footprint-only clip missed
    to_wgs = pyproj.Transformer.from_crs("EPSG:32617", "EPSG:4326", always_xy=True)
    ox, oy = to_wgs.transform(x + 55, y + 55)
    from shapely.geometry import Point
    assert merged.contains(Point(ox, oy))
    assert not FP.contains(Point(ox, oy))           # proves the old clip missed it

def test_falls_back_to_footprint_when_outline_unusable():
    meta = {"footprint_wgs84": FP, "crs": "EPSG:32617"}
    assert _lidar_clip_geometry(meta, _roof(None)) is FP
    assert _lidar_clip_geometry(meta, _roof(box(0, 0, 1, 1), georef=False)) is FP
