"""
Raster Clipping Module

Clips rasters to building footprint polygons.

Reference: agent.md §4.1, agent-2.md clip_to_footprint tool
"""

import logging
from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from shapely.geometry import Polygon, MultiPolygon, mapping
import geopandas as gpd

logger = logging.getLogger(__name__)


def clip_to_footprint(
    raster_path: str,
    footprint: Union[str, Polygon, gpd.GeoDataFrame],
    out_path: str,
    crop: bool = True,
    nodata: float = 0.0
) -> str:
    """
    Clip raster using a polygon footprint.

    Reference: agent-2.md clip_to_footprint tool

    Args:
        raster_path: Path to input raster
        footprint: Building footprint as path to file, Shapely geometry,
                   or GeoDataFrame
        out_path: Output path for clipped raster
        crop: If True, crop to footprint bounds (default True)
        nodata: NoData value for pixels outside footprint

    Returns:
        Output file path

    Raises:
        FileNotFoundError: If raster doesn't exist
        ValueError: If footprint is invalid
        RuntimeError: If clipping fails
    """
    raster_path = Path(raster_path)
    out_path = Path(out_path)

    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve footprint to geometry
    geom = _resolve_footprint(footprint)

    logger.info(f"Clipping {raster_path.name} to footprint")

    try:
        with rasterio.open(raster_path) as src:
            # Convert geometry to raster's CRS if needed
            geom_crs = _ensure_geometry_crs(geom, src.crs)

            # Perform clip
            out_image, out_transform = mask(
                src,
                [mapping(geom_crs)],
                crop=crop,
                filled=True,
                nodata=nodata
            )

            # Update profile
            out_profile = src.profile.copy()
            out_profile.update({
                'height': out_image.shape[1],
                'width': out_image.shape[2],
                'transform': out_transform,
                'nodata': nodata,
                'compress': 'lzw'
            })

            # Write clipped raster
            with rasterio.open(str(out_path), 'w', **out_profile) as dst:
                dst.write(out_image)

        logger.info(
            f"Clipped raster saved: {out_path} "
            f"({out_image.shape[2]}x{out_image.shape[1]} pixels)"
        )

        return str(out_path)

    except Exception as e:
        error_msg = f"Raster clipping failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def clip_and_mask(
    raster_path: str,
    footprint: Union[str, Polygon, gpd.GeoDataFrame],
    out_path: str,
    nodata: float = 0.0
) -> Tuple[str, np.ndarray]:
    """
    Clip raster and return the binary mask.

    Useful for subsequent point extraction where you need to know
    which pixels are inside the footprint.

    Args:
        raster_path: Path to input raster
        footprint: Building footprint
        out_path: Output path for clipped raster
        nodata: NoData value

    Returns:
        Tuple of (output_path, mask_array) where mask_array is True
        for pixels inside footprint
    """
    raster_path = Path(raster_path)
    out_path = Path(out_path)

    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    geom = _resolve_footprint(footprint)

    logger.info(f"Clipping and masking {raster_path.name}")

    try:
        with rasterio.open(raster_path) as src:
            geom_crs = _ensure_geometry_crs(geom, src.crs)

            # Clip raster
            out_image, out_transform = mask(
                src,
                [mapping(geom_crs)],
                crop=True,
                filled=True,
                nodata=nodata
            )

            # Create binary mask
            mask_array = ~geometry_mask(
                [mapping(geom_crs)],
                transform=out_transform,
                invert=False,
                out_shape=(out_image.shape[1], out_image.shape[2])
            )

            # Update profile and write
            out_profile = src.profile.copy()
            out_profile.update({
                'height': out_image.shape[1],
                'width': out_image.shape[2],
                'transform': out_transform,
                'nodata': nodata,
                'compress': 'lzw'
            })

            with rasterio.open(str(out_path), 'w', **out_profile) as dst:
                dst.write(out_image)

        logger.info(
            f"Clipped raster with mask: {out_path} "
            f"(mask covers {mask_array.sum()} pixels)"
        )

        return str(out_path), mask_array

    except Exception as e:
        error_msg = f"Clip and mask failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def get_footprint_mask(
    raster_path: str,
    footprint: Union[str, Polygon, gpd.GeoDataFrame]
) -> np.ndarray:
    """
    Generate binary mask for footprint without clipping.

    Returns mask matching full raster dimensions.

    Args:
        raster_path: Path to reference raster
        footprint: Building footprint

    Returns:
        Boolean array where True = inside footprint
    """
    geom = _resolve_footprint(footprint)

    with rasterio.open(raster_path) as src:
        geom_crs = _ensure_geometry_crs(geom, src.crs)

        mask_array = ~geometry_mask(
            [mapping(geom_crs)],
            transform=src.transform,
            invert=False,
            out_shape=(src.height, src.width)
        )

    return mask_array


def _resolve_footprint(
    footprint: Union[str, Polygon, MultiPolygon, gpd.GeoDataFrame]
) -> Polygon:
    """
    Convert various footprint formats to Shapely Polygon.

    Args:
        footprint: Path to file, Shapely geometry, or GeoDataFrame

    Returns:
        Shapely Polygon

    Raises:
        ValueError: If footprint format is invalid
    """
    if isinstance(footprint, (Polygon, MultiPolygon)):
        if isinstance(footprint, MultiPolygon):
            # Use largest polygon
            footprint = max(footprint.geoms, key=lambda g: g.area)
        return footprint

    if isinstance(footprint, gpd.GeoDataFrame):
        if len(footprint) == 0:
            raise ValueError("GeoDataFrame is empty")
        geom = footprint.geometry.iloc[0]
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda g: g.area)
        return geom

    if isinstance(footprint, str):
        path = Path(footprint)
        if not path.exists():
            raise FileNotFoundError(f"Footprint file not found: {path}")

        gdf = gpd.read_file(path)
        if len(gdf) == 0:
            raise ValueError(f"Footprint file is empty: {path}")

        geom = gdf.geometry.iloc[0]
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda g: g.area)
        return geom

    raise ValueError(f"Unsupported footprint type: {type(footprint)}")


def _ensure_geometry_crs(
    geom: Polygon,
    target_crs,
    source_crs: str = "EPSG:4326"
) -> Polygon:
    """
    Transform geometry to target CRS if needed.

    Assumes input geometry is in source_crs (default WGS84).

    Args:
        geom: Input Shapely geometry
        target_crs: Target CRS (rasterio CRS object)
        source_crs: Source CRS string (default EPSG:4326)

    Returns:
        Transformed geometry
    """
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform

    # If target is WGS84, return as-is
    if str(target_crs) == source_crs:
        return geom

    # Transform to target CRS
    transformer = Transformer.from_crs(
        source_crs,
        str(target_crs),
        always_xy=True
    )

    return shp_transform(transformer.transform, geom)
