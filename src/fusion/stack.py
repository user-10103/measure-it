"""
Band Stacking Module

Stacks separate band files into a multi-band GeoTIFF.
Standard band order: R, G, B, NIR, nDSM (Height)

Reference: agent.md §3.4, agent-2.md stack_bands tool
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import rasterio
from rasterio.crs import CRS

logger = logging.getLogger(__name__)


# Standard 5-band order per agent spec
BAND_ORDER = ["R", "G", "B", "NIR", "nDSM"]
BAND_NAMES = {
    1: "Red",
    2: "Green",
    3: "Blue",
    4: "NIR",
    5: "Height (nDSM)"
}


def stack_bands(
    band_paths: List[str],
    out_path: str,
    band_names: Optional[List[str]] = None
) -> str:
    """
    Stack separate band files into a single multi-band GeoTIFF.

    All input bands must be aligned (same CRS, transform, dimensions).

    Reference: agent-2.md stack_bands tool
    Reference: agent.md §3.4 - "stack = [R, G, B, NIR, nDSM]"

    Args:
        band_paths: List of paths to single-band rasters in order
        out_path: Output path for stacked raster
        band_names: Optional list of band names for metadata

    Returns:
        Output file path

    Raises:
        ValueError: If bands don't match or insufficient bands
        FileNotFoundError: If any band file doesn't exist
        RuntimeError: If stacking fails
    """
    if not band_paths:
        raise ValueError("No band paths provided")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate all files exist
    for path in band_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Band file not found: {path}")

    logger.info(f"Stacking {len(band_paths)} bands into {out_path.name}")

    try:
        # Read first band to get reference properties
        with rasterio.open(band_paths[0]) as ref:
            ref_profile = ref.profile.copy()
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_width = ref.width
            ref_height = ref.height
            ref_dtype = ref.dtypes[0]

        # Validate all bands match reference
        for i, path in enumerate(band_paths[1:], start=2):
            with rasterio.open(path) as src:
                if src.crs != ref_crs:
                    raise ValueError(
                        f"Band {i} CRS mismatch: {src.crs} vs {ref_crs}"
                    )
                if src.width != ref_width or src.height != ref_height:
                    raise ValueError(
                        f"Band {i} dimension mismatch: "
                        f"{src.width}x{src.height} vs {ref_width}x{ref_height}"
                    )
                if list(src.transform) != list(ref_transform):
                    raise ValueError(f"Band {i} transform mismatch")

        # Read all bands into memory
        n_bands = len(band_paths)
        stacked = np.zeros((n_bands, ref_height, ref_width), dtype=ref_dtype)

        for i, path in enumerate(band_paths):
            with rasterio.open(path) as src:
                stacked[i] = src.read(1)
            logger.debug(f"Read band {i+1}: {Path(path).name}")

        # Update profile for output
        out_profile = ref_profile.copy()
        out_profile.update({
            'count': n_bands,
            'compress': 'lzw'
        })

        # Write stacked raster
        with rasterio.open(str(out_path), 'w', **out_profile) as dst:
            dst.write(stacked)

            # Set band descriptions if provided
            if band_names:
                for i, name in enumerate(band_names, start=1):
                    dst.set_band_description(i, name)
            elif n_bands == 5:
                # Default to standard 5-band naming
                for i, name in BAND_NAMES.items():
                    if i <= n_bands:
                        dst.set_band_description(i, name)

        logger.info(
            f"Stacked raster saved: {out_path} "
            f"({ref_width}x{ref_height}, {n_bands} bands)"
        )

        return str(out_path)

    except Exception as e:
        error_msg = f"Band stacking failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def stack_naip_with_ndsm(
    naip_path: str,
    ndsm_path: str,
    out_path: str
) -> str:
    """
    Create 5-band fused raster from 4-band NAIP and 1-band nDSM.

    Convenience function for the most common fusion operation.
    Output band order: R, G, B, NIR, Height

    Reference: agent.md §3.4 - "stack = [R, G, B, NIR, nDSM]"

    Args:
        naip_path: Path to 4-band NAIP (RGBN)
        ndsm_path: Path to 1-band nDSM
        out_path: Output path for 5-band fused raster

    Returns:
        Output file path

    Raises:
        ValueError: If NAIP doesn't have 4 bands
        RuntimeError: If fusion fails
    """
    naip_path = Path(naip_path)
    ndsm_path = Path(ndsm_path)
    out_path = Path(out_path)

    if not naip_path.exists():
        raise FileNotFoundError(f"NAIP file not found: {naip_path}")
    if not ndsm_path.exists():
        raise FileNotFoundError(f"nDSM file not found: {ndsm_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fusing NAIP (4-band) + nDSM into 5-band raster")

    try:
        # Verify NAIP has 4 bands
        with rasterio.open(naip_path) as naip:
            if naip.count != 4:
                raise ValueError(
                    f"NAIP must have 4 bands (RGBN), got {naip.count}"
                )
            naip_profile = naip.profile.copy()
            naip_data = naip.read()

        # Read nDSM
        with rasterio.open(ndsm_path) as ndsm:
            ndsm_data = ndsm.read(1)

            # Verify alignment
            with rasterio.open(naip_path) as naip:
                if ndsm.width != naip.width or ndsm.height != naip.height:
                    raise ValueError(
                        f"nDSM dimensions ({ndsm.width}x{ndsm.height}) "
                        f"don't match NAIP ({naip.width}x{naip.height}). "
                        "Run align_raster_to_target first."
                    )

        # Stack: R, G, B, NIR (from NAIP), Height (from nDSM)
        # Normalize nDSM to same dtype as NAIP if needed
        if naip_data.dtype != ndsm_data.dtype:
            # Scale nDSM to uint8/uint16 range if NAIP is integer
            if np.issubdtype(naip_data.dtype, np.integer):
                max_val = np.iinfo(naip_data.dtype).max
                # Scale height: assume 0-50m range maps to 0-max_val
                ndsm_scaled = np.clip(ndsm_data / 50.0 * max_val, 0, max_val)
                ndsm_data = ndsm_scaled.astype(naip_data.dtype)
            else:
                ndsm_data = ndsm_data.astype(naip_data.dtype)

        # Create 5-band stack
        fused = np.vstack([naip_data, ndsm_data[np.newaxis, :, :]])

        # Update profile
        out_profile = naip_profile.copy()
        out_profile.update({
            'count': 5,
            'compress': 'lzw'
        })

        # Write fused raster
        with rasterio.open(str(out_path), 'w', **out_profile) as dst:
            dst.write(fused)
            for i, name in BAND_NAMES.items():
                dst.set_band_description(i, name)

        logger.info(
            f"5-band fused raster saved: {out_path} "
            f"(bands: R, G, B, NIR, Height)"
        )

        return str(out_path)

    except Exception as e:
        error_msg = f"NAIP+nDSM fusion failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def extract_bands(
    src_path: str,
    out_dir: str,
    band_indices: Optional[List[int]] = None
) -> Dict[int, str]:
    """
    Extract individual bands from multi-band raster.

    Useful for processing individual bands separately.

    Args:
        src_path: Path to multi-band raster
        out_dir: Output directory
        band_indices: List of 1-based band indices (default: all bands)

    Returns:
        Dict mapping band index to output file path
    """
    src_path = Path(src_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        raise FileNotFoundError(f"Source raster not found: {src_path}")

    result = {}

    with rasterio.open(src_path) as src:
        if band_indices is None:
            band_indices = list(range(1, src.count + 1))

        base_name = src_path.stem

        for band_idx in band_indices:
            if band_idx < 1 or band_idx > src.count:
                raise ValueError(f"Invalid band index: {band_idx}")

            band_data = src.read(band_idx)

            out_profile = src.profile.copy()
            out_profile.update({'count': 1, 'compress': 'lzw'})

            out_path = out_dir / f"{base_name}_band{band_idx}.tif"

            with rasterio.open(str(out_path), 'w', **out_profile) as dst:
                dst.write(band_data, 1)

            result[band_idx] = str(out_path)
            logger.debug(f"Extracted band {band_idx} to {out_path.name}")

    logger.info(f"Extracted {len(result)} bands from {src_path.name}")

    return result
