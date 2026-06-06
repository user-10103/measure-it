"""
Raster Fusion Module

Aligns, stacks, and clips rasters to produce fused 5-band analytic rasters.
Band order: R, G, B, NIR, Height (nDSM)

Reference: agent.md §3.4, agent-2.md pipeline.align_naip
"""

from src.fusion.align import align_raster_to_target
from src.fusion.stack import stack_bands
from src.fusion.clip import clip_to_footprint

__all__ = [
    "align_raster_to_target",
    "stack_bands",
    "clip_to_footprint",
]
