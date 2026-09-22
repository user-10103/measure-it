"""
Roof Segmentation Module

Segments roof points into facets using various methods:
- Single plane (MVP)
- Height-based K-means clustering
- Gradient region-growing (advanced)

Reference: agent.md §5
"""

import logging
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


# Default parameters
DEFAULT_K_CANDIDATES = [4, 5, 6, 7, 8]
DEFAULT_MIN_FACET_POINTS = 20
DEFAULT_GRADIENT_THRESHOLD = 0.3


class Facet:
    """
    Represents a roof facet.

    LiDAR path: carries a structured point array (x, y, z, ...).
    RGB/ML path: carries a 2D `polygon` (shapely, in a planar metric CRS) plus a
    `plane` (PlaneModel synthesized from predicted pitch/aspect) and the raw
    `slope_deg` / `aspect_bin`. `points` is None for ML facets.

    Attributes:
        facet_id: Unique identifier
        points: Point array for this facet (None for ML facets)
        label: Cluster label (for k-means)
        polygon: 2D boundary polygon (ML path)
        plane: PlaneModel (ML path; synthesized from pitch/aspect)
        slope_deg: Slope in degrees (ML path)
        aspect_bin: Compass aspect bin (ML path)
    """

    def __init__(self, facet_id: int, points: Optional[np.ndarray] = None, label: int = 0,
                 polygon=None, plane=None, slope_deg: Optional[float] = None,
                 aspect_bin: Optional[str] = None):
        self.facet_id = facet_id
        self.points = points
        self.label = label
        self.polygon = polygon
        self.plane = plane
        self.slope_deg = slope_deg
        self.aspect_bin = aspect_bin

    @property
    def count(self) -> int:
        return 0 if self.points is None else len(self.points)

    @property
    def z_mean(self) -> float:
        return float(self.points['z'].mean()) if self.count > 0 else 0.0

    @property
    def z_std(self) -> float:
        return float(self.points['z'].std()) if self.count > 0 else 0.0

    @property
    def is_flat(self) -> bool:
        return self.slope_deg is not None and self.slope_deg < 5.0

    def to_dict(self) -> dict:
        return {
            "facet_id": self.facet_id,
            "point_count": self.count,
            "z_mean": self.z_mean,
            "z_std": self.z_std,
            "label": self.label,
            "slope_deg": self.slope_deg,
            "aspect_bin": self.aspect_bin,
        }


def segment_single_plane(points: np.ndarray) -> List[Facet]:
    """
    Single-plane segmentation (MVP mode).

    Treats all points as belonging to one facet.

    Reference: agent.md §5 Option A

    Args:
        points: Structured array with x, y, z fields

    Returns:
        List containing single Facet with all points
    """
    logger.info(f"Single-plane segmentation: {len(points):,} points → 1 facet")

    return [Facet(facet_id=0, points=points, label=0)]


def segment_kmeans(
    points: np.ndarray,
    k_candidates: Optional[List[int]] = None,
    min_facet_points: int = DEFAULT_MIN_FACET_POINTS
) -> List[Facet]:
    """
    Height-based K-means clustering segmentation.

    Clusters points by Z (height) value to find distinct roof planes
    at different elevations.

    Reference: agent.md §5 Option B

    Args:
        points: Structured array with z field
        k_candidates: List of K values to try (default [2,3,4])
        min_facet_points: Minimum points per facet

    Returns:
        List of Facet objects, one per cluster
    """
    from sklearn.cluster import KMeans

    if k_candidates is None:
        k_candidates = DEFAULT_K_CANDIDATES

    if len(points) < min_facet_points:
        logger.warning(
            f"Too few points ({len(points)}) for k-means, using single plane"
        )
        return segment_single_plane(points)

    logger.info(f"K-means segmentation: {len(points):,} points, trying K={k_candidates}")

    # Use normalized (x, y, z) for clustering - better spatial separation than z-only
    # This properly separates different spatial regions of the roof
    x_vals = points['x']
    y_vals = points['y']
    z_vals = points['z']

    # Normalize each dimension to have similar scale
    x_norm = (x_vals - x_vals.mean()) / (x_vals.std() + 1e-10)
    y_norm = (y_vals - y_vals.mean()) / (y_vals.std() + 1e-10)
    z_norm = (z_vals - z_vals.mean()) / (z_vals.std() + 1e-10)

    xyz_values = np.column_stack([x_norm, y_norm, z_norm])

    # Try different K values and pick best (silhouette or elbow)
    best_k = k_candidates[0]
    best_score = -np.inf
    best_labels = None

    for k in k_candidates:
        if k > len(points):
            continue

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(xyz_values)

        # Simple scoring: prefer K where all clusters have enough points
        cluster_sizes = [np.sum(labels == i) for i in range(k)]
        min_size = min(cluster_sizes)

        if min_size >= min_facet_points:
            # Score based on variance reduction (using z for roof plane relevance)
            total_var = z_norm.var()
            within_var = sum(
                z_norm[labels == i].var() * (labels == i).sum()
                for i in range(k)
            ) / len(z_norm)
            score = 1 - (within_var / (total_var + 1e-10))

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

    if best_labels is None:
        # Fallback to k=2
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        best_labels = kmeans.fit_predict(xyz_values)
        best_k = 2

    # Create facets from clusters
    facets = []
    for i in range(best_k):
        mask = best_labels == i
        if mask.sum() >= min_facet_points:
            facet = Facet(
                facet_id=len(facets),
                points=points[mask],
                label=i
            )
            facets.append(facet)

    # Sort facets by mean height (highest first)
    facets.sort(key=lambda f: f.z_mean, reverse=True)

    # Re-assign facet IDs after sorting
    for i, facet in enumerate(facets):
        facet.facet_id = i

    logger.info(
        f"K-means result: K={best_k} → {len(facets)} facets "
        f"(sizes: {[f.count for f in facets]})"
    )

    return facets


def segment_region_growing(
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
    min_region_pixels: int = 50
) -> List[Facet]:
    """
    Gradient-based region-growing segmentation.

    Groups pixels with smooth slope and similar gradient direction.

    Reference: agent.md §5 Option C

    Args:
        grid_z: 2D height grid (DSM or nDSM)
        grid_x: 2D X coordinate grid
        grid_y: 2D Y coordinate grid
        gradient_threshold: Max gradient difference for merging
        min_region_pixels: Minimum pixels per region

    Returns:
        List of Facet objects

    Note:
        This method works on gridded data, not point arrays.
        Points are reconstructed from the grid after segmentation.
    """
    from scipy.ndimage import sobel, label

    logger.info(f"Region-growing segmentation on {grid_z.shape} grid")

    # Compute gradient magnitude and direction
    grad_x = sobel(grid_z, axis=1)
    grad_y = sobel(grid_z, axis=0)

    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_dir = np.arctan2(grad_y, grad_x)

    # Threshold gradient magnitude to find smooth regions
    smooth_mask = grad_mag < gradient_threshold

    # Label connected smooth regions
    labeled_array, num_features = label(smooth_mask)

    logger.debug(f"Found {num_features} initial regions")

    # Create facets from labeled regions
    facets = []
    for region_id in range(1, num_features + 1):
        region_mask = labeled_array == region_id
        region_size = region_mask.sum()

        if region_size < min_region_pixels:
            continue

        # Extract points from region
        region_x = grid_x[region_mask]
        region_y = grid_y[region_mask]
        region_z = grid_z[region_mask]

        # Create structured array
        dtype = np.dtype([
            ('x', np.float64),
            ('y', np.float64),
            ('z', np.float32),
            ('r', np.float32),
            ('g', np.float32),
            ('b', np.float32),
            ('nir', np.float32)
        ])

        points = np.zeros(region_size, dtype=dtype)
        points['x'] = region_x
        points['y'] = region_y
        points['z'] = region_z
        # RGB/NIR set to 0 since we don't have that info from grid

        facet = Facet(
            facet_id=len(facets),
            points=points,
            label=region_id
        )
        facets.append(facet)

    # Sort by mean height
    facets.sort(key=lambda f: f.z_mean, reverse=True)
    for i, facet in enumerate(facets):
        facet.facet_id = i

    logger.info(
        f"Region-growing result: {len(facets)} facets "
        f"(sizes: {[f.count for f in facets]})"
    )

    return facets


class RoofModelBackend:
    """Interface for an RGB roof-facet segmentation model.

    Implementations return facets in PIXEL coordinates of the input chip:
        predict(image) -> {
            "outline": [[x,y],...] or None,
            "facets": [{"polygon": [[x,y],...],
                        "slope_deg": float|None,
                        "aspect_bin": str|None}, ...]
        }
    The real model (RF-DETR-Seg, Apache) loads its own weights lazily; until
    those exist, use CocoStandinBackend to exercise the seam on the v2_4 data.
    """

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError


class CocoStandinBackend(RoofModelBackend):
    """Stand-in backend that serves facets from a COCO file keyed by address_id.

    Lets the ML seam (and the whole RGB pipeline) be built and tested before the
    trained model exists, using the cleaned v2_4 facet annotations as if they
    were model predictions.
    """

    def __init__(self, coco: dict):
        from collections import defaultdict
        self.images = {im["address_id"]: im for im in coco["images"]}
        self.by_img: Dict[int, list] = defaultdict(list)
        for a in coco["annotations"]:
            self.by_img[a["image_id"]].append(a)

    @classmethod
    def from_file(cls, path: str) -> "CocoStandinBackend":
        import json
        return cls(json.load(open(path)))

    def predict_for(self, address_id: str) -> Dict[str, Any]:
        from src.roofs.categories import CAT_ROOF, CAT_FACET

        def ring_to_xy(seg):
            return [[seg[i], seg[i + 1]] for i in range(0, len(seg) - 1, 2)]

        im = self.images.get(address_id)
        if im is None:
            return {"outline": None, "facets": []}
        outline = None
        facets = []
        for a in self.by_img.get(im["id"], []):
            if a["category_id"] == CAT_ROOF:
                outline = ring_to_xy(a["segmentation"][0])
            elif a["category_id"] == CAT_FACET:
                attrs = a.get("attributes") or {}
                facets.append({"polygon": ring_to_xy(a["segmentation"][0]),
                               "slope_deg": attrs.get("slope_deg"),
                               "aspect_bin": attrs.get("aspect_label")})
        return {"outline": outline, "facets": facets}

    def predict(self, image: np.ndarray) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError("CocoStandinBackend serves by address_id; use predict_for()")


def segment_facets_ml(
    prediction: Dict[str, Any],
    pixel_to_world=None,
) -> List[Facet]:
    """Build Facet objects from a model prediction dict (RGB path).

    Args:
        prediction: {"outline": [...]|None, "facets": [{polygon, slope_deg, aspect_bin}]}.
        pixel_to_world: optional callable (x, y) -> (X, Y) mapping pixel coords to
            a planar metric CRS (so areas/lengths come out in metres). Identity
            if None.

    Returns:
        List[Facet] each carrying `polygon` (shapely) + synthesized `plane`.
    """
    from shapely.geometry import Polygon
    from src.roofs.plane_fit import plane_from_pitch_aspect

    def to_poly(xy):
        pts = [tuple(pixel_to_world(x, y)) for x, y in xy] if pixel_to_world else \
              [(x, y) for x, y in xy]
        if len(pts) < 3:
            return None
        p = Polygon(pts)
        return p.buffer(0) if not p.is_valid else p

    facets: List[Facet] = []
    for i, f in enumerate(prediction.get("facets", [])):
        poly = to_poly(f["polygon"])
        if poly is None or poly.is_empty:
            continue
        slope = f.get("slope_deg")
        aspect = f.get("aspect_bin")
        plane = plane_from_pitch_aspect(slope if slope is not None else 0.0,
                                        aspect_bin=aspect)
        facets.append(Facet(facet_id=len(facets), points=None, polygon=poly,
                            plane=plane, slope_deg=slope, aspect_bin=aspect))
    return facets


def segment_facets(
    points: Optional[np.ndarray] = None,
    method: str = "single",
    k_candidates: Optional[List[int]] = None,
    grid_z: Optional[np.ndarray] = None,
    grid_x: Optional[np.ndarray] = None,
    grid_y: Optional[np.ndarray] = None,
    prediction: Optional[Dict[str, Any]] = None,
    pixel_to_world=None,
    **kwargs
) -> List[Facet]:
    """
    Unified segmentation interface.

    Reference: agent-2.md segment_facets tool

    Args:
        points: Structured point array (LiDAR methods)
        method: "single", "kmeans", "regiongrow", or "ml"
        k_candidates: K values for k-means
        grid_z, grid_x, grid_y: Grids for region-growing
        prediction: model output dict for method="ml" (see segment_facets_ml)
        pixel_to_world: pixel->metric mapping for method="ml"
        **kwargs: Additional method-specific parameters

    Returns:
        List of Facet objects
    """
    method = method.lower()

    if method == "single":
        return segment_single_plane(points)

    elif method == "kmeans":
        return segment_kmeans(points, k_candidates=k_candidates, **kwargs)

    elif method == "regiongrow":
        if grid_z is None or grid_x is None or grid_y is None:
            raise ValueError(
                "Region-growing requires grid_z, grid_x, grid_y arrays"
            )
        return segment_region_growing(grid_z, grid_x, grid_y, **kwargs)

    elif method == "ml":
        if prediction is None:
            raise ValueError("method='ml' requires a `prediction` dict")
        return segment_facets_ml(prediction, pixel_to_world=pixel_to_world)

    else:
        raise ValueError(f"Unknown segmentation method: {method}")


def merge_small_facets(
    facets: List[Facet],
    min_points: int = DEFAULT_MIN_FACET_POINTS
) -> List[Facet]:
    """
    Merge small facets into nearest larger neighbor.

    Args:
        facets: List of Facet objects
        min_points: Minimum points to keep as separate facet

    Returns:
        Merged list of facets
    """
    if not facets:
        return facets

    # Separate large and small facets
    large = [f for f in facets if f.count >= min_points]
    small = [f for f in facets if f.count < min_points]

    if not large:
        # All facets are small - merge into one
        all_points = np.concatenate([f.points for f in facets])
        return [Facet(facet_id=0, points=all_points)]

    if not small:
        return large

    # Merge small facets into nearest large by z_mean
    for small_facet in small:
        # Find nearest large facet by height
        nearest = min(large, key=lambda f: abs(f.z_mean - small_facet.z_mean))
        # Merge points
        merged_points = np.concatenate([nearest.points, small_facet.points])
        nearest.points = merged_points

    # Re-index
    for i, facet in enumerate(large):
        facet.facet_id = i

    logger.debug(f"Merged {len(small)} small facets into {len(large)} large facets")

    return large


def get_dominant_facet(facets: List[Facet]) -> Optional[Facet]:
    """
    Get the largest facet by point count.

    Args:
        facets: List of Facet objects

    Returns:
        Largest facet or None if empty
    """
    if not facets:
        return None

    return max(facets, key=lambda f: f.count)
