"""
File caching utilities for downloaded data.
Implements "download once, use forever" pattern to avoid redundant network calls.
"""
import logging
import hashlib
from pathlib import Path
from typing import Optional, Callable
import requests
from tqdm import tqdm
from src.config import DATA_CACHE_DIR

logger = logging.getLogger(__name__)


def get_cache_path(url: str, subdirectory: str = "") -> Path:
    """
    Generate a cache file path based on URL.
    Uses URL hash to create unique filenames.

    Args:
        url: Source URL
        subdirectory: Optional subdirectory within cache (e.g., "naip", "msbuildings")

    Returns:
        Path object for cache file

    Example:
        >>> path = get_cache_path("https://example.com/data.csv", "msbuildings")
        >>> print(path)
        /path/to/data/msbuildings/abc123def456.csv
    """
    # Extract filename from URL
    filename = url.split("/")[-1].split("?")[0]

    # Create hash of full URL for uniqueness
    url_hash = hashlib.md5(url.encode()).hexdigest()[:16]

    # Get file extension
    ext = Path(filename).suffix if "." in filename else ""

    # Construct cache path
    cache_dir = DATA_CACHE_DIR / subdirectory
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"{url_hash}{ext}"
    return cache_file


def download_file(
    url: str,
    dest_path: Path,
    force: bool = False,
    show_progress: bool = True,
    headers: Optional[dict] = None
) -> Path:
    """
    Download a file with caching and progress bar.
    Skips download if file already exists (unless force=True).

    Args:
        url: Source URL
        dest_path: Destination file path
        force: Force re-download even if file exists
        show_progress: Show download progress bar
        headers: Optional HTTP headers (e.g., for requester-pays)

    Returns:
        Path to downloaded file

    Raises:
        requests.HTTPError: If download fails

    Example:
        >>> path = download_file(
        ...     "https://example.com/data.csv",
        ...     Path("data/data.csv"),
        ...     headers={"x-amz-request-payer": "requester"}
        ... )
    """
    # Check if file already exists
    if dest_path.exists() and not force:
        logger.debug(f"Cache hit: {dest_path.name}")
        return dest_path

    # Create parent directory
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Download file
    logger.info(f"Downloading: {url}")
    response = requests.get(url, stream=True, headers=headers or {})
    response.raise_for_status()

    # Get file size
    total_size = int(response.headers.get('content-length', 0))

    # Download with progress bar
    with open(dest_path, 'wb') as f:
        if show_progress and total_size > 0:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        else:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    logger.info(f"Downloaded to: {dest_path}")
    return dest_path


def cached_download(
    url: str,
    subdirectory: str = "",
    force: bool = False,
    headers: Optional[dict] = None
) -> Path:
    """
    Download and cache a file.
    Combines get_cache_path and download_file for convenience.

    Args:
        url: Source URL
        subdirectory: Cache subdirectory
        force: Force re-download
        headers: Optional HTTP headers

    Returns:
        Path to cached file

    Example:
        >>> path = cached_download(
        ...     "https://example.com/data.csv",
        ...     subdirectory="msbuildings"
        ... )
    """
    cache_path = get_cache_path(url, subdirectory)
    return download_file(url, cache_path, force=force, headers=headers)


def is_cached(url: str, subdirectory: str = "") -> bool:
    """
    Check if a URL has been cached.

    Args:
        url: Source URL
        subdirectory: Cache subdirectory

    Returns:
        True if file exists in cache

    Example:
        >>> if is_cached("https://example.com/data.csv", "msbuildings"):
        ...     print("Already downloaded")
    """
    cache_path = get_cache_path(url, subdirectory)
    return cache_path.exists()


def clear_cache(subdirectory: str = "") -> int:
    """
    Clear cached files in a subdirectory.

    Args:
        subdirectory: Cache subdirectory to clear (empty = clear all)

    Returns:
        Number of files deleted

    Example:
        >>> count = clear_cache("naip")
        >>> print(f"Deleted {count} files")
    """
    cache_dir = DATA_CACHE_DIR / subdirectory
    if not cache_dir.exists():
        return 0

    count = 0
    for file in cache_dir.rglob("*"):
        if file.is_file():
            file.unlink()
            count += 1

    logger.info(f"Cleared {count} cached files from {cache_dir}")
    return count


def get_cache_size(subdirectory: str = "") -> int:
    """
    Get total size of cached files in bytes.

    Args:
        subdirectory: Cache subdirectory (empty = all)

    Returns:
        Total size in bytes

    Example:
        >>> size_mb = get_cache_size("naip") / (1024 * 1024)
        >>> print(f"Cache size: {size_mb:.1f} MB")
    """
    cache_dir = DATA_CACHE_DIR / subdirectory
    if not cache_dir.exists():
        return 0

    total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
    return total_size
