"""
Custom exceptions for data ingestion operations.
"""


class IngestionError(Exception):
    """Base exception for all ingestion errors."""
    pass


class NAIPError(IngestionError):
    """Base exception for NAIP-related errors."""
    pass


class NAIPNotFoundError(NAIPError):
    """Raised when NAIP data cannot be found for a location."""
    pass


class NAIPDownloadError(NAIPError):
    """Raised when NAIP tile download fails."""
    pass


class NAIPClipError(NAIPError):
    """Raised when NAIP imagery clipping fails."""
    pass


class NAIPTileSearchError(NAIPError):
    """Raised when searching for NAIP tiles fails."""
    pass


class MSBuildingsError(IngestionError):
    """Base exception for Microsoft Buildings-related errors."""
    pass


class MSBuildingsIndexError(MSBuildingsError):
    """Raised when loading or parsing the MS Buildings index fails."""
    pass


class MSBuildingsDownloadError(MSBuildingsError):
    """Raised when downloading MS Buildings data fails."""
    pass


class MSBuildingsParseError(MSBuildingsError):
    """Raised when parsing MS Buildings shard files fails."""
    pass
