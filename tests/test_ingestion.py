"""
Integration tests for data ingestion modules.
"""
import pytest
from pathlib import Path
from src.utils.geocode import get_coordinates
from src.utils.projections import create_meter_buffer
from src.ingestion.ms_footprints import get_buildings_in_buffer
from src.roofs.select_candidates import select_building


class TestGeocode:
    """Test geocoding functionality."""

    def test_explicit_coordinates(self):
        """Test using explicit lat/lon."""
        coords = get_coordinates(lat=28.1178, lon=-82.3951)
        assert coords["lat"] == 28.1178
        assert coords["lon"] == -82.3951

    def test_missing_input(self):
        """Test error handling for missing input."""
        with pytest.raises(ValueError):
            get_coordinates()


class TestProjections:
    """Test CRS transformation utilities."""

    def test_meter_buffer(self):
        """Test meter-accurate buffer creation."""
        buffer = create_meter_buffer(-82.3951, 28.1178, 150)

        assert buffer.crs.to_epsg() == 4326
        assert len(buffer) == 1


class TestMSFootprints:
    """Test Microsoft Buildings footprint ingestion."""

    @pytest.mark.integration
    def test_get_buildings_tampa(self):
        """Integration test: Fetch buildings in Tampa."""
        # Create buffer around known location
        buffer = create_meter_buffer(-82.3951, 28.1178, 150)

        # Get buildings
        buildings = get_buildings_in_buffer(buffer)

        # Should find buildings in populated area
        assert len(buildings) > 0
        assert "geometry" in buildings.columns


class TestBuildingSelection:
    """Test building candidate selection."""

    @pytest.mark.integration
    def test_select_building_tampa(self):
        """Integration test: Select building in Tampa."""
        result = select_building(28.1178, -82.3951, buffer_meters=150)

        assert result["selected"] is not None
        assert result["dist_m"] is not None
        assert len(result["candidates"]) > 0


def run_integration_tests():
    """
    Run integration tests that require network access.

    Example:
        >>> python -m tests.test_ingestion
    """
    import sys

    # Run with pytest
    sys.exit(pytest.main([__file__, "-v", "-m", "integration"]))


if __name__ == "__main__":
    run_integration_tests()
