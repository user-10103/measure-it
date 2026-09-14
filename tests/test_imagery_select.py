"""Imagery source selection: GIS orthophoto first, NAIP as the fallback.

The facet model is fine-tuned on county GIS imagery (3-6 inch) but serving
called the NAIP path (30-100 cm) unconditionally — inference at ~4x the
training GSD. These tests pin the ordering, the fallback, and the requirement
that the chosen source is always recorded.
"""
import pytest

from src.ingestion.imagery_select import candidate_sources, fetch_chip_best


def test_registered_county_outranks_statewide_which_outranks_naip():
    tiers = [t for t, _ in candidate_sources(27.9, -82.7, "FL", "Pinellas")]
    assert tiers == ["county-3in", "fl-statewide", "naip"]


def test_county_flagged_as_blocking_is_still_attempted():
    """`reachable` was measured from ONE datacenter IP. Treating it as a veto
    serves NAIP to a GIS-trained model on the strength of a stale probe; the
    cost of trying and failing is one timeout."""
    tiers = [t for t, _ in candidate_sources(28.0, -82.5, "FL", "Hillsborough")]
    assert tiers[0] == "county-3in-unverified"
    assert tiers[-1] == "naip"


def test_outside_florida_there_is_no_gis_coverage_yet():
    """Stated plainly so it cannot be mistaken for a bug later: the endpoint
    registry is Florida-only, so every other state still serves NAIP and the
    train/serve gap stays open there until the registry grows."""
    assert [t for t, _ in candidate_sources(40.7, -74.0, "NJ", "Hudson")] == ["naip"]
    assert [t for t, _ in candidate_sources(47.6, -122.3, "WA", None)] == ["naip"]


def test_falls_through_to_naip_when_gis_servers_fail(monkeypatch, tmp_path):
    """A county server being down must degrade the resolution, not fail the
    report."""
    import src.ingestion.imagery_select as sel

    def _boom(*a, **kw):
        raise RuntimeError("ImageServer 503")

    sentinel = ("chip", "transform", "png", "anchor", {"crs": "EPSG:26917"})
    monkeypatch.setattr("src.ingestion.gis_chip.fetch_chip_gis", _boom)
    monkeypatch.setattr("src.serve.report_service.fetch_chip",
                        lambda *a, **kw: sentinel)
    _c, _t, _p, _a, meta = sel.fetch_chip_best(27.9, -82.7, "FL", tmp_path,
                                               county="Pinellas")
    assert meta["imagery_source"] == "naip"
    # and it says what it tried, so a silent downgrade is impossible
    assert len(meta["imagery_attempts"]) == 2
    assert "ImageServer 503" in meta["imagery_attempts"][0]


def test_gis_success_records_the_source_and_its_gsd(monkeypatch, tmp_path):
    import src.ingestion.imagery_select as sel

    sentinel = ("chip", "transform", "png", "anchor", {"crs": "EPSG:26917"})
    monkeypatch.setattr("src.ingestion.gis_chip.fetch_chip_gis",
                        lambda *a, **kw: sentinel)
    _c, _t, _p, _a, meta = sel.fetch_chip_best(27.9, -82.7, "FL", tmp_path,
                                               county="Pinellas")
    assert meta["imagery_source"] == "county-3in"
    assert meta["imagery_gsd_m"] == pytest.approx(0.0762)
    assert meta["imagery_attempts"] == []


def test_every_source_failing_raises_with_what_was_tried(monkeypatch, tmp_path):
    import src.ingestion.imagery_select as sel

    def _boom(*a, **kw):
        raise RuntimeError("no imagery")

    monkeypatch.setattr("src.ingestion.gis_chip.fetch_chip_gis", _boom)
    monkeypatch.setattr("src.serve.report_service.fetch_chip", _boom)
    with pytest.raises(RuntimeError, match="No imagery source succeeded"):
        sel.fetch_chip_best(27.9, -82.7, "FL", tmp_path, county="Pinellas")


# --- state must come from the geocode, never from a literal -----------------

def test_state_and_county_come_back_from_one_census_call(monkeypatch):
    """Both are needed and neither is worth a second round trip."""
    import src.ingestion.imagery_select as sel

    payload = {"result": {"geographies": {
        "Counties": [{"BASENAME": "Polk"}],
        "States": [{"STUSAB": "IA", "BASENAME": "Iowa"}]}}}
    monkeypatch.setattr(sel, "_census_geographies", lambda *a, **k: payload["result"]["geographies"], raising=False)
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _FakeResp(payload))
    assert sel.state_county_for(41.59, -93.60) == ("IA", "Polk")


def test_state_falls_back_to_the_name_when_stusab_is_absent(monkeypatch):
    """Older Census vintages carry only the full state name."""
    import src.ingestion.imagery_select as sel
    payload = {"result": {"geographies": {
        "Counties": [], "States": [{"BASENAME": "Wyoming"}]}}}
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: _FakeResp(payload))
    assert sel.state_county_for(44.46, -110.83) == ("WY", None)


def test_lookup_failure_degrades_it_never_raises(monkeypatch):
    import src.ingestion.imagery_select as sel

    def _boom(*a, **k):
        raise OSError("census down")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert sel.state_county_for(41.59, -93.60) == (None, None)


class _FakeResp:
    def __init__(self, payload):
        import json
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_a_degraded_source_warns_rather_than_whispering(monkeypatch, tmp_path, caplog):
    """imagery_attempts was recorded on meta and read by nothing -- not the
    return value, not the PDF, not a visible log line. demo_lib.live_report
    mutes every logger to ERROR, so an INFO message explaining why a 30 cm NAIP
    chip was used where a 15 cm county ortho existed went nowhere at all."""
    import logging

    import src.ingestion.imagery_select as sel

    def _boom(*a, **kw):
        raise RuntimeError("ImageServer 503")

    sentinel = ("chip", "transform", "png", "anchor", {"crs": "EPSG:26917"})
    monkeypatch.setattr("src.ingestion.gis_chip.fetch_chip_gis", _boom)
    monkeypatch.setattr("src.serve.report_service.fetch_chip",
                        lambda *a, **kw: sentinel)
    with caplog.at_level(logging.WARNING, logger="src.ingestion.imagery_select"):
        sel.fetch_chip_best(27.9, -82.7, "FL", tmp_path, county="Pinellas")
    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "imagery DEGRADED to naip" in msg, msg
    assert "ImageServer 503" in msg, msg          # and WHY the better one lost


def test_the_best_source_does_not_warn(monkeypatch, tmp_path, caplog):
    import logging

    import src.ingestion.imagery_select as sel

    sentinel = ("chip", "transform", "png", "anchor", {"crs": "EPSG:26917"})
    monkeypatch.setattr("src.ingestion.gis_chip.fetch_chip_gis",
                        lambda *a, **kw: sentinel)
    with caplog.at_level(logging.WARNING, logger="src.ingestion.imagery_select"):
        sel.fetch_chip_best(27.9, -82.7, "FL", tmp_path, county="Pinellas")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
