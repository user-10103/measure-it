"""Occluded-roof guard: outline present but zero facets must be flagged
through to report_input (so the PDF banner fires), not silently delivered blank."""

from src.rgb_pipeline import process_chip_rgb


def _sq(x, y, s=40):
    return [[x, y], [x + s, y], [x + s, y + s], [x, y + s]]


def test_occluded_flagged_and_threaded_to_report_input():
    res = process_chip_rgb({"outline": _sq(0, 0, 200), "facets": []},
                           address="occ", gsd_m_per_px=0.3, write_outputs=False)
    assert res["occluded_roof"] is True
    assert res["report_input"]["occluded_roof"] is True   # what generate_report reads


def test_normal_roof_not_occluded():
    res = process_chip_rgb(
        {"outline": _sq(0, 0, 200),
         "facets": [{"polygon": _sq(20, 20), "slope_deg": 18.0, "aspect_bin": "S"},
                    {"polygon": _sq(80, 20), "slope_deg": 18.0, "aspect_bin": "N"}]},
        address="norm", gsd_m_per_px=0.3, write_outputs=False)
    assert res["occluded_roof"] is False
    assert res["report_input"]["occluded_roof"] is False


def test_nothing_detected_is_not_occluded():
    # no outline AND no facets is empty, not "occluded" (which means a roof WAS
    # found but couldn't be segmented).
    res = process_chip_rgb({"outline": None, "facets": []},
                           address="empty", gsd_m_per_px=0.3, write_outputs=False)
    assert res["occluded_roof"] is False
