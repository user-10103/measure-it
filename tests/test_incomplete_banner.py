"""The INCOMPLETE stamp must survive being read back out of the PDF.

It regressed silently once: the plain-English reasons added in 0127de3 made the
single centred banner line wider than the page, so it overflowed both margins and
clipped the headline down to "TE - MANUAL REVIEW REQUIRED". Every automated stamp
check saw an unstamped report, and a failing report shipped looking finished —
the exact failure the banner exists to prevent. Assert on the EXTRACTED text, not
on the drawing call.
"""
import shutil
import subprocess

import pytest

from src.output.pdf_report import generate_report

pytestmark = pytest.mark.skipif(shutil.which("pdftotext") is None,
                                reason="needs poppler's pdftotext")


def _base():
    return {"address": "t", "report_id": "T",
            "outline_xy": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "facets": [{"facet_id": 1,
                        "polygon_xy": [[0, 0], [10, 0], [10, 10], [0, 10]],
                        "plan_area_m2": 100.0, "surface_area_m2": 100.0,
                        "slope_deg": 0.0, "pitch_string": "0:12",
                        "aspect_bin": None, "is_flat": True,
                        "needs_review": False}],
            "edges": [{"edge_type": "eave", "length_m": 10,
                       "geometry_xy": [[0, 0], [10, 0]]}]}


def _cover_text(path):
    return subprocess.run(["pdftotext", "-f", "1", "-l", "1", str(path), "-"],
                          capture_output=True, text=True, timeout=60).stdout


LONG = "roof appears under-segmented — elevation data shows more roof faces than were detected"


@pytest.mark.parametrize("reason", [
    "roof edge structure not resolved",              # short
    LONG,                                            # the real one that broke it
    "; ".join([LONG] * 3),                           # absurd, must still stamp
])
def test_incomplete_word_is_readable_at_any_reason_length(tmp_path, reason):
    ri = _base()
    ri["incomplete_reason"] = reason
    out = tmp_path / "r.pdf"
    generate_report(ri, str(out))
    assert "INCOMPLETE" in _cover_text(out).upper()


def test_passing_report_carries_no_stamp(tmp_path):
    out = tmp_path / "clean.pdf"
    generate_report(_base(), str(out))
    assert "INCOMPLETE" not in _cover_text(out).upper()
