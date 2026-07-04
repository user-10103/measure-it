"""Prove the SAM->report bridge produces a real PDF from facet polygons."""
import os

import pytest
from shapely.geometry import Polygon, box

from src.roofs.sam_segment import SamRoof
from src.roofs.segment import Facet


def _sam_roof():
    # a hip-roof-ish partition of a 20x20 outline into 4 facets that tile it
    outline = box(0, 0, 20, 20)
    facets = [
        Facet(facet_id=1, polygon=Polygon([(0, 0), (20, 0), (14, 10), (6, 10)])),   # front
        Facet(facet_id=2, polygon=Polygon([(0, 20), (20, 20), (14, 10), (6, 10)])),  # back
        Facet(facet_id=3, polygon=Polygon([(0, 0), (0, 20), (6, 10)])),              # left
        Facet(facet_id=4, polygon=Polygon([(20, 0), (20, 20), (14, 10)])),           # right
    ]
    return SamRoof(outline=outline, facets=facets, label_map=None, georeferenced=True)


def test_report_input_shape():
    from src.output.sam_report import facets_to_report_input
    ri = facets_to_report_input(_sam_roof(), "123 Test St")
    assert ri["address"] == "123 Test St"
    assert len(ri["facets"]) == 4
    # every facet carries the fields the report/diagram read
    f = ri["facets"][0]
    for k in ("facet_id", "polygon_xy", "plan_area_m2", "pitch_string", "is_flat"):
        assert k in f
    assert f["plan_area_m2"] > 0
    assert len(f["polygon_xy"]) >= 4                 # closed ring
    # outline + facet seams -> typed edge graph (eaves + internal ridge/hip)
    assert len(ri["edges"]) >= 4
    types = {e["edge_type"] for e in ri["edges"]}
    assert "eave" in types and (types & {"ridge", "hip", "valley"})
    assert all(e["length_m"] > 0 for e in ri["edges"])
    assert ri["outline_xy"]                           # bold outline present


def test_generates_a_pdf(tmp_path):
    pytest.importorskip("reportlab")
    from src.output.sam_report import sam_roof_to_pdf
    out = str(tmp_path / "roof_report.pdf")
    sam_roof_to_pdf(_sam_roof(), "123 Test St", out)
    assert os.path.exists(out)
    # a real multi-page PDF, not an empty file
    assert os.path.getsize(out) > 2000
    with open(out, "rb") as fh:
        head = fh.read(5)
    assert head == b"%PDF-"


def test_empty_roof_still_valid_input():
    from src.output.sam_report import facets_to_report_input
    ri = facets_to_report_input(SamRoof(outline=None, facets=[]), "nowhere")
    assert ri["facets"] == [] and ri["edges"] == []
