"""End-to-end test of the RGB-only pipeline: COCO stand-in -> report PDF."""
import json
import os

from src.roofs.segment import CocoStandinBackend
from src.rgb_pipeline import process_address_id


def _gable_coco():
    """A 10x10 m gable (two facets sharing a ridge) as a COCO stand-in."""
    return {
        "images": [{"id": 1, "address_id": "g1", "width": 33, "height": 33}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "segmentation": [[0, 0, 33, 0, 33, 33, 0, 33]]},
            {"id": 2, "image_id": 1, "category_id": 2,
             "segmentation": [[0, 0, 33, 0, 33, 16, 0, 16]],
             "attributes": {"slope_deg": 20.0, "aspect_label": "S"}},
            {"id": 3, "image_id": 1, "category_id": 2,
             "segmentation": [[0, 16, 33, 16, 33, 33, 0, 33]],
             "attributes": {"slope_deg": 20.0, "aspect_label": "N"}},
        ],
    }


def test_rgb_pipeline_end_to_end(tmp_path):
    backend = CocoStandinBackend(_gable_coco())
    result = process_address_id("g1", backend, address="1 Gable St",
                                gsd_m_per_px=0.3, output_dir=str(tmp_path))

    summary = result["summary"]
    assert summary["num_facets"] == 2
    # two ~20deg facets -> predominant pitch is pitched, not flat
    assert summary["predominant_pitch"] != "0:12"
    assert summary["total_area_sqft"] > 0

    # the gable produced a ridge + perimeter eaves/rakes
    edge_types = {e["edge_type"] for e in result["report_input"]["edges"]}
    assert "ridge" in edge_types

    # report artifacts written
    files = result["output_files"]
    assert os.path.getsize(files["pdf"]) > 1000
    data = json.load(open(files["json"]))
    assert data["summary"]["num_facets"] == 2
    assert os.path.exists(files["facets_csv"])
    assert os.path.exists(files["edges_csv"])
