"""Training-data readiness gate: coverage + quality + keep-manifest."""
from training.label_readiness import assess

# a 4x4 square footprint with two clean triangular facets = train-ready;
# one roof with NO facets; one roof with a jagged facet.
def _sq(o=0, s=4):
    return [o, o, o + s, o, o + s, o + s, o, o + s]

COCO = {
    "categories": [{"id": 7, "name": "roof_outline"}, {"id": 3, "name": "facet"},
                   {"id": 11, "name": "edge_ridge"}],
    "images": [{"id": 1, "file_name": "good.tif", "width": 4, "height": 4},
               {"id": 2, "file_name": "nofacets.tif", "width": 4, "height": 4}],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 7, "segmentation": [_sq()], "bbox": [0, 0, 4, 4]},
        # three non-overlapping vertical thirds = a clean 3-facet partition
        {"id": 2, "image_id": 1, "category_id": 3, "segmentation": [[0, 0, 1.34, 0, 1.34, 4, 0, 4]], "bbox": [0, 0, 1.34, 4]},
        {"id": 3, "image_id": 1, "category_id": 3, "segmentation": [[1.34, 0, 2.67, 0, 2.67, 4, 1.34, 4]], "bbox": [1.34, 0, 1.33, 4]},
        {"id": 4, "image_id": 1, "category_id": 3, "segmentation": [[2.67, 0, 4, 0, 4, 4, 2.67, 4]], "bbox": [2.67, 0, 1.33, 4]},
        {"id": 5, "image_id": 1, "category_id": 11, "segmentation": [[0, 2, 4, 2]], "bbox": [0, 2, 4, 1]},
        # image 2: roof only, no facets
        {"id": 6, "image_id": 2, "category_id": 7, "segmentation": [_sq()], "bbox": [0, 0, 4, 4]},
    ],
}


def test_coverage_and_keep():
    a = assess(COCO)
    assert a["roofs"] == 2
    assert a["facet_coverage_pct"] == 50.0           # 1 of 2 roofs has facets
    assert a["failure_reasons"].get("no_facets") == 1
    # the facetted roof passes and appears in the keep list by file_name
    assert "good.tif" in a["keep_ids"]
    assert "nofacets.tif" not in a["keep_ids"]


def test_edge_and_fo_coverage():
    a = assess(COCO)
    assert a["edge_coverage_pct"] == 50.0            # 1 roof has an edge
    assert a["fo_coverage_pct"] == 0.0               # no foreign objects
