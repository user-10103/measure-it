"""Unit tests for the Label Studio -> COCO converter and split function."""
import math

from src.data.ls_to_coco import ls_export_to_coco, make_splits, pitch_to_slope_deg

W, H = 1000, 800


def _facet(rid, points, pitch, aspect):
    poly = {"id": rid, "from_name": "facet", "type": "polygonlabels",
            "original_width": W, "original_height": H,
            "value": {"points": points, "polygonlabels": ["facet"]}}
    pitch_r = {"id": rid, "from_name": "pitch", "type": "choices",
               "value": {"choices": [pitch]}}
    aspect_r = {"id": rid, "from_name": "aspect", "type": "choices",
                "value": {"choices": [aspect]}}
    return [poly, pitch_r, aspect_r]


def _make_export():
    outline = {"id": "r0", "from_name": "outline", "type": "polygonlabels",
               "original_width": W, "original_height": H,
               "value": {"points": [[10, 10], [90, 10], [90, 90], [10, 90]],
                         "polygonlabels": ["roof_outline"]}}
    result = [outline]
    result += _facet("r1", [[10, 10], [50, 10], [50, 50], [10, 50]],
                     "medium (15-30 deg)", "NW")
    result += _facet("r2", [[50, 50], [90, 50], [90, 90], [50, 90]],
                     "steep (>30 deg)", "SE")
    return [{"data": {"image": "s3://b/k/addr_001.png"},
             "annotations": [{"result": result}]}]


def test_basic_conversion():
    coco = ls_export_to_coco(_make_export())
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 3
    img = coco["images"][0]
    assert img["address_id"] == "addr_001"
    assert img["file_name"] == "addr_001.png"
    cats = sorted(a["category_id"] for a in coco["annotations"])
    assert cats == [1, 2, 2]


def test_percent_to_pixel():
    coco = ls_export_to_coco(_make_export())
    outline = next(a for a in coco["annotations"] if a["category_id"] == 1)
    # first point (10%, 10%) of 1000x800 -> (100, 80)
    assert math.isclose(outline["segmentation"][0][0], 100.0)
    assert math.isclose(outline["segmentation"][0][1], 80.0)


def test_facet_attributes_and_area():
    coco = ls_export_to_coco(_make_export())
    facets = [a for a in coco["annotations"] if a["category_id"] == 2]
    assert len(facets) == 2
    for a in facets:
        assert "pitch_label" in a["attributes"]
        assert "aspect_label" in a["attributes"]
        assert a["area"] > 0
    by_slope = {a["attributes"]["slope_deg"] for a in facets}
    assert by_slope == {22.5, 45.0}
    nw = next(a for a in facets if a["attributes"]["aspect_label"] == "NW")
    assert nw["attributes"]["slope_deg"] == 22.5


def test_pitch_to_slope():
    assert pitch_to_slope_deg("flat (<5 deg)") == 2.5
    assert pitch_to_slope_deg("low (5-15)") == 10.0
    assert pitch_to_slope_deg(None) is None
    assert pitch_to_slope_deg("unknown") is None


def test_make_splits_determinism():
    ids = [f"a{i:03d}" for i in range(100)]
    s1 = make_splits(ids, seed=42)
    s2 = make_splits(list(reversed(ids)), seed=42)
    assert s1 == s2
    allids = s1["train"] + s1["val"] + s1["test"]
    assert sorted(allids) == sorted(ids)         # cover all
    assert len(set(allids)) == 100               # disjoint
    assert len(s1["train"]) == 80
    assert len(s1["val"]) == 10
    assert len(s1["test"]) == 10
