"""SAM3 data prep: facet-filter + concept rename + polygon->RLE."""
import json, os
import numpy as np
from PIL import Image


def test_prep_split(tmp_path):
    import sys; sys.path.insert(0, "training/sam3")
    from prep_sam3_facets import prep_split
    imdir = tmp_path / "imgs"; imdir.mkdir()
    Image.fromarray(np.zeros((80, 80, 3), "uint8")).save(imdir / "r1.png")
    coco = {"images": [{"id": 1, "file_name": "r1.png", "height": 80, "width": 80}],
            "categories": [{"id": 1, "name": "roof_polygon"}, {"id": 2, "name": "facet"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[0,0,80,0,80,80]], "bbox": [0,0,80,80], "area": 6400, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 2, "segmentation": [[10,10,40,10,40,40,10,40]], "bbox": [10,10,30,30], "area": 900, "iscrowd": 0}]}
    cj = tmp_path / "coco.json"; json.dump(coco, open(cj, "w"))
    ni, na = prep_split(str(cj), str(imdir), str(tmp_path / "out/roof/train"), "roof facet")
    assert ni == 1 and na == 1                        # roof_polygon dropped, facet kept
    out = json.load(open(tmp_path / "out/roof/train/_annotations.coco.json"))
    assert out["categories"] == [{"id": 1, "name": "roof facet", "supercategory": "roof facet"}]
    a = out["annotations"][0]
    assert a["category_id"] == 1
    assert isinstance(a["segmentation"], dict) and isinstance(a["segmentation"]["counts"], str)  # RLE
