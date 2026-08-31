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


def test_prep_basename_collision_across_batches(tmp_path):
    # Two batches, SAME basename, DISTINCT pixels. A basename-flat copy would
    # overwrite one; the fix must keep both with correct pixel pairing.
    import sys; sys.path.insert(0, "training/sam3")
    from prep_sam3_facets import prep_split
    imdir = tmp_path / "imgs"
    for batch, val in (("batch2_miami", 40), ("batch3_broward", 200)):
        (imdir / batch).mkdir(parents=True)
        Image.fromarray(np.full((60, 60, 3), val, "uint8")).save(imdir / batch / "roof_001.png")
    seg = [[5, 5, 50, 5, 50, 50, 5, 50]]
    coco = {"categories": [{"id": 0, "name": "facet"}, {"id": 1, "name": "roof_outline"}],
            "images": [
                {"id": 1, "file_name": "batch2_miami/roof_001.png", "height": 60, "width": 60},
                {"id": 2, "file_name": "batch3_broward/roof_001.png", "height": 60, "width": 60}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "segmentation": seg, "bbox": [5, 5, 45, 45], "area": 2025, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 0, "segmentation": seg, "bbox": [5, 5, 45, 45], "area": 2025, "iscrowd": 0}]}
    cj = tmp_path / "coco.json"; json.dump(coco, open(cj, "w"))
    out_dir = tmp_path / "out/roof/train"
    ni, na = prep_split(str(cj), str(imdir), str(out_dir), "roof facet")
    assert ni == 2 and na == 2                              # both images survive
    out = json.load(open(out_dir / "_annotations.coco.json"))
    names = sorted(im["file_name"] for im in out["images"])
    assert names == ["batch2_miami__roof_001.png", "batch3_broward__roof_001.png"]  # collision-safe
    # pixel pairing intact: each output file holds its own batch's value
    for im in out["images"]:
        px = np.asarray(Image.open(out_dir / im["file_name"]))[0, 0, 0]
        assert px == (40 if "batch2" in im["file_name"] else 200), "pixels crossed!"


def test_prep_keep_matches_normalized_and_raw(tmp_path):
    # keep manifest may hold raw LS paths OR normalized <batch>/<name>; prep must
    # match either against the split's file_names (else keep_ids empties silently).
    import sys; sys.path.insert(0, "training/sam3")
    from prep_sam3_facets import prep_split
    imdir = tmp_path / "imgs"; (imdir / "batch1_3inch").mkdir(parents=True)
    Image.fromarray(np.zeros((40, 40, 3), "uint8")).save(imdir / "batch1_3inch" / "a.png")
    coco = {"categories": [{"id": 0, "name": "facet"}],
            "images": [{"id": 1, "file_name": "batch1_3inch/a.png", "height": 40, "width": 40}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0,
                             "segmentation": [[1, 1, 30, 1, 30, 30]], "bbox": [1, 1, 29, 29], "area": 400, "iscrowd": 0}]}
    cj = tmp_path / "coco.json"; json.dump(coco, open(cj, "w"))
    # keep given as a RAW LS path must still match the normalized split name
    ni, na = prep_split(str(cj), str(imdir), str(tmp_path / "out/roof/train"), "roof facet",
                        keep_names=["../../label-studio/data/local/batch1_3inch/a.png"])
    assert ni == 1 and na == 1                              # matched, not silently dropped
