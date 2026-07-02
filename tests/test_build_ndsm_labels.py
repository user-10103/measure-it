"""Local test for the label factory geometry (world facets -> chip pixels).
Writes synthetic NAIP + nDSM GeoTIFFs (no LiDAR/network) and checks COCO output."""
import json
import numpy as np
import rasterio
from rasterio.transform import Affine


def _write_tif(path, arr, tf, count):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[-2],
                       width=arr.shape[-1], count=count, dtype=arr.dtype,
                       crs="EPSG:26917", transform=tf) as d:
        d.write(arr if count > 1 else arr[None], indexes=list(range(1, count + 1)))


def test_chip_to_coco_pixels(tmp_path):
    from training.build_ndsm_labels import chip_to_coco
    H = W = 80
    yy, xx = np.mgrid[0:H, 0:W]
    z = (0.25 * np.minimum.reduce([xx, W - 1 - xx, yy, H - 1 - yy])).astype("float32")
    tf = Affine(0.5, 0, 356000, 0, -0.5, 3111500)

    work = tmp_path / "work"; work.mkdir()
    _write_tif(str(work / "bldg_ndsm.tif"), z, tf, 1)
    naip = np.zeros((3, H, W), "uint8"); naip[:] = 120
    _write_tif(str(work / "naip.tif"), naip, tf, 3)   # same grid as nDSM
    json.dump({"naip": {"tif_path": str(work / "naip.tif")}},
              open(work / "results.json", "w"))

    images_dir = tmp_path / "images"; images_dir.mkdir()
    r = chip_to_coco(str(work), 1, str(images_dir))
    assert r is not None
    assert r["image"]["width"] == W and r["image"]["height"] == H
    assert len(r["annotations"]) >= 4                 # 4 hip facets
    for a in r["annotations"]:
        pts = np.array(a["segmentation"][0]).reshape(-1, 2)
        assert (pts[:, 0] >= -2).all() and (pts[:, 0] <= W + 2).all()   # x in-frame
        assert (pts[:, 1] >= -2).all() and (pts[:, 1] <= H + 2).all()   # y in-frame
        assert a["category_id"] == 2 and a["area"] > 0
    assert (images_dir / "000001.png").exists()
