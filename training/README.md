# RF-DETR-Seg facet model — RunPod training

Fine-tunes **RF-DETR-Seg** (Apache-2.0, DINOv2 backbone) to predict **roof_outline + facet** instance masks from an aerial chip. Pitch and aspect are **not** trained — the measure-it deliverable applies the pitch *policy* (4:12 prior + flat-detect) and derives aspect *geometrically*, so this is a clean 2-class instance-seg fine-tune on ~2,400 roofs.

## Pipeline

```
LS export ──build_dataset.py──▶ roof_dataset/{train,valid,test}/_annotations.coco.json
              (LOCAL, CPU)        + chips_needed.txt
roof_dataset ──fetch_chips.py──▶ chip PNGs into each split dir   (LOCAL or RunPod, needs S3)
roof_dataset ──train_rfdetr.py─▶ output/ weights                 (RunPod GPU)
```

## 1. Build the dataset (local)

```bash
PYTHONPATH=/home/salter/Desktop/measure-it/measure-it-main \
  python build_dataset.py \
    --ls /path/to/project-1-export.json \
    --out roof_dataset
```
Produces `roof_dataset/{train,valid,test}/_annotations.coco.json` (2,392 / 299 / 300 images) + per-split `chips_needed.txt`. Already verified locally.

## 2. Fetch chip images

Chips live at `s3://<bucket>/phase1/<chip_id>.png`. Use the instance role or AWS
env credentials — **no keys in the repo**.

```bash
python fetch_chips.py --bucket florida-roofs-v2-chips --prefix phase1 --dataset roof_dataset
```
(You can run this locally and upload `roof_dataset/` to the pod, or run it on the pod.)

## 3. Train on RunPod

- **Pod:** a PyTorch 2.x + CUDA template, 1× GPU. A24–48 GB card (L4 / A40 / A6000)
  is plenty for ~2,400 512-px images. RF-DETR-Seg Nano/Small fits comfortably.
- Upload `roof_dataset/` (or fetch chips on the pod), then:

```bash
pip install -r requirements.txt
python train_rfdetr.py --dataset roof_dataset --output output --epochs 50 --batch-size 4
```

Start with the smallest seg variant; scale up only if val mask-AP is short. Watch
val AP on the `valid` split; the held-out `test` split is for the final read.

## 4. Wire the weights back into measure-it

The trained model replaces `CocoStandinBackend`. Implement a `RoofModelBackend`
in `measure-it/src/roofs/segment.py` whose `predict_for()` returns
`{"outline": [...], "facets": [{"polygon": [...]}]}` (pixel coords) from the
RF-DETR-Seg masks. Everything downstream (tiling → geometric aspect → pitch
policy → typed edges → report map) already works on that contract.

## Licensing

RF-DETR-Seg segmentation tiers are Apache-2.0 (only the XL/2XL *detection*
checkpoints are PML 1.0 — not used here). Verify the specific checkpoint's weight
license tag on Hugging Face before shipping the paid deliverable.
