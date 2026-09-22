# SAM 3 fine-tune on NAIP roof facets

Teach SAM 3 to segment **roof facets** (concept prompt "roof facet") from NAIP,
using your existing COCO facet annotations. Strong pretrained base + your labels.

## 1. Prep data (COCO -> SAM 3 layout)
```
python training/sam3/prep_sam3_facets.py \
  --train-coco <roof_dataset>/train/_annotations.coco.json --train-images <roof_dataset>/train \
  --val-coco   <roof_dataset>/valid/_annotations.coco.json --val-images   <roof_dataset>/valid \
  --out /content/sam3_roof_data
```
Produces `/content/sam3_roof_data/roof/{train,test}/` (facets only, category renamed
to "roof facet", segmentation as RLE).

## 2. Point the config at your paths
Edit `training/sam3/roof_facet_ft.yaml` placeholders:
- `paths.roboflow_vl_100_root: /content/sam3_roof_data`
- `paths.experiment_log_dir: /content/drive/MyDrive/sam3_roof_ft`  (checkpoints land here)
- `paths.bpe_path: /content/bpe_simple_vocab_16e6.txt.gz`

## 3. Train (single GPU, local)
```
cd /content/sam3repo
cp <repo>/training/sam3/roof_facet_ft.yaml sam3/train/configs/roof_facet_ft.yaml
python sam3/train/train.py -c configs/roof_facet_ft.yaml --use-cluster 0 --num-gpus 1
```
Fine-tuned checkpoints -> `experiment_log_dir/checkpoints/`.

## Notes / likely tweaks
- **OOM** on a small GPU: lower `scratch.resolution` 1008 -> 768 or 512.
- Config adapted from `roboflow_v100_full_ft_100_images.yaml` with segmentation
  enabled (Masks + SemanticSeg loss), single-GPU local, checkpoints saved.
- Base weights load from HF (you have SAM3 access + `login()`).
- This is a first, untested config — expect 1-2 Hydra field fixes on first run;
  paste any error and it's a quick fix.
