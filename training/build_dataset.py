#!/usr/bin/env python3
"""
Multi-source dataset builder for RF-DETR training.
Downloads phase1+carecamp93+switzerland from S3, merges into roof_dataset_v2.
IDEMPOTENT - safe to re-run.
Usage: python training/build_dataset.py
"""
# ============================================================
# CELL 2 — Build Full Dataset (ALL sources)
# Sources:  phase1      ~2710 imgs  (NAIP Florida chips)
#           carecamp93  ~1143 imgs  (Florida carecamp chips)
#           switzerland ~7668 imgs  (Switzerland buildings)
# Downloads annotations from S3, downloads missing images,
# merges all into roof_dataset_v2 with one COCO JSON per split.
# IDEMPOTENT — skips sources already present on disk.
# ============================================================
import boto3, json, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.chdir('/content/measure-it')

S3_BUCKET  = 'florida-roofs-v4'
DATASET    = Path('training/roof_dataset_v2')
s3         = boto3.client('s3')

# ── Source definitions ────────────────────────────────────
# Each entry: (name, ann_key_template, img_s3_prefix, img_s3_bucket)
SOURCES = [
    {
        'name':    'phase1',
        'ann':     {'train': 'annotations/phase1/train/_annotations.coco.json',
                    'valid': 'annotations/phase1/valid/_annotations.coco.json',
                    'test':  'annotations/phase1/test/_annotations.coco.json'},
        'img_bucket': S3_BUCKET,
        'img_prefix': 'phase1/',          # phase1/{filename}
    },
    {
        'name':    'carecamp93',
        'ann':     {'train': 'carecamp93/annotations/train.json',
                    'valid': 'carecamp93/annotations/valid.json',
                    'test':  'carecamp93/annotations/test.json'},
        'img_bucket': S3_BUCKET,
        'img_prefix': 'carecamp93/',      # carecamp93/{filename}
    },
    {
        'name':    'switzerland',
        'ann':     {'train': 'switzerland/annotations/train.json',
                    'valid': 'switzerland/annotations/valid.json',
                    'test':  None},        # no test split
        'img_bucket': S3_BUCKET,
        'img_prefix': 'switzerland/',     # switzerland/{filename}
    },
]

# ── Helpers ───────────────────────────────────────────────
def download_img(args):
    bucket, key, dst = args
    if Path(dst).exists():
        return 'skip'
    try:
        s3.download_file(bucket, key, str(dst))
        return 'ok'
    except Exception as e:
        return f'err:{e}'

def load_coco(bucket, key):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj['Body'].read())
    except Exception as e:
        print(f'  WARN: could not load {key}: {e}')
        return None

# ── Per-split merge ───────────────────────────────────────
for split in ['train', 'valid', 'test']:
    out_dir  = DATASET / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / '_annotations.coco.json'

    merged = {'info': {}, 'licenses': [], 'categories': [], 'images': [], 'annotations': []}
    next_img_id = 1
    next_ann_id = 1
    seen_files  = set()

    for src in SOURCES:
        ann_key = src['ann'].get(split)
        if ann_key is None:
            continue

        print(f'  [{split}] loading {src["name"]} annotations...')
        coco = load_coco(S3_BUCKET, ann_key)
        if coco is None:
            continue

        # Adopt categories from first source (all identical)
        if not merged['categories']:
            merged['categories'] = coco.get('categories', [])

        # Remap IDs and collect download tasks
        tasks = []
        id_map = {}
        for img in coco.get('images', []):
            fname = img['file_name']
            dst   = out_dir / fname
            if fname in seen_files:
                continue          # deduplicate across sources
            seen_files.add(fname)

            s3_key = src['img_prefix'] + fname
            tasks.append((src['img_bucket'], s3_key, str(dst)))

            new_img    = dict(img)
            new_img['id'] = next_img_id
            id_map[img['id']] = next_img_id
            merged['images'].append(new_img)
            next_img_id += 1

        for ann in coco.get('annotations', []):
            if ann['image_id'] not in id_map:
                continue
            new_ann = dict(ann)
            new_ann['id']       = next_ann_id
            new_ann['image_id'] = id_map[ann['image_id']]
            merged['annotations'].append(new_ann)
            next_ann_id += 1

        # Download missing images (parallel, 32 workers)
        to_download = [(b, k, d) for b, k, d in tasks if not Path(d).exists()]
        if to_download:
            print(f'  [{split}] downloading {len(to_download)} {src["name"]} images...')
            ok = skip = err = 0
            with ThreadPoolExecutor(max_workers=32) as pool:
                futures = {pool.submit(download_img, t): t for t in to_download}
                for fut in as_completed(futures):
                    r = fut.result()
                    if r == 'ok':    ok   += 1
                    elif r == 'skip': skip += 1
                    else:            err  += 1
                    if (ok + err) % 200 == 0:
                        print(f'    {ok} downloaded, {err} failed...')
            print(f'  [{split}] {src["name"]}: {ok} downloaded, {err} failed')
        else:
            print(f'  [{split}] {src["name"]}: all images already on disk')

    # Write merged COCO JSON
    with open(out_json, 'w') as f:
        json.dump(merged, f)
    print(f'\n[{split}] DONE: {len(merged["images"])} images, '
          f'{len(merged["annotations"])} annotations → {out_json}\n')

# ── Final summary ─────────────────────────────────────────
print('\n' + '='*60)
print('DATASET SUMMARY')
print('='*60)
for split in ['train', 'valid', 'test']:
    out_dir  = DATASET / split
    n_pngs   = len(list(out_dir.glob('*.png')))
    coco_f   = out_dir / '_annotations.coco.json'
    if coco_f.exists():
        d = json.loads(coco_f.read_text())
        print(f'  {split:5s}  {n_pngs:5d} PNGs on disk  '
              f'{len(d["images"]):5d} in COCO  {len(d["annotations"]):6d} annotations')
    else:
        print(f'  {split:5s}  {n_pngs:5d} PNGs  NO COCO JSON')
print('\n✓ Run Cell 3 to start training.')
