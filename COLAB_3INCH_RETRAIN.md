# Colab Runbook — Retrain RF-DETR-Seg on **real 3-inch** county imagery

**For:** an agent operating Google Colab in a browser.
**Goal:** test whether real 7.6 cm (3-inch) county aerial imagery breaks the recall ceiling that capped the roof-facet model on blurry imagery — then ship the number.

---

## 0. The one-paragraph why (read this first)

The facet/outline model (RF-DETR-Seg) was trained on chips that are **0.3 m NAIP fake-upscaled ×4 with RealESRGAN** to ~0.075 m. It stalled at facet AP50-95 ≈ **0.26** and F1 ≈ **0.51** — recall-limited, because the "detail" in those chips is hallucinated, not real. Florida counties publish **real 3-inch orthoimagery** (Pinellas/Hillsborough/Pasco), free, public-record. This job **re-fetches each annotated roof at real 3-inch over the identical ground footprint**, keeps the existing labels (same framing → labels barely change), retrains, and re-evaluates. It is a clean A/B: **only the pixels change — hallucinated detail → real detail.** If F1 jumps, the data lever is proven and the next step is label-sharpening; if it's flat, the ceiling is the labels, not resolution.

**Decision gate at the end:** facet F1 on the held-out split vs the 0.51 NAIP baseline (and vs the 0.85 grade target).

---

## 1. What you need before starting

| Item | Where | Notes |
|---|---|---|
| `measure-it` repo | github `user-10103/measure-it` (private) | clone via PAT, or have the user upload a zip |
| `state.db` | `florida_roofs_v2/state.db` | the v2 pipeline DB — chip_id → lat/lon, GeoTIFF paths. Upload to Colab or have it in Drive |
| Geo-referenced source chips | Drive `MyDrive/florida_roofs_v2/chips/raw/{id}.tif` (0.3 m NAIP, EPSG:26917) | used ONLY to read each chip's exact ground bbox; optional if you reconstruct from footprint |
| Clean label splits | **in the repo:** `training/roof_dataset_clean/{train,valid,test}/_annotations.coco.json` | 764/95/97 unique chips, disjoint. These are the labels we reuse |
| Colab runtime | GPU: **T4 (free) works**; A100 (Pro+) ~5× faster | RF-DETR-Seg at resolution 512 |

> **Runtime:** Runtime → Change runtime type → **GPU**. Confirm with `!nvidia-smi`.

> **County reachability (important):** Pinellas (`egis.pinellas.gov`) is reachable everywhere. Hillsborough (`maps.hillsboroughcounty.org`) and Pasco (`pascogis.pascocountyfl.net`) **block some datacenter IPs**. Colab usually reaches them, but **verify in Phase 0** — if blocked, fall back to the statewide 6-inch (FCDOP) source for those counties, still a big lift over 0.3 m.

---

## 2. Phase 0 — Setup & verify (do not skip the checks)

### 2.1 Mount Drive, clone repo, install
```python
from google.colab import drive
drive.mount('/content/drive')

# Clone the private repo (replace TOKEN, or upload a zip instead)
# !git clone https://<TOKEN>@github.com/user-10103/measure-it.git
%cd /content/measure-it

!pip -q install rfdetr rasterio shapely pillow numpy supervision pycocotools
!nvidia-smi -L
```

### 2.2 Self-contained 3-inch fetcher (works even if county_imagery.py isn't committed)
```python
import io, math, json, urllib.request, urllib.parse
_UA = {"User-Agent": "Mozilla/5.0"}

# county -> ImageServer (3-inch / 7.6cm), with statewide 6-inch fallback
ENDPOINTS = {
  "Pinellas":     ("https://egis.pinellas.gov/gis/rest/services/Aerials/Aerials2024/ImageServer", 0.0762),
  "Hillsborough": ("https://maps.hillsboroughcounty.org/arcgis/rest/services/AerialsNew/Aerials2025_3_inch_MrSid/ImageServer", 0.0762),
  "Pasco":        ("https://pascogis.pascocountyfl.net/giswebi/rest/services/Aerials/Aerials2023/ImageServer", 0.0762),
}
FCDOP = ("https://ca.dep.state.fl.us/arcgis/rest/services/Imagery/Aerial_Imagery_2019/ImageServer", 0.15)

def export_bbox(base, xmin, ymin, xmax, ymax, bbox_sr, w, h, timeout=60):
    """Fetch a PNG over an EXACT bbox (in bbox_sr) at w x h pixels."""
    q = urllib.parse.urlencode({"bbox": f"{xmin},{ymin},{xmax},{ymax}", "bboxSR": bbox_sr,
                                "size": f"{w},{h}", "imageSR": bbox_sr, "format": "png", "f": "image"})
    req = urllib.request.Request(f"{base}/exportImage?{q}", headers=_UA)
    png = urllib.request.urlopen(req, timeout=timeout).read()
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"non-PNG (likely error/blocked): {png[:120]!r}")
    return png
```

### 2.3 VERIFY reachability + resolution before processing 956 chips
```python
# Pinellas (should always work), then Hillsborough + Pasco (may be blocked)
tests = {
  "Pinellas":     (ENDPOINTS["Pinellas"][0],     -82.79, 27.84),
  "Hillsborough": (ENDPOINTS["Hillsborough"][0], -82.46, 27.95),
  "Pasco":        (ENDPOINTS["Pasco"][0],        -82.46, 28.23),
}
import math
for name,(base,lon,lat) in tests.items():
    try:
        h=25; dlat=h/111320; dlon=h/(111320*math.cos(math.radians(lat)))
        png=export_bbox(base, lon-dlon, lat-dlat, lon+dlon, lat+dlat, 4326, 600, 600)
        print(f"{name:13s} OK  ({len(png)} bytes)")
    except Exception as e:
        print(f"{name:13s} BLOCKED -> use FCDOP fallback  ({str(e)[:60]})")
```
**Decision:** counties that print `BLOCKED` → set their entry to `FCDOP` in `ENDPOINTS` (6-inch is still 5× better than 0.3 m). Counties that print `OK` → keep 3-inch.

---

## 3. Phase 1 — Build the 3-inch dataset (re-fetch + reuse labels)

The labels live in `training/roof_dataset_clean/{split}/_annotations.coco.json` with `image.width/height` (e.g. 800×756) and segmentation polygons in that pixel space. For each chip we (a) find its **exact ground bbox**, (b) fetch real 3-inch over that **same bbox** at the **same pixel size**, (c) keep the polygons (scale only if dims differ). Same framing → the labels stay valid.

### 3.1 Per-chip ground bbox
Preferred: read the GeoTIFF bounds (exact). Fallback: reconstruct from chip center + size.
```python
import sqlite3, rasterio
from rasterio.warp import transform as warp_xy

DB = "/content/drive/MyDrive/florida_roofs_v2/state.db"   # adjust to where you put it
RAW_TIF = "/content/drive/MyDrive/florida_roofs_v2/chips/raw/{cid}.tif"
con = sqlite3.connect(DB)

def chip_geo(cid, png_w, png_h):
    """Return (xmin,ymin,xmax,ymax, bbox_sr, lon,lat). Prefer GeoTIFF bounds."""
    try:
        with rasterio.open(RAW_TIF.format(cid=cid)) as r:
            b = r.bounds; sr = int(r.crs.to_epsg())
            lon,lat = warp_xy(r.crs, "EPSG:4326", [(b.left+b.right)/2], [(b.top+b.bottom)/2])
            return b.left, b.bottom, b.right, b.top, sr, lon[0], lat[0]
    except Exception:
        # fallback: center from geocode_results, assume 0.3m * raw_px extent
        lon,lat = con.execute("select lon,lat from geocode_results where id=?", (cid,)).fetchone()
        # raw NAIP was 0.3m; png is the x4 upscale, so ground extent = png_dims/4 * 0.3m
        half_w = (png_w/4*0.3)/2; half_h = (png_h/4*0.3)/2
        dlat=half_h/111320; dlon=half_w/(111320*math.cos(math.radians(lat)))
        return lon-dlon, lat-dlat, lon+dlon, lat+dlat, 4326, lon, lat
```

### 3.2 Route a chip to its county
`state.db` doesn't store county; derive it from the address city (fast) with a lat/lon fallback.
```python
def chip_county(cid, lat):
    row = con.execute("select raw_address from addresses where id=?", (cid,)).fetchone()
    addr = (row[0] if row else "").lower()
    PIN = ("st petersburg","saint petersburg","clearwater","largo","seminole","pinellas park",
           "dunedin","tarpon springs","palm harbor","safety harbor","oldsmar","gulfport","st pete")
    HILL = ("tampa","brandon","riverview","valrico","lutz","plant city","ruskin","seffner",
            "thonotosassa","apollo beach","wimauma","gibsonton","odessa","dover")
    PAS = ("land o lakes","new port richey","wesley chapel","zephyrhills","hudson","port richey",
           "dade city","holiday","spring hill","trinity","odessa")
    if any(c in addr for c in PIN):  return "Pinellas"
    if any(c in addr for c in HILL): return "Hillsborough"
    if any(c in addr for c in PAS):  return "Pasco"
    return None   # -> FCDOP fallback
```

### 3.3 Build each split
```python
import os
from PIL import Image
SRC = "training/roof_dataset_clean"
DST = "training/roof_dataset_3inch"

def build_split(split):
    sd = f"{SRC}/{split}"; dd = f"{DST}/{split}"; os.makedirs(dd, exist_ok=True)
    coco = json.load(open(f"{sd}/_annotations.coco.json"))
    ok = miss = fcdop = 0
    for im in coco["images"]:
        cid = os.path.splitext(im["file_name"])[0]; W,H = im["width"], im["height"]
        try:
            xmin,ymin,xmax,ymax,sr,lon,lat = chip_geo(cid, W, H)
            cty = chip_county(cid, lat)
            base,gsd = ENDPOINTS.get(cty, FCDOP)
            if cty is None or ENDPOINTS.get(cty, FCDOP) is FCDOP: fcdop += 1
            png = export_bbox(base, xmin, ymin, xmax, ymax, sr, W, H)   # SAME WxH -> labels unchanged
            open(f"{dd}/{im['file_name']}", "wb").write(png)
            ok += 1
        except Exception as e:
            miss += 1
            if miss <= 5: print("  miss", cid, str(e)[:70])
    json.dump(coco, open(f"{dd}/_annotations.coco.json","w"))   # labels reused as-is
    print(f"{split}: fetched {ok}, missed {miss}, of which fcdop-fallback {fcdop}")

for s in ("valid","test","train"):   # valid/test first — fast feedback
    build_split(s)
```
> **Sanity check before training:** open 3–4 fetched chips next to their polygons (overlay the segmentation) and confirm the labels still sit on the roofs. If they're shifted, the bbox source is wrong — debug `chip_geo` on those chips before proceeding. **Do not train on misaligned labels.**

```python
# quick overlay check
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
coco = json.load(open(f"{DST}/valid/_annotations.coco.json"))
for im in coco["images"][:3]:
    img = Image.open(f"{DST}/valid/{im['file_name']}")
    anns = [a for a in coco["annotations"] if a["image_id"]==im["id"]]
    fig,ax = plt.subplots(figsize=(5,5)); ax.imshow(img)
    for a in anns:
        s=a["segmentation"][0]; ax.add_patch(MplPoly(np.array(s).reshape(-1,2), fill=False, ec="lime", lw=1.5))
    ax.set_title(im["file_name"]); plt.show()
```

---

## 4. Phase 2 — Train (mind the three gotchas)

```python
# Checkpoint to Drive so a Colab disconnect doesn't lose the run
OUT = "/content/drive/MyDrive/florida_roofs_v2/runs/rfdetr_3inch"
import os; os.makedirs(OUT, exist_ok=True)

!python training/train_rfdetr.py \
    --dataset training/roof_dataset_3inch \
    --output "{OUT}" \
    --epochs 50 \
    --batch-size 1 \
    --grad-accum 16 \
    --resolution 512
```

**The three gotchas — all already baked into `train_rfdetr.py`, do not "fix" them:**
1. **`--batch-size 1` is mandatory.** Chips are variable-sized; DETR can't collate a batch of different-sized images (the `Expected 756, got 832` tensor crash). Use `--grad-accum 16` for an effective batch of 16. **Never raise batch-size.**
2. **Colab is ephemeral.** Output dir is on Drive. If it disconnects, re-run the same cell — it resumes from the last Drive checkpoint.
3. **Watch the *best* facet AP, not train loss.** The number that matters is held-out facet AP50-95. On the old NAIP data it froze at **0.2588** by ep3 then overfit. **The whole point of this run is whether real 3-inch pushes that past ~0.4.**

**Speed:** T4 ≈ 6 min/epoch (~5 h for 50). If on Pro+ A100, it's ~1 h. You can stop early once `best` clearly plateaus (see the decision gate).

**Optional resolution lever (only if T4 memory allows / on A100):** add `--resolution 728`. Real 3-inch carries detail beyond 512px; 728 lets the model use more of it. Must be divisible by 56. Start with 512 to match the baseline, then try 728 as a second run if 512 improves.

---

## 5. Phase 3 — Eval + decision gate (the deliverable number)

```python
!python training/eval_model.py \
    --checkpoint "{OUT}/checkpoint_best_regular.pth" \
    --gt    training/roof_dataset_3inch/valid/_annotations.coco.json \
    --chips training/roof_dataset_3inch/valid \
    --out-pred "{OUT}/pred_valid.json"
```
This prints **facet F1 (IoU≥0.5), outline IoU, plan-area %** on the held-out split — model-vs-annotation geometry, no leak.

### Decision matrix
| Result on held-out facet F1 | Read | Next move |
|---|---|---|
| **≥ 0.85** | Hits grade | Ship: wire `checkpoint_best_regular.pth` into `RFDETRBackend`, run real address→report end-to-end |
| **0.6–0.85** (up from 0.51) | Real detail helped; labels now the ceiling | Do a **3-inch re-annotation pass** in Label Studio on a few hundred chips, retrain |
| **≈ 0.51, flat** | Resolution wasn't the bottleneck — labels were | Stop tuning resolution; the lever is label quality / more annotation |
| **< 0.45** | Something broke | Check label alignment (Phase 3.1 overlay), county-fallback ratio, chip count |

Also render a few honest maps to eyeball (optional, needs the repo's render path):
```python
# spot-check 5 reports end-to-end on the 3-inch chips
# (uses RFDETRBackend -> process_chip_rgb at gsd_m_per_px=0.0762)
```

---

## 6. Report back to the user (paste this filled in)
```
3-INCH RETRAIN RESULT
- counties reachable from Colab: Pinellas[ ] Hillsborough[ ] Pasco[ ]   (blocked -> FCDOP count: __)
- dataset built: train __/764  valid __/95  test __/97   (missed: __)
- label-alignment overlay checked: yes/no
- best facet AP50-95 @ epoch __ :  ____   (NAIP baseline 0.2588)
- eval_model.py held-out:  facet F1 ____  | outline IoU ____ | area% ____
   (NAIP baseline: F1 0.51 / IoU 0.656 / area 49.7% ; grade: 0.85 / 0.90 / 5%)
- VERDICT (per decision matrix): ____
- checkpoint saved at: <Drive path>
```

---

## 7. Troubleshooting quick-reference
- **`non-PNG (likely error/blocked)`** → that county blocks Colab's IP; set it to `FCDOP` in `ENDPOINTS`.
- **`Expected NNN, got MMM` crash** → batch-size got raised above 1. Set it back to 1.
- **Labels look shifted on the overlay** → `chip_geo` returned the wrong bbox (GeoTIFF missing → fell back to reconstruct, which assumes 0.3m×4). Fix the GeoTIFF path or the assumed GSD before training.
- **Run died mid-epoch** → re-run the train cell; it resumes from the Drive checkpoint.
- **All chips fall to FCDOP** → `chip_county` city lists didn't match; print a few `addresses.raw_address` and extend the city tuples, or add a true point-in-polygon county lookup.
- **AP climbs then declines** → small-data overfit; that's the signal the lever is annotation volume/quality, not more epochs.

---

### Provenance (why this is trustworthy)
- 3-inch sources verified 2026-06-08: Pinellas `Aerials2024` pixelSize = **0.0762 m**, live `exportImage` returns real tiles (cars/lane-stripes resolved). Hillsborough `Aerials2025_3_inch` (Pictometry, Jan 2025) and Pasco documented; both blocked our test IP — Colab is the workaround.
- FL county/state orthoimagery is a **public record** — free, commercial use permitted (attribution as courtesy). Safe to use in a paid report.
- Label reuse is valid because we re-fetch over each chip's **identical ground bbox at identical pixel dimensions** — only the pixel *content* changes (hallucinated → real).
