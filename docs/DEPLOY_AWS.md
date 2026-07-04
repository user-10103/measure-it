# Deploying the SAM 3 Roof-Report Service on AWS

Target output: the Roofr-style 6-page report (see `110 HOLLAND LN.pdf` benchmark) —
cover (aerial + headline stats) / diagram / length report / area report / pitch /
summary. The code path is `src/serve/report_service.generate_roof_report`, which is
the SAME logic the Colab demo converged on. Everything below is what we learned
getting there (2026-07-04) — do not rediscover it.

## The one entry point

```python
from src.roofs.sam3_predictors import load_sam3_predictors
from src.serve.report_service import generate_roof_report

predict_facets, predict_outline = load_sam3_predictors("/opt/models/sam3_roof_ft.pt")
res = generate_roof_report("909 Spring Island Way, Melbourne, FL", "FL",
                           predict_facets, predict_outline, out_dir="/tmp/job1")
# res.pdf_path, res.num_facets, res.plan_area_m2, res.edge_totals_m
```

`location` accepts a street address, `"lat, lon"`, a Google Maps link, or a
(lat, lon) tuple — `src/utils/resolve_location.py` handles all of it with the
variant retries (the ZIP bug) and the provider chain.

## Colab-validated facts (baked into defaults — listed so nobody "fixes" them)

| Finding | Where |
|---|---|
| Zero-shot base SAM 3 for the OUTLINE, fine-tuned for FACETS (2 models, ~7 GB) | `sam3_predictors.load_sam3_predictors` |
| `torch.autocast(bf16)` around inference or dtype crash | `sam3_predictors._make_predict` |
| Processor `confidence_threshold=0.1`; real filtering in `masks_to_facets` | `sam3_predictors` default |
| LOOSE chip crop (18 m buffer) — tight crops starve the facet model | `report_service.CHIP_BUFFER_M` |
| `score_thr=0.15` for the in-training checkpoint | `report_service.SCORE_THR` |
| Drop SAM's whole-roof mask or it swallows every facet ("1 facet" bug) | `mask_facets.max_area_frac=0.9` |
| Outline fallback = facet union, so eaves always exist | `report_service._fallback_outline` |
| Typed edges (eave/ridge/hip/valley) from outline+facet geometry | `roofs/geom_edges.py` |
| Resolution stays 1008 (RoPE baked at build) | `sam3_predictors` default |
| Checkpoint: use the box-only epoch-9 until the mask-supervised run cooks | S3 `models/` |
| RF-DETR must NOT be installed alongside sam3 (transformers pin conflict) | container image |

## Hugging Face auth — never interactive on AWS

Colab needed `notebook_login()` every session because the VM is ephemeral. On AWS:

1. **Preferred: no HF at runtime at all.** One-time, from any machine that has the
   weights cached: upload BOTH model weight sets to S3 —
   the fine-tuned checkpoint AND the base SAM 3 weights (the HF cache dir
   `~/.cache/huggingface/hub/models--facebook--sam3` after one successful build).
   The container syncs `s3://<bucket>/models/` to disk at startup and sets
   `HF_HOME=/opt/models/hf` so `build_sam3_image_model()` finds the cache and
   never phones home. Zero tokens, zero gating, works in air-gapped subnets.
2. **Fallback: token via env.** Store a read token in SSM Parameter Store
   (SecureString `/measure-it/hf-token`); the service startup exports it as
   `HF_TOKEN`, which `huggingface_hub` reads automatically — non-interactive.
   The EBS volume persists the download cache, so it downloads once per volume.

Either way: nobody ever types a token into a prompt in production.

## Infrastructure (Phase 1 — see aws_setup_brief.md for console steps)

- **us-west-2 everything** (NAIP lives there; requester-pays via the instance
  IAM role — the InvalidAccessKeyId dance from Colab disappears).
- g5.xlarge (A10G 24 GB) fits both models in bf16.
- S3: `models/` (weights), `reports/` (output PDFs, served via presigned URLs).
- Amazon Location place index `measure-it-places` + env
  `AWS_LOCATION_PLACE_INDEX=measure-it-places` → commercial geocoding becomes
  the primary provider (fixes landmark names the free geocoders miss).

## Honest gaps (say them to the client, don't hide them)

- **Pitch** shows "unspecified" and areas are PLAN areas until the LiDAR fusion
  lands (the Holland Ln benchmark's pitch page needs it).
- **Eave vs rake** can't be split without slope direction — LiDAR again.
- Facet quality reflects the in-training checkpoint; it improves with the
  dataset growth toward ~5,000 labeled roofs.
