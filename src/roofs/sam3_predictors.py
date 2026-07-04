"""Load the two SAM 3 predictors the pipeline uses (Colab-proven recipe).

Design (validated 2026-07-04 on real NAIP roofs):
  * ZERO-SHOT base SAM 3        -> "roof" OUTLINE (it nails this out of the box;
                                    fine-tuning on facets drifts the concept)
  * FINE-TUNED checkpoint       -> "roof facet" FACETS

Hard-won details baked in:
  * torch.autocast(bf16) around inference — the model runs bf16; without it a
    matmul mixes BFloat16/Float and crashes ("mat1 and mat2 must have the same
    dtype").
  * confidence_threshold=0.1 — the in-training checkpoint scores facets low;
    the processor filters BEFORE the caller sees masks, so keep this permissive
    and let masks_to_facets do the real filtering.
  * resolution stays 1008 — RoPE freqs are baked for the 72x72 grid at build.
  * The checkpoint holds the WHOLE model; load with strict=False and expect
    missing/unexpected ~0.

Heavy deps (torch / sam3) import lazily so the rest of the package stays
importable on CPU-only dev machines.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

PredictFn = Callable[["np.ndarray", str], Tuple["np.ndarray", "np.ndarray"]]  # noqa: F821


def _make_predict(processor):
    import numpy as np
    import torch
    from PIL import Image

    @torch.inference_mode()
    def _p(chip_arr, concept):
        # (image, text) -> (N x H x W bool masks, N scores)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            st = processor.set_image(Image.fromarray(chip_arr))
            out = processor.set_text_prompt(state=st, prompt=concept)
        m = out["masks"].detach().float().cpu().numpy()
        if m.ndim == 4:
            m = m[:, 0]
        m = m > (0.5 if (m.min() >= 0.0 and m.max() <= 1.0) else 0.0)
        s = out["scores"].detach().float().cpu().numpy().reshape(-1)
        return m, s

    return _p


def load_sam3_predictors(
    ckpt_path: str,
    use_zeroshot_outline: bool = True,
    device: str = "cuda",
    confidence_threshold: float = 0.1,
    resolution: int = 1008,
) -> Tuple[PredictFn, Optional[PredictFn]]:
    """Build (predict_facets, predict_outline) for ``segment_roof_sam``.

    Args:
        ckpt_path: fine-tuned SAM 3 checkpoint (local path; download from S3
            first in deployment). Trusted first-party file — loaded with
            ``weights_only=False`` because it carries optimizer state; never
            load an untrusted .pt this way.
        use_zeroshot_outline: also build the pristine base model for the
            outline (2 models ≈ 7 GB of weights; needs a ~16 GB+ GPU). When
            False, the fine-tuned model serves both prompts.
    """
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    ft = build_sam3_image_model()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = ft.load_state_dict(ckpt["model"], strict=False)
    logger.info("fine-tuned SAM3 loaded (epoch %s) missing=%d unexpected=%d",
                ckpt.get("epoch"), len(missing), len(unexpected))
    if len(missing) > 10 or len(unexpected) > 10:
        raise RuntimeError(
            f"checkpoint/model mismatch: missing={len(missing)} "
            f"unexpected={len(unexpected)} — wrong checkpoint or sam3 version")
    ft.eval().to(device)
    predict_facets = _make_predict(Sam3Processor(
        ft, resolution=resolution, device=device,
        confidence_threshold=confidence_threshold))

    predict_outline = None
    if use_zeroshot_outline:
        base = build_sam3_image_model()          # pretrained weights, untouched
        base.eval().to(device)
        predict_outline = _make_predict(Sam3Processor(
            base, resolution=resolution, device=device,
            confidence_threshold=confidence_threshold))
        logger.info("zero-shot base SAM3 loaded for the outline")
    return predict_facets, predict_outline
