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


def _load_checkpoint(path):
    """``torch.load`` for a multi-GB checkpoint, memory-mapped when possible.

    The fine-tuned checkpoint is ~10 GB and the Colab box has ~12 GB of RAM, so
    reading it wholly into memory is tight. ``mmap=True`` maps it instead.

    It is deliberately applied HERE, to our own checkpoint path, rather than by
    monkeypatching ``torch.load`` globally: a global patch that forces mmap
    breaks every file-like load in the process (mmap needs a real
    zipfile-serialised path), which is a much larger blast radius than the
    problem it solves. mmap also needs torch>=2.1, so fall back cleanly.
    """
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, ValueError, RuntimeError) as e:
        # TypeError: torch<2.1 has no mmap kwarg. ValueError/RuntimeError: the
        # file is not zipfile-serialised, or not a real path.
        logger.info("mmap checkpoint load unavailable (%s) — loading normally", e)
        return torch.load(path, map_location="cpu", weights_only=False)


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
    ckpt = _load_checkpoint(ckpt_path)
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
        # HF_HOME is read ONCE by huggingface_hub at import time — setting it in a
        # later notebook cell silently does nothing and the ~6.5 GB base weights
        # re-download to the default cache. Log where they actually land so a
        # mis-ordered env var is visible instead of costing a silent re-download.
        try:
            import os as _os
            from huggingface_hub.constants import HF_HUB_CACHE
            want = _os.environ.get("HF_HOME")
            if want and not str(HF_HUB_CACHE).startswith(str(want)):
                # WARNING, not info: the whole point is to be seen, and a fresh
                # Colab kernel's root logger sits at WARNING — an info-level
                # warning about silent misconfiguration is itself silent.
                logger.warning(
                    "HF_HOME=%s but huggingface_hub is caching in %s — HF_HOME was "
                    "set AFTER huggingface_hub was first imported, so it had no "
                    "effect and the ~6.5 GB base weights will re-download", 
                    want, HF_HUB_CACHE)
            else:
                logger.info("HF cache in use: %s", HF_HUB_CACHE)
        except Exception:  # noqa: BLE001 — observability only, never fatal
            pass
        base = build_sam3_image_model()          # pretrained weights, untouched
        base.eval().to(device)
        predict_outline = _make_predict(Sam3Processor(
            base, resolution=resolution, device=device,
            confidence_threshold=confidence_threshold))
        logger.info("zero-shot base SAM3 loaded for the outline")
    return predict_facets, predict_outline
