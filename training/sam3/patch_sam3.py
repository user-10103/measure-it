#!/usr/bin/env python3
"""Idempotent patches to the *installed* sam3 source for fine-tuning on Colab.

The edits live in the cloned sam3 files, NOT this repo — so a fresh clone / VM
recycle silently loses them (we've been bitten by this). Re-run this after any
sam3 clone: ``python patch_sam3.py --sam3-root /content/sam3repo``. Safe to run
repeatedly (each patch no-ops if already applied).

Patches:
  1. perflib/fused.py::addmm_act — the fused MLP is inference-only (detaches
     weights, raises when grad is enabled). Add a differentiable fallback so the
     ViT MLP trains.
  2. train/matcher.py::_do_matching — a degenerate target box yields NaN in the
     Hungarian giou cost; scipy.linear_sum_assignment then rejects the matrix.
     Sanitize non-finite entries to 1e9 (the framework's own "invalid" sentinel)
     and log, so one bad sample can't crash the run.
"""
import argparse
import os
import sys

FUSED_OLD = '        raise ValueError("Expected grad to be disabled.")'
FUSED_NEW = (
    '        import torch.nn.functional as _F\n'
    '        y = _F.linear(mat1, linear.weight, linear.bias)\n'
    '        if activation in (_F.gelu, torch.nn.GELU): return _F.gelu(y)\n'
    '        if activation in (_F.relu, torch.nn.ReLU): return _F.relu(y)\n'
    '        raise ValueError(f"Unexpected activation {activation}")'
)

MATCH_OLD = "    i, j = linear_sum_assignment(cost)"
MATCH_NEW = (
    "    if not np.all(np.isfinite(cost)):\n"
    "        import logging as _lg\n"
    "        _lg.getLogger(__name__).warning(\n"
    '            "matcher: %d non-finite cost entries -> 1e9",\n'
    "            int((~np.isfinite(cost)).sum()))\n"
    "        cost = np.nan_to_num(cost, nan=1e9, posinf=1e9, neginf=1e9)\n"
    "    i, j = linear_sum_assignment(cost)"
)


def _apply(path: str, old: str, new: str, sentinel: str) -> str:
    """Apply a one-shot text patch; no-op if `sentinel` already present."""
    name = os.path.basename(path)
    if not os.path.exists(path):
        return f"{name}: NOT FOUND at {path} (SKIPPED)"
    s = open(path).read()
    if sentinel in s:
        return f"{name}: already patched"
    if old not in s:
        return f"{name}: anchor not found (SKIPPED — sam3 version changed?)"
    open(path, "w").write(s.replace(old, new, 1))
    return f"{name}: patched"


def patch_fused(root: str) -> str:
    return _apply(os.path.join(root, "sam3/perflib/fused.py"),
                  FUSED_OLD, FUSED_NEW, sentinel="_F.linear(mat1, linear.weight")


def patch_matcher(root: str) -> str:
    return _apply(os.path.join(root, "sam3/train/matcher.py"),
                  MATCH_OLD, MATCH_NEW, sentinel="non-finite cost entries")


def apply_all(root: str):
    return [patch_fused(root), patch_matcher(root)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3-root", default="/content/sam3repo",
                    help="clone root containing the sam3/ package")
    args = ap.parse_args()
    if not os.path.isdir(os.path.join(args.sam3_root, "sam3")):
        sys.exit(f"sam3 package not found under {args.sam3_root}")
    for line in apply_all(args.sam3_root):
        print(line)


if __name__ == "__main__":
    main()
