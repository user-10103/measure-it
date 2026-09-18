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
    '        print(f"[matcher] {int((~np.isfinite(cost)).sum())} non-finite cost entries -> 1e9", flush=True)\n'
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


# INCOMPLETE COPY. A THIRD patch — the trainer NaN guard — exists only in S3:
#   s3://measure-it-prod-017341176694/models/patch_sam3.py
#   md5 507c023d029e55d062491f3f3a6cfc73   (this git copy: f0fca9a7…)
# This file has zero occurrences of patch_trainer or nan-guard. Applying it and
# training is what produced the epoch-6 crash: two of three patches land, the
# script reports success, and the guard that stops the run dying mid-training is
# simply not there.
#
# Until the S3 version is committed, this copy refuses to report success rather
# than silently applying 2 of 3. Override only if you are deliberately patching
# something that does not train.
TRAINER_GUARD_SENTINEL = "nan-guard"
S3_COMPLETE_COPY = ("s3://measure-it-prod-017341176694/models/patch_sam3.py "
                    "(md5 507c023d029e55d062491f3f3a6cfc73)")


def has_trainer_guard() -> bool:
    """Does THIS file carry the trainer NaN guard? Currently: no.

    Asks the MODULE NAMESPACE, never the source text. Two earlier versions
    grepped this file — first for "patch_trainer", then for "def patch_trainer" —
    and both returned True on a file that defines no such function, because the
    search string is itself in the file doing the searching. A detector matching
    its own source is the purest form of the failure this module exists to
    prevent, and it took two goes to stop writing it.

    A name is in globals() only if a def actually executed.
    """
    return "patch_trainer" in globals()


def apply_all(root: str):
    return [patch_fused(root), patch_matcher(root)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3-root", default="/content/sam3repo",
                    help="clone root containing the sam3/ package")
    ap.add_argument("--allow-missing-trainer-guard", action="store_true",
                    help="proceed even though this copy lacks the trainer NaN "
                         "guard (see S3_COMPLETE_COPY)")
    args = ap.parse_args()
    if not os.path.isdir(os.path.join(args.sam3_root, "sam3")):
        sys.exit(f"sam3 package not found under {args.sam3_root}")
    if not has_trainer_guard() and not args.allow_missing_trainer_guard:
        sys.exit(
            "\nFAILED: this copy of patch_sam3.py does NOT carry the trainer "
            "NaN guard.\n"
            "  fused.py and matcher.py may have been patched above, but the "
            "guard that\n"
            "  keeps training alive mid-run is absent — which is exactly how the "
            "epoch-6\n  run died.\n\n"
            f"  Complete copy: {S3_COMPLETE_COPY}\n\n"
            "  Fetch that, or pass --allow-missing-trainer-guard if you are "
            "deliberately\n  patching something that will not train.")
    results = apply_all(args.sam3_root)
    for line in results:
        print(line)
    # FAIL LOUD. Every result was printed and the process exited 0 regardless —
    # including "NOT FOUND (SKIPPED)" and "anchor not found (SKIPPED)". A caller
    # checking the exit code, or a runbook saying "STOP if it reports no files
    # changed", saw success on a run where NOTHING was patched.
    #
    # That is how the mix15 run reached the GPU without the matcher NaN guard:
    # the word SKIPPED was in the output and nothing enforced it. The guard was
    # credited twice, by two different readers, because the script said 0.
    skipped = [r for r in results if "SKIPPED" in r]
    if skipped:
        sys.exit("\nFAILED: " + str(len(skipped)) + " patch(es) did not apply:\n  "
                 + "\n  ".join(skipped)
                 + "\n\nThe matcher NaN guard and/or the fused-MLP grad fallback are "
                   "NOT in place. Training will run without them and can crash "
                   "mid-run with no checkpoint beyond the last sync. Fix the "
                   "--sam3-root path, or update the anchors for this sam3 version, "
                   "before renting GPU time.")
    print("\nOK: every patch applied or already present.")


if __name__ == "__main__":
    main()
