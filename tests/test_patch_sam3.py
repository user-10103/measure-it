"""Tests for training/sam3/patch_sam3.py — idempotent sam3-source patches."""
import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location(
    "patch_sam3",
    os.path.join(os.path.dirname(__file__), "..", "training", "sam3", "patch_sam3.py"),
)
patch_sam3 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(patch_sam3)


FUSED_SRC = '''import torch
def addmm_act(activation, linear, mat1):
    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")
    self = linear.bias.detach()
    return self
'''

MATCHER_SRC = '''import numpy as np
from scipy.optimize import linear_sum_assignment
def _do_matching(cost, repeats=1):
    i, j = linear_sum_assignment(cost)
    return i
'''


def _fake_sam3(tmp_path):
    os.makedirs(tmp_path / "sam3" / "perflib", exist_ok=True)
    os.makedirs(tmp_path / "sam3" / "train", exist_ok=True)
    (tmp_path / "sam3" / "perflib" / "fused.py").write_text(FUSED_SRC)
    (tmp_path / "sam3" / "train" / "matcher.py").write_text(MATCHER_SRC)
    return str(tmp_path)


def test_patches_apply(tmp_path):
    root = _fake_sam3(tmp_path)
    out = patch_sam3.apply_all(root)
    assert all("patched" in o and "already" not in o for o in out)

    fused = open(os.path.join(root, "sam3/perflib/fused.py")).read()
    assert "_F.linear(mat1, linear.weight" in fused          # differentiable fallback added
    assert 'raise ValueError(f"Unexpected activation' in fused

    matcher = open(os.path.join(root, "sam3/train/matcher.py")).read()
    assert "non-finite cost entries" in matcher
    assert "nan_to_num" in matcher
    # guard sits immediately before the solver call
    assert matcher.index("nan_to_num") < matcher.index("i, j = linear_sum_assignment")


def test_idempotent(tmp_path):
    root = _fake_sam3(tmp_path)
    patch_sam3.apply_all(root)
    before = open(os.path.join(root, "sam3/train/matcher.py")).read()
    out2 = patch_sam3.apply_all(root)                        # second run
    after = open(os.path.join(root, "sam3/train/matcher.py")).read()
    assert all("already patched" in o for o in out2)
    assert before == after                                   # no double-patching


def test_missing_files_skip_cleanly(tmp_path):
    os.makedirs(tmp_path / "sam3", exist_ok=True)            # package dir but no files
    out = patch_sam3.apply_all(str(tmp_path))
    assert all("SKIPPED" in o for o in out)
