#!/usr/bin/env python3
"""
Run HEAT (Hierarchical Edge Attention Transformer) inference on raw image chips.

Bypasses OutdoorBuildingDataset. Loads all three HEAT components
(backbone + corner_model + edge_model) from the checkpoint's own saved `args`,
then runs forward inference on a folder of PNG chips.

Usage:
    python training/heat_infer.py \
        --checkpoint /tmp/heat_ckpts/SO2m/checkpoint_best.pth \
        --chips-dir  training/naip_dataset/valid \
        --output     /tmp/heat_results \
        --n-chips    10
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_load(path: str, map_location='cpu') -> dict:
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        import warnings
        warnings.warn(
            f"weights_only=True failed for {path} — loading with weights_only=False. "
            "Only use checkpoints from trusted sources.",
        )
        return torch.load(path, map_location=map_location, weights_only=False)  # noqa: S614


def _install_pip_deps(req_path: Path):
    """Install only pip-installable lines from requirements.txt (skip conda lines)."""
    if not req_path.exists():
        return
    lines = []
    for raw in req_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('conda') or line.startswith('--'):
            print(f"  skip (non-pip): {line}")
            continue
        lines.append(line)
    if lines:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q'] + lines,
            check=False,
        )


def _find_heat_master(root: Path) -> Path:
    """Walk repo root to find the heat-master source directory."""
    # Direct path (user confirmed)
    candidate = root / '2.- TRAINING-TESTING' / 'heat-master'
    if candidate.exists():
        return candidate
    # Fallback: find first dir containing infer_TESTING.py
    for p in root.rglob('infer_TESTING.py'):
        return p.parent
    return root


def _print_section(text: str, keyword: str, chars: int = 2000):
    """Print the section of a file around a keyword."""
    idx = text.lower().find(keyword.lower())
    if idx == -1:
        print(f"  ('{keyword}' not found)")
        return
    start = max(0, idx - 100)
    print(text[start: start + chars])


# ── model loading ─────────────────────────────────────────────────────────────

def _strip_ddp(state_dict: dict) -> dict:
    """Remove 'module.' prefix added by DistributedDataParallel."""
    out = {}
    for k, v in state_dict.items():
        out[k[len('module.'):] if k.startswith('module.') else k] = v
    return out


def load_heat_model(checkpoint_path: str, heat_src: Path, device='cuda'):
    """
    Load all three HEAT components using the checkpoint's own saved args.

    HEAT checkpoints:
      ck['args']         — original argparse.Namespace (has image_size, max_corner_num, …)
      ck['backbone']     — DDP state_dict (keys prefixed with 'module.')
      ck['corner_model'] — DDP state_dict
      ck['edge_model']   — DDP state_dict

    build_model() lives in arguments.py inside the heat-master directory.
    """
    import torch
    import importlib.util

    print(f"\nLoading checkpoint: {checkpoint_path}")
    ck = _safe_load(checkpoint_path)
    print("Checkpoint keys:", list(ck.keys()))

    ck_args = ck.get('args')
    if ck_args is not None:
        print(f"Checkpoint args: {ck_args}")
    else:
        print("WARNING: no 'args' key in checkpoint")

    # The HEAT source must be in sys.path before any import
    if str(heat_src) not in sys.path:
        sys.path.insert(0, str(heat_src))
    print(f"sys.path includes: {heat_src}")

    # build_model lives in arguments.py (confirmed)
    arguments_py = heat_src / 'arguments.py'
    if not arguments_py.exists():
        print(f"ERROR: {arguments_py} not found. Repo structure:")
        for p in heat_src.rglob('*.py'):
            print(' ', p.relative_to(heat_src))
        return None, None, None

    spec = importlib.util.spec_from_file_location('heat_arguments', arguments_py)
    args_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(args_mod)

    if not hasattr(args_mod, 'build_model'):
        print(f"ERROR: build_model not found in arguments.py")
        _print_section(arguments_py.read_text(), 'def ')
        return None, None, None

    print("Found build_model in arguments.py ✓")

    # Build models using stored args
    try:
        backbone, corner_model, edge_model = args_mod.build_model(ck_args)
    except Exception as e:
        print(f"build_model(ck_args) failed: {e}")
        print("Trying build_model() with no args...")
        try:
            backbone, corner_model, edge_model = args_mod.build_model()
        except Exception as e2:
            print(f"build_model() also failed: {e2}")
            return None, None, None

    # Load state dicts — strip DDP 'module.' prefix
    backbone.load_state_dict(_strip_ddp(ck['backbone']), strict=False)
    corner_model.load_state_dict(_strip_ddp(ck['corner_model']), strict=False)
    edge_model.load_state_dict(_strip_ddp(ck['edge_model']), strict=False)

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    backbone     = backbone.to(dev).eval()
    corner_model = corner_model.to(dev).eval()
    edge_model   = edge_model.to(dev).eval()

    print(f"Loaded backbone + corner_model + edge_model on {dev} ✓")
    return backbone, corner_model, edge_model


# ── inference ─────────────────────────────────────────────────────────────────

def chip_to_tensor(chip_path: Path, image_size: int = 256):
    import torch
    import numpy as np
    import cv2

    bgr = cv2.imread(str(chip_path))
    if bgr is None:
        return None, 0, 0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = rgb.shape[:2]
    resized = cv2.resize(rgb, (image_size, image_size))
    arr = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr  = (arr - mean) / std
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, h_orig, w_orig


def run_heat_inference(backbone, corner_model, edge_model, tensor, device):
    """
    Run the three-stage HEAT forward pass.
    Returns raw outputs for inspection if we can't decode polygons automatically.
    """
    import torch
    tensor = tensor.to(device)
    with torch.no_grad():
        try:
            # Standard HEAT forward: backbone → feature, corner prediction
            features = backbone(tensor)
            corner_out = corner_model(features, tensor)
            edge_out   = edge_model(features, corner_out, tensor)
            return corner_out, edge_out
        except Exception as e:
            print(f"  3-stage forward failed ({e}), trying direct call...")
            try:
                out = corner_model(tensor)
                return out, None
            except Exception as e2:
                return None, str(e2)


def decode_corners(corner_out, image_size: int, orig_h: int, orig_w: int):
    """Extract predicted corner coordinates from HEAT corner output."""
    import torch
    import numpy as np

    if corner_out is None:
        return []

    corners = []
    if isinstance(corner_out, (tuple, list)):
        for item in corner_out:
            if isinstance(item, torch.Tensor) and item.ndim >= 2:
                arr = item[0].detach().cpu().numpy()
                if arr.shape[-1] == 2:
                    # [N, 2] normalised coords
                    for pt in arr:
                        x = float(pt[0]) * orig_w
                        y = float(pt[1]) * orig_h
                        if 0 <= x <= orig_w and 0 <= y <= orig_h:
                            corners.append((x, y))
                elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                    # Heatmap [H, W] — extract local maxima
                    from scipy.ndimage import maximum_filter
                    hm = arr
                    local_max = (hm == maximum_filter(hm, size=7)) & (hm > 0.3)
                    ys, xs = np.where(local_max)
                    for xi, yi in zip(xs, ys):
                        cx = float(xi) / arr.shape[1] * orig_w
                        cy = float(yi) / arr.shape[0] * orig_h
                        corners.append((cx, cy))
    elif isinstance(corner_out, torch.Tensor):
        arr = corner_out[0].detach().cpu().numpy()
        if arr.shape[-1] == 2:
            for pt in arr:
                corners.append((float(pt[0]) * orig_w, float(pt[1]) * orig_h))

    return corners


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint',  required=True)
    ap.add_argument('--chips-dir',   required=True)
    ap.add_argument('--output',      default='/tmp/heat_results')
    ap.add_argument('--image-size',  type=int, default=256)
    ap.add_argument('--n-chips',     type=int, default=10)
    ap.add_argument('--repo-dir',    default='/content/Automatic_Roof_Plane_Extraction')
    ap.add_argument('--device',      default='cuda')
    args = ap.parse_args()

    import torch

    repo_root  = Path(args.repo_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clone if needed
    if not repo_root.exists():
        print("Cloning HEAT repo...")
        subprocess.run([
            'git', 'clone', '--depth', '1',
            'https://github.com/carecamp93/Automatic_Roof_Plane_Extraction',
            str(repo_root),
        ], check=True)

    # Find heat-master subdirectory
    heat_src = _find_heat_master(repo_root)
    print(f"HEAT source dir: {heat_src}")

    # Install pip-only deps
    _install_pip_deps(heat_src / 'requirements.txt')

    device = args.device if torch.cuda.is_available() else 'cpu'

    backbone, corner_model, edge_model = load_heat_model(
        args.checkpoint, heat_src, device
    )
    if backbone is None:
        print("\nCould not load model. Paste the output above and we'll fix it.")
        sys.exit(1)

    # Run on N chips
    chips = sorted(Path(args.chips_dir).glob('*.png'))
    if args.n_chips > 0:
        chips = chips[:args.n_chips]
    print(f"\nRunning inference on {len(chips)} chips...")

    results = []
    for chip_path in chips:
        tensor, h_orig, w_orig = chip_to_tensor(chip_path, args.image_size)
        if tensor is None:
            continue

        corner_out, edge_out = run_heat_inference(
            backbone, corner_model, edge_model, tensor, torch.device(device)
        )

        if isinstance(edge_out, str):
            print(f"  {chip_path.name}: ERROR — {edge_out}")
            results.append({'chip': chip_path.name, 'error': edge_out})
            continue

        corners = decode_corners(corner_out, args.image_size, h_orig, w_orig)
        entry = {
            'chip':     chip_path.name,
            'n_corners': len(corners),
            'corners':  corners,
            'raw_corner_shapes': (
                [list(x.shape) for x in corner_out
                 if hasattr(x, 'shape')]
                if isinstance(corner_out, (list, tuple)) else []
            ),
        }
        results.append(entry)
        print(f"  {chip_path.name}: {len(corners)} corners predicted")

    # Save
    out_json = output_dir / 'heat_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    ok = [r for r in results if 'error' not in r]
    print(f"\n{'='*50}")
    print(f"Chips processed: {len(results)}, successful: {len(ok)}")
    print(f"Avg corners/chip: {sum(r['n_corners'] for r in ok) / max(len(ok), 1):.1f}")
    print(f"Results: {out_json}")
    if ok:
        print("\nSample output (first chip):")
        print(json.dumps(ok[0], indent=2)[:800])


if __name__ == '__main__':
    main()
