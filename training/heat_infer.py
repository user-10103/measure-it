#!/usr/bin/env python3
"""
Run HEAT (Hierarchical Edge Attention Transformer) inference on raw image chips
without the OutdoorBuildingDataset wrapper.

Clones carecamp93/Automatic_Roof_Plane_Extraction, loads a pretrained checkpoint,
and runs forward inference on a directory of PNG chips. Outputs one JSON per chip
with the predicted roof plane polygons (list of [x,y] vertices).

Usage (Colab):
    python training/heat_infer.py \
        --checkpoint /tmp/heat_ckpts/SO2m/checkpoint_best.pth \
        --chips-dir  training/naip_dataset/valid \
        --output     /tmp/heat_results \
        --image-size 256 \
        --n-chips    10
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def clone_heat(repo_dir: Path):
    if not repo_dir.exists():
        print("Cloning HEAT repo...")
        subprocess.run([
            'git', 'clone', '--depth', '1',
            'https://github.com/carecamp93/Automatic_Roof_Plane_Extraction',
            str(repo_dir),
        ], check=True)
    else:
        print(f"HEAT repo already at {repo_dir}")


def install_deps(repo_dir: Path):
    req = repo_dir / 'requirements.txt'
    if req.exists():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(req)])


def _safe_load(path: str, map_location='cpu') -> dict:
    """torch.load with weights_only=True (safe), falling back with a warning."""
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        import warnings
        warnings.warn(
            f"weights_only=True failed for {path} — falling back to weights_only=False. "
            "Only load checkpoints from trusted sources.",
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False)  # noqa: S614


def load_model(checkpoint_path: str, repo_dir: Path, image_size: int = 256, device='cuda'):
    """Load HEAT model from checkpoint without the dataset wrapper."""
    sys.path.insert(0, str(repo_dir))

    import torch

    # Try to import the model class — name varies by HEAT version
    model = None
    for module_name in ['models.HEAT', 'model.HEAT', 'HEAT', 'models.heat', 'heat']:
        try:
            mod = __import__(module_name, fromlist=['HEAT', 'build_model', 'HEATNet'])
            for cls_name in ['HEAT', 'HEATNet', 'build_model']:
                if hasattr(mod, cls_name):
                    cls = getattr(mod, cls_name)
                    print(f"Found model class: {module_name}.{cls_name}")
                    model = cls
                    break
            if model:
                break
        except ImportError:
            continue

    if model is None:
        # Fallback: load checkpoint and inspect what's in it
        ck = _safe_load(checkpoint_path)
        print("Checkpoint keys:", list(ck.keys())[:10])
        print("Cannot locate HEAT model class — inspect repo structure:")
        for p in Path(repo_dir).rglob('*.py'):
            print(' ', p.relative_to(repo_dir))
        return None, None

    # Build or instantiate
    try:
        net = model(image_size=image_size)
    except TypeError:
        try:
            net = model()
        except Exception as e:
            print(f"Model instantiation failed: {e}")
            return None, None

    ck = _safe_load(checkpoint_path)
    state = ck.get('model', ck.get('state_dict', ck))
    net.load_state_dict(state, strict=False)
    net = net.to(device)
    net.eval()
    print(f"Loaded HEAT checkpoint: {checkpoint_path}")
    return net, torch.device(device)


def chip_to_tensor(chip_path: Path, image_size: int = 256):
    import torch
    import numpy as np
    import cv2

    bgr = cv2.imread(str(chip_path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = rgb.shape[:2]
    resized = cv2.resize(rgb, (image_size, image_size))
    arr = resized.astype(np.float32) / 255.0
    # ImageNet normalisation
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, h_orig, w_orig


def decode_output(output, image_size: int, orig_h: int, orig_w: int):
    """
    Attempt to extract polygon vertices from HEAT output.
    HEAT returns: (polygon_logits, edge_logits, vertex_coords)
    Shape varies by version — we try common patterns.
    """
    import torch
    polygons = []

    if isinstance(output, (tuple, list)):
        # Common HEAT output: (corner_pred, edge_pred, polygon_pred)
        # corner_pred shape: [B, N_corners, 2]  (normalised 0-1 coords)
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim == 3:
                coords = item[0].detach().cpu().numpy()  # [N, 2]
                # Convert normalised coords to original pixel space
                pts = [(float(c[0]) * orig_w, float(c[1]) * orig_h) for c in coords]
                if len(pts) >= 3:
                    polygons.append(pts)
    elif isinstance(output, dict):
        for key in ['pred_polygons', 'polygons', 'pred_vertices']:
            if key in output:
                item = output[key]
                if hasattr(item, 'detach'):
                    coords = item[0].detach().cpu().numpy()
                    pts = [(float(c[0]) * orig_w, float(c[1]) * orig_h) for c in coords]
                    polygons.append(pts)

    return polygons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint',  required=True)
    ap.add_argument('--chips-dir',   required=True)
    ap.add_argument('--output',      default='/tmp/heat_results')
    ap.add_argument('--image-size',  type=int, default=256)
    ap.add_argument('--n-chips',     type=int, default=10,
                    help='number of chips to test (0 = all)')
    ap.add_argument('--repo-dir',    default='/content/Automatic_Roof_Plane_Extraction')
    ap.add_argument('--device',      default='cuda')
    args = ap.parse_args()

    repo_dir   = Path(args.repo_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    clone_heat(repo_dir)
    install_deps(repo_dir)

    import torch
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    net, dev = load_model(args.checkpoint, repo_dir, args.image_size, device)
    if net is None:
        print("\nCould not load model automatically.")
        print("Please inspect the repo and report:")
        print("  1. Which .py file defines the main model class")
        print("  2. The class name")
        print("  3. What checkpoint keys look like")
        sys.exit(1)

    chips = sorted(Path(args.chips_dir).glob('*.png'))
    if args.n_chips > 0:
        chips = chips[:args.n_chips]
    print(f"\nRunning inference on {len(chips)} chips...")

    results = []
    for chip_path in chips:
        result = chip_to_tensor(chip_path, args.image_size)
        if result is None:
            continue
        tensor, h_orig, w_orig = result
        tensor = tensor.to(dev)

        with torch.no_grad():
            try:
                output = net(tensor)
            except Exception as e:
                print(f"  {chip_path.name}: inference failed — {e}")
                results.append({'chip': chip_path.name, 'error': str(e)})
                continue

        polygons = decode_output(output, args.image_size, h_orig, w_orig)
        entry = {'chip': chip_path.name, 'n_polygons': len(polygons), 'polygons': polygons}
        results.append(entry)
        print(f"  {chip_path.name}: {len(polygons)} polygons")

    # Save results
    out_json = output_dir / 'heat_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Summary
    ok = [r for r in results if 'error' not in r]
    print(f"\n{'='*50}")
    print(f"Processed: {len(results)} chips")
    print(f"Success:   {len(ok)} chips")
    print(f"Avg polygons per chip: {sum(r['n_polygons'] for r in ok) / max(len(ok),1):.1f}")

    if not ok:
        print("\nNo successful inferences — model interface likely needs adjustment.")
        print("Report the error messages above and the repo structure.")


if __name__ == '__main__':
    main()
