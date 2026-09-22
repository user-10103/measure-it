#!/usr/bin/env python3
"""
Run HEAT inference on raw NAIP chips and return roof plane polygons.

Uses get_results() and get_pixel_features() imported directly from
infer_TESTING.py — no reimplementation of the forward pass.

Output per chip: list of polygons (each a list of [x,y] pixel coords),
scaled to the original chip resolution.

Usage:
    python training/heat_infer.py \
        --checkpoint /tmp/heat_ckpts/SO2m/checkpoint_best.pth \
        --chips-dir  training/naip_dataset/valid \
        --output     /tmp/heat_results \
        --n-chips    10 \
        --repo-dir   "/content/Automatic_Roof_Plane_Extraction"
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ── safe torch.load ───────────────────────────────────────────────────────────

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


def _strip_ddp(state_dict: dict) -> dict:
    """Remove 'module.' prefix added by DistributedDataParallel."""
    return {(k[len('module.'):] if k.startswith('module.') else k): v
            for k, v in state_dict.items()}


# ── HEAT repo setup ───────────────────────────────────────────────────────────

def _find_heat_master(root: Path) -> Path:
    candidate = root / '2.- TRAINING-TESTING' / 'heat-master'
    if candidate.exists():
        return candidate
    for p in root.rglob('infer_TESTING.py'):
        return p.parent
    return root


def _install_pip_deps(req_path: Path):
    if not req_path.exists():
        return
    lines = [l.strip() for l in req_path.read_text().splitlines()
             if l.strip() and not l.startswith('#')
             and not l.startswith('conda') and not l.startswith('--')]
    if lines:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + lines,
                       check=False)


def setup_heat(repo_root: Path, device: str):
    """Clone repo if needed, install deps, add to sys.path."""
    if not repo_root.exists():
        print("Cloning HEAT repo...")
        subprocess.run(['git', 'clone', '--depth', '1',
                        'https://github.com/carecamp93/Automatic_Roof_Plane_Extraction',
                        str(repo_root)], check=True)

    heat_src = _find_heat_master(repo_root)
    print(f"HEAT source dir: {heat_src}")

    _install_pip_deps(heat_src / 'requirements.txt')

    if str(heat_src) not in sys.path:
        sys.path.insert(0, str(heat_src))

    return heat_src


# ── model loading ─────────────────────────────────────────────────────────────

def load_heat_model(checkpoint_path: str, heat_src: Path, device: str):
    import torch
    import importlib.util
    import traceback as _tb

    print(f"\nLoading checkpoint: {checkpoint_path}")
    ck = _safe_load(checkpoint_path)
    ck_args = ck.get('args')
    print(f"corner_to_edge_multiplier = {ck_args.corner_to_edge_multiplier}")
    print(f"max_corner_num = {ck_args.max_corner_num}")

    def _import_class(module_file: str, class_name: str):
        fpath = heat_src / module_file
        spec  = importlib.util.spec_from_file_location(class_name, fpath)
        mod   = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, class_name)

    ResNetBackbone = _import_class('models/resnet.py',         'ResNetBackbone')
    HeatCorner     = _import_class('models/corner_models.py',  'HeatCorner')
    HeatEdge       = _import_class('models/edge_models.py',    'HeatEdge')
    print("Imported ResNetBackbone, HeatCorner, HeatEdge ✓")

    # Exact pattern from train.py lines 188-200
    backbone     = ResNetBackbone()
    strides      = backbone.strides
    num_channels = backbone.num_channels
    print(f"Backbone: strides={strides}, num_channels={num_channels}")

    try:
        corner_model = HeatCorner(
            input_dim=128, hidden_dim=256, num_feature_levels=4,
            backbone_strides=strides, backbone_num_channels=num_channels,
        )
        print("HeatCorner ✓")
    except Exception:
        print("HeatCorner FAILED:"); _tb.print_exc(); return None, None, None, None

    try:
        edge_model = HeatEdge(
            input_dim=128, hidden_dim=256, num_feature_levels=4,
            backbone_strides=strides, backbone_num_channels=num_channels,
        )
        print("HeatEdge ✓")
    except Exception:
        print("HeatEdge FAILED:"); _tb.print_exc(); return None, None, None, None

    try:
        backbone.load_state_dict(_strip_ddp(ck['backbone']), strict=False)
        corner_model.load_state_dict(_strip_ddp(ck['corner_model']), strict=False)
        edge_model.load_state_dict(_strip_ddp(ck['edge_model']), strict=False)
        print("State dicts loaded ✓")
    except Exception:
        print("State dict loading FAILED:"); _tb.print_exc(); return None, None, None, None

    dev = torch.device(device if __import__('torch').cuda.is_available() else 'cpu')
    backbone     = backbone.to(dev).eval()
    corner_model = corner_model.to(dev).eval()
    edge_model   = edge_model.to(dev).eval()

    print(f"All models on {dev} ✓")
    return backbone, corner_model, edge_model, ck_args


# ── polygon assembly from planar graph ───────────────────────────────────────

def planar_graph_to_polygons(corners, edges, min_vertices: int = 3):
    """
    Convert HEAT planar graph (corners + edges) to face polygons.

    Uses NetworkX to find minimal cycles in the undirected graph.
    Each cycle with ≥ min_vertices corners becomes a roof face polygon.

    corners: [N, 2] array of (x, y) pixel coords
    edges:   [M, 2] array of (i, j) corner index pairs
    Returns: list of polygons, each a list of [x, y] pairs
    """
    try:
        import networkx as nx
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'networkx'],
                       check=False)
        import networkx as nx

    import numpy as np

    if len(corners) < min_vertices or len(edges) == 0:
        return []

    G = nx.Graph()
    G.add_nodes_from(range(len(corners)))
    G.add_edges_from(edges.tolist() if hasattr(edges, 'tolist') else edges)

    # Find minimal cycles (faces) via minimum cycle basis
    try:
        cycles = nx.minimum_cycle_basis(G)
    except Exception:
        # fallback: simple cycles
        cycles = list(nx.simple_cycles(G))

    polygons = []
    for cycle in cycles:
        if len(cycle) < min_vertices:
            continue
        pts = [[float(corners[i][0]), float(corners[i][1])] for i in cycle]
        # Close the polygon
        pts.append(pts[0])
        polygons.append(pts)

    return polygons


# ── main inference ────────────────────────────────────────────────────────────

def run_on_chip(chip_path: Path, backbone, corner_model, edge_model,
                ck_args, get_results_fn, get_pixel_features_fn,
                postprocess_fn, process_image_fn,
                image_size: int = 256, infer_times: int = 3, device='cuda'):
    import cv2
    import torch
    import numpy as np

    bgr = cv2.imread(str(chip_path))
    if bgr is None:
        return None

    h_orig, w_orig = bgr.shape[:2]
    bgr_resized = cv2.resize(bgr, (image_size, image_size))

    # process_image expects BGR (cv2 native, no RGB conversion — confirmed)
    img_tensor = process_image_fn(bgr_resized).to(
        torch.device(device if torch.cuda.is_available() else 'cpu'))

    pixels, pixel_features = get_pixel_features_fn(image_size=image_size)
    pixel_features = pixel_features.to(img_tensor.device)

    with torch.no_grad():
        pred_corners, pred_confs, pos_edges, edge_confs, c_heatmap = get_results_fn(
            img_tensor, None,
            backbone, corner_model, edge_model,
            pixels, pixel_features, ck_args,
            infer_times=infer_times,
            corner_thresh=0.01,
            image_size=image_size,
        )

    pred_corners, pred_confs, pos_edges = postprocess_fn(
        pred_corners, pred_confs, pos_edges)

    # Scale corners from 256×256 to original chip resolution
    scale_x = w_orig / image_size
    scale_y = h_orig / image_size
    corners_orig = np.stack([
        pred_corners[:, 0] * scale_x,
        pred_corners[:, 1] * scale_y,
    ], axis=1) if len(pred_corners) > 0 else pred_corners

    polygons = planar_graph_to_polygons(corners_orig, pos_edges)

    return {
        'n_corners':  len(pred_corners),
        'n_edges':    len(pos_edges),
        'n_polygons': len(polygons),
        'corners':    corners_orig.tolist() if hasattr(corners_orig, 'tolist') else [],
        'edges':      pos_edges.tolist()    if hasattr(pos_edges,    'tolist') else [],
        'polygons':   polygons,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint',  required=True)
    ap.add_argument('--chips-dir',   required=True)
    ap.add_argument('--output',      default='/tmp/heat_results')
    ap.add_argument('--image-size',  type=int, default=256)
    ap.add_argument('--infer-times', type=int, default=3)
    ap.add_argument('--n-chips',     type=int, default=10)
    ap.add_argument('--repo-dir',    default='/content/Automatic_Roof_Plane_Extraction')
    ap.add_argument('--device',      default='cuda')
    args = ap.parse_args()

    import torch
    import importlib.util
    import traceback as _tb

    repo_root  = Path(args.repo_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    heat_src = setup_heat(repo_root, args.device)

    device = args.device if torch.cuda.is_available() else 'cpu'
    backbone, corner_model, edge_model, ck_args = load_heat_model(
        args.checkpoint, heat_src, device)

    if backbone is None:
        print("Model loading failed — see traceback above.")
        sys.exit(1)

    # Import inference helpers directly from infer_TESTING.py
    infer_py = heat_src / 'infer_TESTING.py'
    spec = importlib.util.spec_from_file_location('heat_infer_testing', infer_py)
    infer_mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(infer_mod)
    except SystemExit:
        pass  # infer_TESTING.py may call main() at module level

    get_results_fn        = infer_mod.get_results
    get_pixel_features_fn = infer_mod.get_pixel_features
    postprocess_fn        = infer_mod.postprocess_preds
    process_image_fn      = infer_mod.process_image
    print("Imported inference helpers from infer_TESTING.py ✓")

    chips = sorted(Path(args.chips_dir).glob('*.png'))
    if args.n_chips > 0:
        chips = chips[:args.n_chips]
    print(f"\nRunning on {len(chips)} chips (infer_times={args.infer_times})...")

    results = []
    for chip_path in chips:
        try:
            r = run_on_chip(
                chip_path, backbone, corner_model, edge_model, ck_args,
                get_results_fn, get_pixel_features_fn,
                postprocess_fn, process_image_fn,
                image_size=args.image_size,
                infer_times=args.infer_times,
                device=device,
            )
            if r is None:
                print(f"  {chip_path.name}: could not read image")
                continue
            r['chip'] = chip_path.name
            results.append(r)
            print(f"  {chip_path.name}: {r['n_corners']} corners, "
                  f"{r['n_edges']} edges, {r['n_polygons']} polygons")
        except Exception:
            print(f"  {chip_path.name}: FAILED")
            _tb.print_exc()
            results.append({'chip': chip_path.name,
                             'error': _tb.format_exc()})

    out_json = output_dir / 'heat_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    ok = [r for r in results if 'error' not in r]
    print(f"\n{'='*60}")
    print(f"Processed: {len(results)}  Success: {len(ok)}")
    if ok:
        avg_poly = sum(r['n_polygons'] for r in ok) / len(ok)
        avg_corn = sum(r['n_corners']  for r in ok) / len(ok)
        print(f"Avg corners/chip: {avg_corn:.1f}")
        print(f"Avg polygons/chip: {avg_poly:.1f}")
        print(f"\nSample (first chip):")
        sample = {k: v for k, v in ok[0].items() if k != 'polygons'}
        sample['first_polygon'] = ok[0]['polygons'][0] if ok[0]['polygons'] else []
        print(json.dumps(sample, indent=2))
    print(f"\nFull results → {out_json}")


if __name__ == '__main__':
    import traceback as _tb
    try:
        main()
    except Exception:
        _tb.print_exc()
        sys.exit(1)
