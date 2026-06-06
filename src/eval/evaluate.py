"""
Evaluation Harness

Scores predicted roof annotations against ground-truth annotations using the
client contract tolerances. This is the gate that decides whether the model
"meets contract."

Both prediction and ground-truth are COCO instance-segmentation dicts:
    {
      "images": [{"id", "address_id", "width", "height"}],
      "annotations": [{"id", "image_id", "category_id", "segmentation", "attributes"}],
      "categories": [...]
    }
Category ids: 1=roof_polygon(outline), 2=facet, 8=eave, 9=rake, 10=ridge,
11=valley, 12=hip. Predicted images are matched to GT images by address_id.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


# --- Category ids -----------------------------------------------------------
CAT_OUTLINE = 1
CAT_FACET = 2
EDGE_CATS: Dict[int, str] = {8: "eave", 9: "rake", 10: "ridge", 11: "valley", 12: "hip"}


# --- Contract tolerances (the pass/fail gate) -------------------------------
AREA_TOL_PCT = 5.0
EDGE_LEN_TOL_PCT = 5.0
PITCH_TOL_DEG = 5.0
FACET_F1_MIN = 0.85
OUTLINE_IOU_MIN = 0.90

# IoU threshold for facet matching
FACET_IOU_THRESH = 0.5


@dataclass
class EvalResult:
    """
    Aggregate evaluation result with per-metric pass/fail gates.

    Attributes:
        n_images: Number of GT images that were scored (matched by address_id).
        facet_precision: Facet detection precision (greedy IoU>=0.5 matching).
        facet_recall: Facet detection recall.
        facet_f1: Facet detection F1.
        facet_mean_iou: Mean IoU of matched (TP) facet pairs.
        outline_iou: Mean predicted-vs-GT roof outline IoU over images.
        area_err_pct: Mean absolute relative plan-area error (%).
        edge_len_err_pct: Mean absolute relative total edge-length error (%).
        edge_len_err_pct_by_type: Per-edge-type length error (%).
        pitch_err_deg: Mean absolute slope difference (deg) over matched facets.
        gates: Per-metric pass/fail booleans.
        passed: True iff every gate passes.
    """

    n_images: int
    facet_precision: float
    facet_recall: float
    facet_f1: float
    facet_mean_iou: float
    outline_iou: float
    area_err_pct: float
    edge_len_err_pct: float
    edge_len_err_pct_by_type: Dict[str, float] = field(default_factory=dict)
    pitch_err_deg: Optional[float] = None
    gates: Dict[str, bool] = field(default_factory=dict)
    passed: bool = False

    def to_dict(self) -> dict:
        """Export as a JSON-serializable dict."""
        return {
            "n_images": self.n_images,
            "facet_precision": round(self.facet_precision, 4),
            "facet_recall": round(self.facet_recall, 4),
            "facet_f1": round(self.facet_f1, 4),
            "facet_mean_iou": round(self.facet_mean_iou, 4),
            "outline_iou": round(self.outline_iou, 4),
            "area_err_pct": round(self.area_err_pct, 4),
            "edge_len_err_pct": round(self.edge_len_err_pct, 4),
            "edge_len_err_pct_by_type": {
                k: round(v, 4) for k, v in self.edge_len_err_pct_by_type.items()
            },
            "pitch_err_deg": (
                round(self.pitch_err_deg, 4) if self.pitch_err_deg is not None else None
            ),
            "gates": dict(self.gates),
            "passed": self.passed,
        }


# --- Geometry helpers -------------------------------------------------------
def _ring_to_polygon(segmentation: List[float]) -> Optional[Polygon]:
    """
    Build a (validity-repaired) shapely Polygon from a flat COCO ring.

    Args:
        segmentation: Flat list [x0, y0, x1, y1, ...].

    Returns:
        A valid Polygon, or None if it cannot form an area.
    """
    if not segmentation or len(segmentation) < 6:
        return None
    coords = list(zip(segmentation[0::2], segmentation[1::2]))
    if len(coords) < 3:
        return None
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None
    return poly


def _segmentation_rings(ann: dict) -> List[List[float]]:
    """Return the list of flat rings for an annotation's segmentation field."""
    seg = ann.get("segmentation") or []
    if not seg:
        return []
    # COCO polygon format is a list of rings; tolerate a single flat ring too.
    if isinstance(seg[0], (int, float)):
        return [seg]
    return [ring for ring in seg if ring]


def _polygons_for_anns(anns: List[dict]) -> List[Polygon]:
    """Build all valid polygons (across all rings) for a list of annotations."""
    polys: List[Polygon] = []
    for ann in anns:
        for ring in _segmentation_rings(ann):
            poly = _ring_to_polygon(ring)
            if poly is not None:
                polys.append(poly)
    return polys


def _iou(a: Polygon, b: Polygon) -> float:
    """Intersection-over-union of two polygons (0.0 on degenerate input)."""
    if a is None or b is None:
        return 0.0
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def _polyline_length(segmentation: List[float]) -> float:
    """Total length of an open polyline given a flat [x0,y0,x1,y1,...] list."""
    pts = list(zip(segmentation[0::2], segmentation[1::2]))
    total = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        total += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return total


# --- Per-image accumulation -------------------------------------------------
def _anns_by_category(anns: List[dict]) -> Dict[int, List[dict]]:
    """Group annotations by category_id."""
    out: Dict[int, List[dict]] = {}
    for ann in anns:
        out.setdefault(ann.get("category_id"), []).append(ann)
    return out


def _greedy_facet_match(
    pred_facets: List[dict], gt_facets: List[dict]
) -> Tuple[int, int, int, List[float], List[Tuple[dict, dict]]]:
    """
    Greedy IoU matching of predicted to GT facets at FACET_IOU_THRESH.

    Returns:
        (tp, fp, fn, matched_ious, matched_pairs) where matched_pairs are
        (pred_ann, gt_ann) tuples for each true positive.
    """
    pred_polys = [(_polygons_for_anns([f]), f) for f in pred_facets]
    gt_polys = [(_polygons_for_anns([f]), f) for f in gt_facets]

    # Flatten: one representative polygon per facet (union of its rings).
    pred_items = [(_union(ps), ann) for ps, ann in pred_polys]
    gt_items = [(_union(ps), ann) for ps, ann in gt_polys]

    candidates: List[Tuple[float, int, int]] = []
    for pi, (pp, _) in enumerate(pred_items):
        if pp is None:
            continue
        for gi, (gp, _) in enumerate(gt_items):
            if gp is None:
                continue
            iou = _iou(pp, gp)
            if iou >= FACET_IOU_THRESH:
                candidates.append((iou, pi, gi))

    candidates.sort(key=lambda c: c[0], reverse=True)
    used_pred, used_gt = set(), set()
    matched_ious: List[float] = []
    matched_pairs: List[Tuple[dict, dict]] = []
    for iou, pi, gi in candidates:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matched_ious.append(iou)
        matched_pairs.append((pred_items[pi][1], gt_items[gi][1]))

    tp = len(matched_pairs)
    fp = len(pred_facets) - tp
    fn = len(gt_facets) - tp
    return tp, fp, fn, matched_ious, matched_pairs


def _union(polys: List[Polygon]) -> Optional[Polygon]:
    """Union a list of polygons into a single geometry (None if empty)."""
    if not polys:
        return None
    geom = polys[0]
    for p in polys[1:]:
        geom = geom.union(p)
    if geom.is_empty or geom.area <= 0:
        return None
    return geom


def _total_facet_area(anns: List[dict]) -> float:
    """Total polygon area (pixels) across facet annotations."""
    return sum(p.area for p in _polygons_for_anns(anns))


def _edge_lengths_by_type(anns_by_cat: Dict[int, List[dict]]) -> Dict[str, float]:
    """Total polyline length per edge type (eave/rake/ridge/valley/hip)."""
    lengths: Dict[str, float] = {name: 0.0 for name in EDGE_CATS.values()}
    for cat, name in EDGE_CATS.items():
        for ann in anns_by_cat.get(cat, []):
            for ring in _segmentation_rings(ann):
                lengths[name] += _polyline_length(ring)
    return lengths


def _rel_err_pct(pred: float, gt: float) -> Optional[float]:
    """Absolute relative error in percent; None if GT is zero (undefined)."""
    if gt == 0:
        return None
    return abs(pred - gt) / gt * 100.0


# --- Main evaluation --------------------------------------------------------
def evaluate(
    pred_coco: dict, gt_coco: dict, ids: Optional[List[str]] = None
) -> EvalResult:
    """
    Score predicted annotations against ground truth using contract tolerances.

    Args:
        pred_coco: Predicted COCO instance-seg dict.
        gt_coco: Ground-truth COCO instance-seg dict.
        ids: Optional list of address_id to restrict evaluation to (a split).

    Returns:
        EvalResult with aggregate metrics and pass/fail gates.
    """
    id_filter = set(ids) if ids is not None else None

    def index_images(coco: dict) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for img in coco.get("images", []):
            addr = img.get("address_id")
            if addr is None:
                continue
            if id_filter is not None and addr not in id_filter:
                continue
            out[addr] = img["id"]
        return out

    gt_imgs = index_images(gt_coco)
    pred_imgs = index_images(pred_coco)

    def anns_for(coco: dict, image_id: int) -> List[dict]:
        return [a for a in coco.get("annotations", []) if a.get("image_id") == image_id]

    # Accumulators
    tp_sum = fp_sum = fn_sum = 0
    matched_iou_vals: List[float] = []
    outline_ious: List[float] = []
    area_errs: List[float] = []
    edge_total_errs: List[float] = []
    edge_type_errs: Dict[str, List[float]] = {name: [] for name in EDGE_CATS.values()}
    pitch_diffs: List[float] = []

    n_scored = 0
    for addr, gt_image_id in gt_imgs.items():
        n_scored += 1
        gt_anns = anns_for(gt_coco, gt_image_id)
        gt_by_cat = _anns_by_category(gt_anns)

        if addr in pred_imgs:
            pred_anns = anns_for(pred_coco, pred_imgs[addr])
        else:
            pred_anns = []
        pred_by_cat = _anns_by_category(pred_anns)

        # 1. Facet detection
        tp, fp, fn, ious, pairs = _greedy_facet_match(
            pred_by_cat.get(CAT_FACET, []), gt_by_cat.get(CAT_FACET, [])
        )
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        matched_iou_vals.extend(ious)

        # 5. Pitch error (matched facets that both carry slope_deg)
        for pred_ann, gt_ann in pairs:
            ps = (pred_ann.get("attributes") or {}).get("slope_deg")
            gs = (gt_ann.get("attributes") or {}).get("slope_deg")
            if ps is not None and gs is not None:
                pitch_diffs.append(abs(float(ps) - float(gs)))

        # 2. Outline IoU
        pred_outline = _union(_polygons_for_anns(pred_by_cat.get(CAT_OUTLINE, [])))
        gt_outline = _union(_polygons_for_anns(gt_by_cat.get(CAT_OUTLINE, [])))
        if gt_outline is not None:
            outline_ious.append(_iou(pred_outline, gt_outline) if pred_outline else 0.0)

        # 3. Plan-area error %
        gt_area = _total_facet_area(gt_by_cat.get(CAT_FACET, []))
        pred_area = _total_facet_area(pred_by_cat.get(CAT_FACET, []))
        err = _rel_err_pct(pred_area, gt_area)
        if err is not None:
            area_errs.append(err)

        # 4. Edge-length error % (total + per-type)
        gt_edges = _edge_lengths_by_type(gt_by_cat)
        pred_edges = _edge_lengths_by_type(pred_by_cat)
        total_err = _rel_err_pct(sum(pred_edges.values()), sum(gt_edges.values()))
        if total_err is not None:
            edge_total_errs.append(total_err)
        for name in EDGE_CATS.values():
            te = _rel_err_pct(pred_edges[name], gt_edges[name])
            if te is not None:
                edge_type_errs[name].append(te)

    # --- Aggregate ----------------------------------------------------------
    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mean_matched_iou = (
        sum(matched_iou_vals) / len(matched_iou_vals) if matched_iou_vals else 0.0
    )
    outline_iou = sum(outline_ious) / len(outline_ious) if outline_ious else 0.0
    area_err_pct = sum(area_errs) / len(area_errs) if area_errs else 0.0
    edge_len_err_pct = (
        sum(edge_total_errs) / len(edge_total_errs) if edge_total_errs else 0.0
    )
    edge_by_type = {
        name: (sum(vals) / len(vals) if vals else 0.0)
        for name, vals in edge_type_errs.items()
    }
    pitch_err_deg = (
        sum(pitch_diffs) / len(pitch_diffs) if pitch_diffs else None
    )

    # --- Gates --------------------------------------------------------------
    gates = {
        "facet_f1": f1 >= FACET_F1_MIN,
        "outline_iou": outline_iou >= OUTLINE_IOU_MIN,
        "area_err_pct": area_err_pct <= AREA_TOL_PCT,
        "edge_len_err_pct": edge_len_err_pct <= EDGE_LEN_TOL_PCT,
        # Pitch gate passes vacuously when no comparable facets exist.
        "pitch_err_deg": (pitch_err_deg is None) or (pitch_err_deg <= PITCH_TOL_DEG),
    }
    passed = all(gates.values())

    result = EvalResult(
        n_images=n_scored,
        facet_precision=precision,
        facet_recall=recall,
        facet_f1=f1,
        facet_mean_iou=mean_matched_iou,
        outline_iou=outline_iou,
        area_err_pct=area_err_pct,
        edge_len_err_pct=edge_len_err_pct,
        edge_len_err_pct_by_type=edge_by_type,
        pitch_err_deg=pitch_err_deg,
        gates=gates,
        passed=passed,
    )
    logger.info(
        "Evaluated %d images: F1=%.3f outline_iou=%.3f area_err=%.2f%% "
        "edge_err=%.2f%% pitch_err=%s passed=%s",
        n_scored,
        f1,
        outline_iou,
        area_err_pct,
        edge_len_err_pct,
        f"{pitch_err_deg:.2f}" if pitch_err_deg is not None else "n/a",
        passed,
    )
    return result


def evaluate_files(
    pred_path, gt_path, split_path=None, split_name: str = "test"
) -> EvalResult:
    """
    Load COCO files from disk and evaluate, optionally restricted to a split.

    Args:
        pred_path: Path to predicted COCO JSON.
        gt_path: Path to ground-truth COCO JSON.
        split_path: Optional path to a split file {"train":[...],"val":[...],"test":[...]}.
        split_name: Which split key to use from the split file.

    Returns:
        EvalResult.
    """
    with open(pred_path, "r") as fh:
        pred_coco = json.load(fh)
    with open(gt_path, "r") as fh:
        gt_coco = json.load(fh)

    ids: Optional[List[str]] = None
    if split_path is not None:
        with open(split_path, "r") as fh:
            splits = json.load(fh)
        if split_name not in splits:
            raise KeyError(f"Split '{split_name}' not in {list(splits.keys())}")
        ids = list(splits[split_name])

    return evaluate(pred_coco, gt_coco, ids=ids)


def format_report(result: EvalResult) -> str:
    """
    Render a human-readable report with PASS/FAIL per gate.

    Args:
        result: EvalResult to format.

    Returns:
        Formatted multi-line string.
    """

    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    pitch_str = (
        f"{result.pitch_err_deg:.2f}" if result.pitch_err_deg is not None else "n/a"
    )
    lines = []
    lines.append("=" * 60)
    lines.append("CONTRACT EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Images scored: {result.n_images}")
    lines.append("")
    lines.append(f"{'Metric':<26}{'Value':>14}{'Tol/Min':>10}{'Gate':>8}")
    lines.append("-" * 60)
    lines.append(
        f"{'Facet F1':<26}{result.facet_f1:>14.3f}{FACET_F1_MIN:>10.2f}"
        f"{mark(result.gates['facet_f1']):>8}"
    )
    lines.append(
        f"{'  precision':<26}{result.facet_precision:>14.3f}{'':>10}{'':>8}"
    )
    lines.append(f"{'  recall':<26}{result.facet_recall:>14.3f}{'':>10}{'':>8}")
    lines.append(f"{'  mean matched IoU':<26}{result.facet_mean_iou:>14.3f}{'':>10}{'':>8}")
    lines.append(
        f"{'Outline IoU':<26}{result.outline_iou:>14.3f}{OUTLINE_IOU_MIN:>10.2f}"
        f"{mark(result.gates['outline_iou']):>8}"
    )
    lines.append(
        f"{'Plan-area error %':<26}{result.area_err_pct:>14.2f}{AREA_TOL_PCT:>10.2f}"
        f"{mark(result.gates['area_err_pct']):>8}"
    )
    lines.append(
        f"{'Edge-length error %':<26}{result.edge_len_err_pct:>14.2f}"
        f"{EDGE_LEN_TOL_PCT:>10.2f}{mark(result.gates['edge_len_err_pct']):>8}"
    )
    for name, val in result.edge_len_err_pct_by_type.items():
        lines.append(f"{'  ' + name:<26}{val:>14.2f}{'':>10}{'':>8}")
    lines.append(
        f"{'Pitch error (deg)':<26}{pitch_str:>14}{PITCH_TOL_DEG:>10.2f}"
        f"{mark(result.gates['pitch_err_deg']):>8}"
    )
    lines.append("-" * 60)
    lines.append(f"OVERALL: {'PASS' if result.passed else 'FAIL'}")
    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: print the report and exit 0 if passed else 1."""
    parser = argparse.ArgumentParser(
        description="Score predicted roof annotations against ground truth."
    )
    parser.add_argument("--pred", required=True, help="Path to predicted COCO JSON.")
    parser.add_argument("--gt", required=True, help="Path to ground-truth COCO JSON.")
    parser.add_argument("--split", default=None, help="Path to split JSON file.")
    parser.add_argument("--split-name", default="test", help="Split key to use.")
    args = parser.parse_args(argv)

    result = evaluate_files(
        args.pred, args.gt, split_path=args.split, split_name=args.split_name
    )
    print(format_report(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
