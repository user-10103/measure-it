#!/usr/bin/env python3
"""Guard the three data-distribution rules for the masks_v1 retrain.

The oversampler can only multiply, reads its multiplier from img["source"]
(absent -> "unknown" -> every repeat silently becomes 1x), and does not dedupe.
These three subcommands cover what it does not:

  dedupe  collapse duplicate records per roof id      (rule 2, run BEFORE oversampling)
  exclude drop held-out eval roofs from a corpus      (rule 1, run BEFORE oversampling)
  check   no held-out roof present in training        (rule 1, run on the FINAL set)
  plan    counts -> repeats that satisfy the targets  (rule 3)

Roof identity is img["address_id"] when present, else the file_name stem.

PROVENANCE. Drafted by the RunPod/AWS browser agent and relayed as text; it could
not push from its sandbox. It was reviewed and re-typed here, and the load-bearing
behaviour was verified against the REAL corpus rather than taken on trust:

    $ mix_guard.py check --dataset training/roof_dataset \
                         --eval training/roof_dataset_clean_eval
    held-out roofs: 197   leaked into training: 240
      [train] 913 roofs, 191 also held out
      [valid] 266 roofs,  49 also held out

That measurement, not the authorship, is why this is trusted. The four CLI-flag
corrections reported alongside it were likewise checked against the scripts
before the runbook was changed.

NO UNIT TESTS YET. A test file exists in the browser agent's session and has not
reached this repo. Until it lands, `check` is gating a paid GPU run on one live
verification and nothing else.
"""
from __future__ import annotations
import argparse, itertools, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

TARGETS = "GIS+NAIP >= 50%, no source > 35%, rid2 >= 10%"


def roof_id(im: dict) -> str:
    a = im.get("address_id")
    if a not in (None, ""):
        return str(a)
    return os.path.splitext(os.path.basename(im["file_name"]))[0]


def load(split_dir: Path) -> dict | None:
    p = split_dir / "_annotations.coco.json"
    return json.load(open(p)) if p.exists() else None


def cmd_dedupe(a):
    src, dst = Path(a.input), Path(a.output)
    for split in ("train", "valid", "test"):
        d = load(src / split)
        if d is None:
            continue
        by_img = defaultdict(list)
        for an in d["annotations"]:
            by_img[an["image_id"]].append(an)
        # keep the record with the most annotations per roof; ties -> lowest id
        best: dict[str, dict] = {}
        for im in d["images"]:
            r = roof_id(im)
            cur = best.get(r)
            if cur is None or (len(by_img[im["id"]]), -im["id"]) > (len(by_img[cur["id"]]), -cur["id"]):
                best[r] = im
        keep = {im["id"] for im in best.values()}
        out = dict(d)
        out["images"] = [im for im in d["images"] if im["id"] in keep]
        out["annotations"] = [an for an in d["annotations"] if an["image_id"] in keep]
        (dst / split).mkdir(parents=True, exist_ok=True)
        json.dump(out, open(dst / split / "_annotations.coco.json", "w"))
        print(f"  [{split}] {len(d['images'])} records -> {len(out['images'])} unique roofs "
              f"({len(d['annotations'])} -> {len(out['annotations'])} anns)")
    print(f"\ndeduped dataset written to {dst}")


FACET_NAMES = {"facet", "roof_facet", "roof facet", "roof face"}
STD_CATEGORIES = [{"id": 1, "name": "roof_polygon", "supercategory": "roof"},
                  {"id": 2, "name": "facet", "supercategory": "roof"}]


def cmd_remap(a):
    """Force a corpus onto the standard schema BEFORE merging.

    merge_datasets.py writes a HARDCODED categories block
    [roof_polygon:1, facet:2] while copying each source's category_id UNCHANGED.
    It never remaps. The GIS export declares a single class
    [{id:1, name:'roof facet'}], so after merging its 14,068 facet annotations
    still carry category_id 1 — which the merged file now calls roof_polygon.

    prep_sam3_facets then resolves the facet class by NAME, gets {2}, and DELETES
    every GIS annotation. The run trains with zero GIS data while the mix table
    reports GIS at ~33%. Nothing errors.

    Watch for the confirming symptom: merge_datasets counts facets as
    category_id == 2, so an unremapped GIS source reports "0 facets" in its own
    merge summary.
    """
    src, dst = Path(a.input), Path(a.output)
    moved_total = 0
    for split in ("train", "valid", "test"):
        d = load(src / split)
        if d is None:
            continue
        cats = d.get("categories", [])
        facet_ids = {c["id"] for c in cats if c.get("name", "").lower() in FACET_NAMES}
        if not facet_ids:
            sys.exit(f"STOP: {src/split} declares no facet-named category "
                     f"({[c.get('name') for c in cats]}). Refusing to guess which "
                     f"class is the facet class.")
        out = dict(d)
        out["categories"] = STD_CATEGORIES
        anns, moved = [], 0
        for an in d["annotations"]:
            an = dict(an)
            if an["category_id"] in facet_ids:
                if an["category_id"] != 2:
                    moved += 1
                an["category_id"] = 2
            else:
                an["category_id"] = 1
            anns.append(an)
        out["annotations"] = anns
        moved_total += moved
        (dst / split).mkdir(parents=True, exist_ok=True)
        json.dump(out, open(dst / split / "_annotations.coco.json", "w"))
        n2 = sum(1 for x in anns if x["category_id"] == 2)
        print(f"  [{split}] {len(anns)} anns -> {n2} facet(id 2), "
              f"{len(anns)-n2} outline(id 1); {moved} remapped")
    print(f"\nremapped {moved_total} annotation(s) onto the standard schema -> {dst}")
    if moved_total == 0:
        print("(already on the standard schema — nothing to do)")


def held_out_ids(eval_dir: str, splits: tuple[str, ...]) -> set[str]:
    """Roofs that must never be trained on.

    Only the eval's TEST split is held out - clean_eval/train is a legitimate
    training pool, and treating it as forbidden would discard 759 usable roofs.
    """
    ids: set[str] = set()
    for split in splits:
        d = load(Path(eval_dir) / split)
        if d:
            ids |= {roof_id(i) for i in d["images"]}
    return ids


def cmd_exclude(a):
    held = held_out_ids(a.eval, tuple(a.splits))
    src, dst = Path(a.input), Path(a.output)
    for split in ("train", "valid", "test"):
        d = load(src / split)
        if d is None:
            continue
        keep = {im["id"] for im in d["images"] if roof_id(im) not in held}
        out = dict(d)
        out["images"] = [im for im in d["images"] if im["id"] in keep]
        out["annotations"] = [an for an in d["annotations"] if an["image_id"] in keep]
        (dst / split).mkdir(parents=True, exist_ok=True)
        json.dump(out, open(dst / split / "_annotations.coco.json", "w"))
        print(f"  [{split}] {len(d['images'])} -> {len(out['images'])} images "
              f"({len(d['images']) - len(out['images'])} held-out roofs removed)")
    print(f"\nheld-out roofs excluded: {len(held)}   written to {dst}")


def cmd_check(a):
    eval_ids = held_out_ids(a.eval, tuple(a.splits))
    bad_total = 0
    for split in ("train", "valid"):
        d = load(Path(a.dataset) / split)
        if d is None:
            continue
        ids = {roof_id(i) for i in d["images"]}
        bad = ids & eval_ids
        bad_total += len(bad)
        print(f"  [{split}] {len(ids)} roofs, {len(bad)} also held out "
              f"{'' if not bad else sorted(bad)[:5]}")
    print(f"\nheld-out roofs: {len(eval_ids)}   leaked into training: {bad_total}")
    if bad_total:
        print("STOP (rule 1): held-out roofs present in training - the eval is destroyed.")
        sys.exit(1)
    print("PASS (rule 1): no held-out roof appears in training.")


def shares(counts: dict[str, int], reps: dict[str, int]) -> dict[str, float]:
    tot = sum(counts[s] * reps.get(s, 1) for s in counts)
    return {s: 100 * counts[s] * reps.get(s, 1) / tot for s in counts} if tot else {}


def ok(sh: dict[str, float]) -> bool:
    gis_naip = sh.get("gis", 0) + sh.get("phase1", 0)
    return (gis_naip >= 50.0 and max(sh.values(), default=0) <= 35.0
            and sh.get("rid2", 0) >= 10.0)


def cmd_plan(a):
    d = load(Path(a.dataset) / "train")
    if d is None:
        sys.exit(f"no train split under {a.dataset}")
    counts = Counter(im.get("source", "unknown") for im in d["images"])
    if list(counts) == ["unknown"]:
        sys.exit("STOP: every image has source='unknown'. Merge first - the oversampler "
                 "reads img['source'] and would silently apply 1x to everything.")
    print("source counts:", dict(counts))
    names = sorted(counts)
    best = None
    for combo in itertools.product(range(1, a.max_repeat + 1), repeat=len(names)):
        reps = dict(zip(names, combo))
        if min(combo) != 1:          # someone must stay at 1x, else it is just scaling
            continue
        sh = shares(counts, reps)
        if not ok(sh):
            continue
        # Objective: GIS is the target domain for this retrain, so among
        # combinations that satisfy all three rules prefer the one giving GIS the
        # largest share; break ties toward the smaller corpus.
        key = (-sh.get("gis", 0.0), sum(counts[s] * reps[s] for s in names))
        if best is None or key < best[0]:
            best = (key, reps, sh)
    if best is None:
        print(f"\nNo repeat combination up to {a.max_repeat}x satisfies: {TARGETS}")
        print("Raise --max-repeat, or the corpus cannot hit the targets by multiplying alone.")
        sys.exit(1)
    _, reps, sh = best
    print(f"\ntargets: {TARGETS}")
    print(f"{'source':14}{'images':>9}{'repeat':>8}{'sampled':>10}{'share':>8}")
    for s in sorted(sh, key=lambda x: -sh[x]):
        print(f"{s:14}{counts[s]:>9}{reps.get(s,1):>8}{counts[s]*reps.get(s,1):>10}{sh[s]:>7.1f}%")
    print(f"\nGIS+NAIP {sh.get('gis',0)+sh.get('phase1',0):.1f}%   "
          f"max source {max(sh.values()):.1f}%   rid2 {sh.get('rid2',0):.1f}%")
    print("\n--repeat " + " ".join(f"{s}:{n}" for s, n in sorted(reps.items()) if n > 1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dedupe"); p.add_argument("--input", required=True)
    p.add_argument("--output", required=True); p.set_defaults(fn=cmd_dedupe)
    p = sub.add_parser("exclude"); p.add_argument("--input", required=True)
    p.add_argument("--output", required=True); p.add_argument("--eval", required=True)
    p.add_argument("--splits", nargs="+", default=["test"]); p.set_defaults(fn=cmd_exclude)
    p = sub.add_parser("check"); p.add_argument("--dataset", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--splits", nargs="+", default=["test"]); p.set_defaults(fn=cmd_check)
    p = sub.add_parser("remap"); p.add_argument("--input", required=True)
    p.add_argument("--output", required=True); p.set_defaults(fn=cmd_remap)
    p = sub.add_parser("plan"); p.add_argument("--dataset", required=True)
    p.add_argument("--max-repeat", type=int, default=6); p.set_defaults(fn=cmd_plan)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
