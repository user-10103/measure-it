#!/usr/bin/env python3
"""Compute convergence metrics from a render_maps_model.py output directory.

Reads <out_dir>/_summary.json (written by render_maps_model.py) and prints:
  - abstention_rate  : chips with 0 facets / chips with outline
  - edges_per_facet  : total interior edges / total facets (target: ~2-3 clean)
  - interior_edges   : raw total (hips/valleys/ridges only, not eave/rake)

Usage:
  python render_metrics.py model_maps/            # reads model_maps/_summary.json
  python render_metrics.py model_maps_ep50/       # epoch-50 comparison

Falls back to regex-parsing a log file if the JSON doesn't exist (legacy).
"""
import json, re, sys, pathlib

arg = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "model_maps")

# --- JSON path (preferred) ---
json_path = arg if arg.suffix == ".json" else arg / "_summary.json"

if json_path.exists():
    chips_raw = json.loads(json_path.read_text())
    chips = []
    for r in chips_raw:
        outline = 1 if r.get("outline") else 0
        facets  = r.get("facets_pred", 0)
        edges_d = r.get("edges", {})
        # interior = ridge + hip + valley (not eave/rake which are perimeter)
        interior = sum(v for k, v in edges_d.items()
                       if k.lower() in ("ridge", "hip", "valley"))
        chips.append((outline, facets, interior))
    source = f"JSON: {json_path}"
else:
    # --- Fallback: regex log parsing ---
    log_path = arg if arg.suffix in (".log", ".txt") else arg / "render.log"
    if not log_path.exists():
        print(f"Neither {json_path} nor {log_path} found")
        sys.exit(1)
    log = log_path.read_text()
    pattern = re.compile(r'(\d+)\s+outline\s+(\d+)\s+facets?\s+(\d+)\s+interior')
    chips = [(int(o), int(f), int(e)) for o, f, e in pattern.findall(log)]
    source = f"log regex: {log_path}"

if not chips:
    print(f"No chip data found in {source}")
    sys.exit(1)

total     = len(chips)
with_out  = sum(1 for o,f,e in chips if o > 0)
abstained = sum(1 for o,f,e in chips if o > 0 and f == 0)
total_fac = sum(f for o,f,e in chips)
total_edg = sum(e for o,f,e in chips)

abstention_rate = abstained / with_out if with_out else float('nan')
edges_per_facet = total_edg / total_fac if total_fac else float('nan')

# --- Edge type aggregation from JSON (Gate 3) ---
type_totals = {}
chips_with_eave = 0
chips_interior_only_one_type = []
chip_type_data = []  # (chip_id, outline, facets, interior, edges_dict)

if json_path.exists():
    for r in chips_raw:
        o = 1 if r.get("outline") else 0
        f = r.get("facets_pred", 0)
        edges_d = r.get("edges", {})
        interior = sum(v for k, v in edges_d.items() if k.lower() in ("ridge", "hip", "valley"))
        chip_id = r.get("chip", "?")
        chip_type_data.append((chip_id, o, f, interior, edges_d))
        for k, v in edges_d.items():
            type_totals[k] = type_totals.get(k, 0) + v
        if edges_d.get("eave", 0) + edges_d.get("rake", 0) > 0:
            chips_with_eave += 1
        if f > 0 and interior > 0:
            int_types = {k: v for k, v in edges_d.items() if k.lower() in ("ridge", "hip", "valley") and v > 0}
            if len(int_types) == 1:
                chips_interior_only_one_type.append((chip_id, list(int_types.keys())[0], interior))
else:
    chip_type_data = [("?", o, f, e, {}) for o, f, e in chips]

print(f"=== Render convergence metrics  [{source}] ===")
print(f"Chips processed       : {total}")
print(f"With outline          : {with_out}")
print(f"Abstained (0 facets)  : {abstained}")
print(f"Abstention rate       : {abstention_rate:.1%}  (target: < 15%, falling each epoch)")
print(f"Total interior edges  : {total_edg}  (ridge+hip+valley only)")
print(f"Total facets          : {total_fac}")
print(f"Edges per facet       : {edges_per_facet:.1f}  (target: ~2-3 clean; ~6+ = fragmented)")
print()

if type_totals:
    print("Edge type distribution (Gate 3):")
    max_n = max(type_totals.values()) if type_totals else 1
    for t in ("eave", "rake", "hip", "valley", "ridge"):
        n = type_totals.get(t, 0)
        bar = "█" * min(40, n * 40 // max(max_n, 1))
        print(f"  {t:>7}: {n:5d}  {bar}")
    print()
    eave_pct = chips_with_eave / with_out if with_out else 0
    print(f"Chips with eaves       : {chips_with_eave}/{with_out} ({eave_pct:.0%})  (should be ~100%)")
    if chips_interior_only_one_type:
        print(f"WARNING: {len(chips_interior_only_one_type)} chip(s) have all interior edges one type (mis-typing?):")
        for cid, t, n in chips_interior_only_one_type[:5]:
            print(f"     {cid[:12]}  all-{t} ({n} edges)")
    else:
        print("Interior type mix      : OK (no chip is all-one-type)")
    print()

print("Per-chip detail:")
print(f"  {'chip':>16} {'out':>3} {'fac':>4} {'int':>5} {'e/f':>5}  edge types")
for chip_id, o, f, e, ed in chip_type_data:
    epf = f"{e/f:.1f}" if f else "—"
    abstain = " ← abstained" if (o and f == 0) else ""
    type_str = " ".join(f"{k}:{v}" for k, v in sorted(ed.items()) if v > 0)
    print(f"  {chip_id[:16]:>16}  {o:>3}  {f:>4}  {e:>5}  {epf:>5}  {type_str}{abstain}")
