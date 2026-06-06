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

print(f"=== Render convergence metrics  [{source}] ===")
print(f"Chips processed       : {total}")
print(f"With outline          : {with_out}")
print(f"Abstained (0 facets)  : {abstained}")
print(f"Abstention rate       : {abstention_rate:.1%}  (target: < 15%, falling each epoch)")
print(f"Total interior edges  : {total_edg}  (ridge+hip+valley only)")
print(f"Total facets          : {total_fac}")
print(f"Edges per facet       : {edges_per_facet:.1f}  (target: ~2-3 clean; ~6+ = fragmented)")
print()
print("Per-chip detail:")
print(f"  {'outline':>7}  {'facets':>6}  {'int_edges':>9}  {'e/f':>5}")
for o,f,e in chips:
    epf = f"{e/f:.1f}" if f else "—"
    abstain = " ← abstained" if (o and f == 0) else ""
    print(f"  {o:>7}  {f:>6}  {e:>9}  {epf:>5}{abstain}")
