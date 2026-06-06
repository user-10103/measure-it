#!/usr/bin/env python3
"""Compute convergence metrics from a render_maps_model.py run.

Reads render_maps.log (or any log from render_maps_model.py) and prints:
  - abstention_rate  : chips with 0 facets / total chips with outline
  - edges_per_facet  : total interior edges / total facets (target: ~2-3 clean, ~6 = fragmented)
  - interior_edges   : raw total (for trending)
  - chip_count       : chips processed

Usage (run after each render):
  python render_metrics.py render_maps.log
  python render_metrics.py render_maps_ep50.log
"""
import re, sys, pathlib

log = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "render_maps.log").read_text()

# Parse per-chip lines: "  1 outline  14 facets  85 interior edges"
pattern = re.compile(r'(\d+)\s+outline\s+(\d+)\s+facets?\s+(\d+)\s+interior')
chips = [(int(o), int(f), int(e)) for o, f, e in pattern.findall(log)]

if not chips:
    print("No chip data found — check log path")
    sys.exit(1)

total     = len(chips)
with_out  = sum(1 for o,f,e in chips if o > 0)
abstained = sum(1 for o,f,e in chips if o > 0 and f == 0)
total_fac = sum(f for o,f,e in chips)
total_edg = sum(e for o,f,e in chips)

abstention_rate   = abstained / with_out if with_out else float('nan')
edges_per_facet   = total_edg / total_fac if total_fac else float('nan')

print(f"=== Render convergence metrics ===")
print(f"Chips processed       : {total}")
print(f"With outline          : {with_out}")
print(f"Abstained (0 facets)  : {abstained}")
print(f"Abstention rate       : {abstention_rate:.1%}  (target: falling each epoch)")
print(f"Total interior edges  : {total_edg}")
print(f"Total facets          : {total_fac}")
print(f"Edges per facet       : {edges_per_facet:.1f}  (target: ~2-3 clean; ~6+ = fragmented segments)")
print()
print("Per-chip detail:")
print(f"  {'outline':>7}  {'facets':>6}  {'int_edges':>9}  {'e/f':>5}")
for o,f,e in chips:
    epf = f"{e/f:.1f}" if f else "—"
    print(f"  {o:>7}  {f:>6}  {e:>9}  {epf:>5}")
