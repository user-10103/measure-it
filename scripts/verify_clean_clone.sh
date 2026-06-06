#!/usr/bin/env bash
# Verify the COMMITTED tree is complete and importable.
#
# Why this exists: local `pytest` passes even when a whole module is untracked
# (the files are on disk; only git can't see them). That's how src/output/ was
# silently un-tracked by a .gitignore rule for several pushes. This clones the
# COMMITTED state into a temp dir -- where untracked files do NOT exist -- and
# proves the key modules import. Run it before any client-facing run / handoff.
#
# Usage:  bash scripts/verify_clean_clone.sh
# Run it where the runtime deps are installed (the pod, or an env with shapely/
# reportlab/etc). cv2/rfdetr are NOT needed -- rfdetr_backend imports them lazily.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> cloning COMMITTED tree of $REPO_ROOT (untracked files excluded by design)"
git clone --quiet "$REPO_ROOT" "$TMP/clone"
cd "$TMP/clone"
echo "    at commit $(git rev-parse --short HEAD)"

echo "==> import smoke -- the check local pytest can't do"
python - <<'PY'
import importlib, sys
mods = [
    "src.rgb_pipeline",
    "src.output.pdf_report", "src.output.diagram", "src.output.report_data",
    "src.output.units", "src.output.json_export", "src.output.csv_export",
    "src.roofs.tiling", "src.roofs.edges", "src.roofs.pitch_policy",
    "src.roofs.geometric_aspect", "src.roofs.segment", "src.roofs.metrics",
    "src.roofs.rfdetr_backend",          # lazy cv2/rfdetr -> imports without them
    "src.data.ls_to_coco", "src.eval.evaluate",
]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
        print("  ok  ", m)
    except Exception as e:
        print("  FAIL", m, "->", type(e).__name__, e)
        missing.append(m)
if missing:
    print(f"\n*** {len(missing)} module(s) failed to import from the committed tree ***")
    print("*** likely an untracked file or a missing dependency ***")
    sys.exit(1)
print("\nall key modules import from a clean clone")
PY

echo "==> pytest on the clean clone (skip ingestion -- needs geopandas)"
python -m pytest tests/ -q --ignore=tests/test_ingestion.py

echo "✓ clean clone is complete, importable, and green"
