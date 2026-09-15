"""Rule 2: a module reachable only from tests/ is not wired, whatever its tests say.

Five subsystems were built, tested, and left running nowhere — plane_segment,
pearl_segment, fuse_facets, arrangement_input_from_sam, relabel_flat_junction_flashing
— and the git history contains NO commit that abandoned any of them. There was no
unwiring event. src/pipeline.py simply froze on 2026-07-03 and a parallel serving
path was built beside it the next day, leaving the modules standing with green
tests. As the audit put it: they "all still exist and still pass — which is
precisely why nobody noticed they had stopped running in production."

Green tests are not evidence of being wired. This test makes that visible, and it
is deliberately an INVENTORY rather than a hard failure: an orphan can be a
legitimate choice (a staged replacement, a tool used only by training scripts).
What is not legitimate is not knowing. Adding a module to src/ without a call
site now requires editing this list, which is the whole point.

It works in both directions. plane_segment.py was on this list when it was
written; wiring it made test_the_orphan_list_does_not_go_stale fail until the
entry was removed. A list that only ever grows would rot into the same silence
it exists to break.
"""
from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

# Modules with no call site inside src/, each with the reason it is tolerated.
# REMOVE an entry when you wire it; ADD one only with a reason.
# Entry points are legitimately not imported by anything: a CLI, an ASGI app, a
# script. They are reachable from OUTSIDE src/. Detect them rather than list them,
# so adding a new CLI never requires touching this file.
ENTRY_POINT_MARKERS = ("__main__", "FastAPI(", "app = ")


def _is_entry_point(path: pathlib.Path) -> bool:
    try:
        txt = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return any(m in txt for m in ENTRY_POINT_MARKERS)


# Modules with no call site inside src/, each with the reason it is tolerated.
# REMOVE an entry when you wire it; ADD one only with a reason.
KNOWN_ORPHANS = {
    "roofs/pearl_segment.py":
        "PEARL energy minimization. Needs gco/pygco, which is not declared or "
        "installed, so its graph-cut path has never executed.",
    "roofs/fuse_facets.py":
        "imagery shapes + LiDAR planes. Superseded by fuse_sam_lidar. Carries a "
        "live bug: compute_aspect_bin(slope, plane) — wrong arity AND wrong "
        "quantity — swallowed by a bare except, so aspect_bin is always None.",
    "lidar/shadow_cast.py":
        "VALIDATED shadow predictor (sun geometry + LiDAR DSM -> shadow pixels). "
        "Its own docstring lists 'discount facet edges that coincide with "
        "predicted shadow boundaries' as its first intended use. Never wired, "
        "and shadow is still an unexplained source of sawtooth eaves.",
    "roofs/pitch_mono.py":
        "monocular pitch estimation — no call site; predates the LiDAR path.",
    "rgb_pipeline.py":
        "the third pipeline. Reached only from scripts/ (render_maps_model.py, "
        "scripts/coord_to_report.py). Rule 3 says one pipeline: port or delete.",
}


def _module_names(path: pathlib.Path) -> set[str]:
    rel = path.relative_to(SRC).with_suffix("")
    return {rel.name, ".".join(("src",) + rel.parts)}


def _imports_in(path: pathlib.Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_every_src_module_is_reachable_or_listed():
    modules = [p for p in SRC.rglob("*.py")
               if p.name != "__init__.py" and "__pycache__" not in p.parts]
    imported: set[str] = set()
    for p in modules:
        imported |= _imports_in(p)

    orphans = []
    for p in modules:
        names = _module_names(p)
        if not (names & imported) and not _is_entry_point(p):
            orphans.append(str(p.relative_to(SRC)))

    unexpected = sorted(set(orphans) - set(KNOWN_ORPHANS))
    assert not unexpected, (
        "These src/ modules have no call site inside src/ — they are not wired, "
        "whatever their tests say. Wire them, delete them, or add them to "
        "KNOWN_ORPHANS with a reason:\n  " + "\n  ".join(unexpected))


def test_the_orphan_list_does_not_go_stale():
    """A module that got wired must be removed from the list, or the list stops
    meaning anything — which is exactly how the original five went unnoticed."""
    modules = {str(p.relative_to(SRC)) for p in SRC.rglob("*.py")}
    imported: set[str] = set()
    for p in SRC.rglob("*.py"):
        if p.name != "__init__.py" and "__pycache__" not in p.parts:
            imported |= _imports_in(p)

    for listed in KNOWN_ORPHANS:
        assert listed in modules, f"{listed} is listed as an orphan but no longer exists"
        p = SRC / listed
        assert not (_module_names(p) & imported) or _is_entry_point(p), (
            f"{listed} IS wired now — remove it from KNOWN_ORPHANS")
