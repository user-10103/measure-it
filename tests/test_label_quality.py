"""Quality-gate: clean labels pass, over-segmented/jagged labels fail."""
import numpy as np
from shapely.geometry import Polygon, box
from training.label_quality import score_labels


def _clean_hip():
    C = (20, 20)
    return [Polygon([(0, 0), (40, 0), C]), Polygon([(40, 0), (40, 40), C]),
            Polygon([(40, 40), (0, 40), C]), Polygon([(0, 40), (0, 0), C])]


def test_clean_labels_pass():
    m = score_labels(_clean_hip(), box(0, 0, 40, 40))
    assert m["passed"], m["reasons"]
    assert m["n_facets"] == 4 and m["coverage"] > 0.95


def test_oversegmented_fails():
    rng = np.random.default_rng(0)
    polys = []
    for _ in range(40):                                # 40 tiny random slivers
        x, y = rng.uniform(0, 38), rng.uniform(0, 38)
        polys.append(box(x, y, x + rng.uniform(0.5, 1.5), y + rng.uniform(0.5, 1.5)))
    m = score_labels(polys, box(0, 0, 40, 40))
    assert not m["passed"]
    assert any("n_facets" in r or "sliver" in r for r in m["reasons"])


def test_low_coverage_fails():
    # one small facet on a big roof -> big gap
    m = score_labels([box(0, 0, 8, 8), box(9, 0, 17, 8), box(0, 9, 8, 17)],
                     box(0, 0, 40, 40))
    assert not m["passed"]
    assert any("coverage" in r for r in m["reasons"])
