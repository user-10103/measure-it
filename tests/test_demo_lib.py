"""demo_lib is the Colab demo surface. It died with a disposable VM twice, so it
lives in the repo now — and its pure logic is tested so a reconstruction can't
drift silently. The pipeline-calling parts need a GPU and are not covered here."""
import io
from contextlib import redirect_stdout

import demo_lib


def test_address_id_matches_the_dataset_builder():
    """Must be the SAME hash the corpus used, or prove_unseen silently reports
    every address as unseen — a confident answer that means nothing."""
    import hashlib
    addr = "1600 Sarno Rd, Melbourne, FL 32935"
    expect = hashlib.sha1(addr.strip().lower().encode()).hexdigest()[:16]
    assert demo_lib.address_id(addr) == expect
    assert len(demo_lib.address_id(addr)) == 16
    # whitespace/case insensitive, as the builder is
    assert demo_lib.address_id("  1600 SARNO RD, MELBOURNE, FL 32935 ") == expect


def _row(addr, facets, passed=True, fails=None):
    return {"address": addr, "facets": facets, "pitched": facets,
            "outline": "found", "passed": passed, "fails": fails or [],
            "incomplete": not passed, "secs": 40, "out_dir": "/tmp", "chip": "x"}


def test_shortlist_drops_under_segmented_roofs_even_when_the_gate_passed():
    """The whole point of the eyeball step: the gate is blind to under-segmentation,
    so a roof the model called 1 facet and a human counted 6 must be dropped even
    though every check passed."""
    rows = [_row("under", 1), _row("good", 6), _row("failed", 4, passed=False,
                                                    fails=["edges_typed"])]
    counted = {"under": 6, "good": 6, "failed": 4}
    out = io.StringIO()
    with redirect_stdout(out):
        keep = demo_lib.shortlist(rows, counted)
    text = out.getvalue()
    assert [r["address"] for r in keep] == ["good"]
    assert "DROP under-segmented by 5" in text
    assert "drop (gate FAIL)" in text


def test_shortlist_holds_back_uncounted_addresses():
    rows = [_row("not-looked-at", 6)]
    out = io.StringIO()
    with redirect_stdout(out):
        keep = demo_lib.shortlist(rows, {})
    assert keep == []                      # never cleared without a human count
    assert "not counted yet" in out.getvalue()


def test_show_table_renders_errors_without_raising():
    rows = [_row("ok", 5), {"address": "bad", "error": "RuntimeError: no imagery"}]
    out = io.StringIO()
    with redirect_stdout(out):
        demo_lib.show_table(rows)
    text = out.getvalue()
    assert "ERR" in text and "ok" in text
