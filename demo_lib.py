"""Demo helpers for the facet sweep.

Kept out of the notebook so the presented cell stays short and readable.
"""
from __future__ import annotations

import glob
import hashlib
import pathlib
import subprocess
import time

TRAIN_GLOB = "/content/code/training/roof_dataset/*/chips_needed.txt"


def address_id(addr: str) -> str:
    """Same hash the dataset builder used (build_pseudo_dataset.py:55)."""
    return hashlib.sha1(addr.strip().lower().encode()).hexdigest()[:16]


def _corpus_stems() -> set:
    stems = set()
    for path in glob.glob(TRAIN_GLOB):
        with open(path) as handle:
            stems |= {ln.strip().split(".")[0] for ln in handle if ln.strip()}
    return stems


def prove_unseen(addresses):
    """Print, per address, whether it appears in the training corpus."""
    stems = _corpus_stems()
    print(f"training corpus: {len(stems)} unique chip stems\n")
    unseen = []
    for addr in addresses:
        digest = address_id(addr)
        seen = digest in stems
        label = "SEEN" if seen else "unseen"
        print(f"  {label:<7}{digest}  {addr}")
        if not seen:
            unseen.append(addr)
    print(f"\n{len(unseen)} of {len(addresses)} addresses are unseen by the model.")
    return unseen


def _banner_incomplete(pdf_path) -> bool:
    """True if page 1 carries the INCOMPLETE stamp added in 589600b."""
    if not pdf_path:
        return False
    try:
        done = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "1", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return False
    return "INCOMPLETE" in done.stdout.upper()


def run_sweep(addresses, predict_facets, predict_outline, state=None,
              out_root="/content/demo", use_lidar=True):
    # state=None -> derived per address from its coordinates. It used to default
    # to "FL", which silently sent every out-of-state address to the Florida
    # NAIP archive; the national probe had to work around it per-address.
    """Run the full pipeline per address. Returns one dict per address."""
    from src.serve.report_service import generate_roof_report

    root = pathlib.Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for addr in addresses:
        digest = address_id(addr)
        out_dir = root / digest
        started = time.time()
        row = {"address": addr, "id": digest, "out_dir": str(out_dir)}
        try:
            res = generate_roof_report(
                addr, state, predict_facets, predict_outline,
                out_dir=str(out_dir), use_lidar=use_lidar, label=addr,
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:140]
            row["secs"] = round(time.time() - started)
            rows.append(row)
            print(f"  ERROR   {addr}")
            print(f"          {row['error']}")
            continue
        checks = res.qc.get("checks", [])
        row.update(
            facets=res.num_facets,
            pitched=res.num_pitched,
            outline="found" if res.outline_found else "none",
            passed=bool(res.qc.get("passed")),
            fails=[c["id"] for c in checks
                   if not c.get("ok") and c.get("severity") == "FAIL"],
            pdf=res.pdf_path,
            chip=res.chip_path,
            incomplete=_banner_incomplete(res.pdf_path),
            secs=round(time.time() - started),
            eyeball=None,
        )
        rows.append(row)
        gate = "PASS" if row["passed"] else "FAIL"
        print(f"  done    {addr}")
        print(f"          facets={row['facets']} pitched={row['pitched']} "
              f"gate={gate} ({row['secs']}s)")
    return rows


COLS = "{a:<40}{f:>7}{p:>7}{o:>13}{g:>6}{i:>12}  {n}"


def show_table(rows):
    """One line per address: what the model said and what the gate said."""
    header = COLS.format(a="address", f="facets", p="pitched", o="outline",
                         g="gate", i="INCOMPLETE", n="failing checks")
    print(header)
    print("-" * len(header))
    for row in rows:
        if "error" in row:
            print(COLS.format(a=row["address"][:39], f="-", p="-", o="-",
                              g="ERR", i="-", n=row["error"][:34]))
            continue
        print(COLS.format(
            a=row["address"][:39],
            f=row["facets"],
            p=row["pitched"],
            o=row["outline"],
            g="PASS" if row["passed"] else "FAIL",
            i="yes" if row["incomplete"] else "no",
            n=",".join(row["fails"])[:34] or "-",
        ))


def show_overlays(rows, width=820):
    """Render each facet overlay big enough to count the planes by eye."""
    from IPython.display import display, HTML, Image

    for row in rows:
        if "error" in row:
            continue
        overlay = pathlib.Path(row["out_dir"]) / "facets.png"
        image = overlay if overlay.exists() else pathlib.Path(row["chip"])
        display(HTML(
            f"<h3 style='margin:18px 0 2px'>{row['address']}</h3>"
            f"<div style='color:#888;margin-bottom:6px'>model says "
            f"{row['facets']} facets &mdash; count them yourself</div>"
        ))
        display(Image(filename=str(image), width=width))


def shortlist(rows, counted, margin=2):
    """counted: {address: facets_you_counted}. Drops under-segmented roofs."""
    fmt = "{a:<40}{m:>7}{e:>5}{g:>6}  {v}"
    header = fmt.format(a="address", m="model", e="eye", g="gate", v="verdict")
    print(header)
    print("-" * len(header))
    keep = []
    for row in rows:
        if "error" in row:
            print(fmt.format(a=row["address"][:39], m="-", e="-", g="ERR",
                             v="drop (errored)"))
            continue
        eye = counted.get(row["address"])
        row["eyeball"] = eye
        if eye is None:
            verdict = "not counted yet"
        elif row["facets"] <= eye - margin:
            verdict = f"DROP under-segmented by {eye - row['facets']}"
        elif not row["passed"]:
            verdict = "drop (gate FAIL)"
        else:
            verdict = "KEEP"
            keep.append(row)
        print(fmt.format(a=row["address"][:39], m=row["facets"],
                         e="-" if eye is None else eye,
                         g="PASS" if row["passed"] else "FAIL", v=verdict))
    print(f"\n{len(keep)} address(es) cleared for the demo.")
    return keep


def merge_pdfs(rows, dest="/content/demo/demo_reports.pdf"):
    """Merge the kept reports into one file for the demo."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    count = 0
    for row in rows:
        if row.get("pdf"):
            writer.append(row["pdf"])
            count += 1
    writer.write(dest)
    writer.close()
    print(f"merged {count} report(s) -> {dest}")
    return dest


# Brevard County, FL - strong 3DEP LiDAR coverage.
# 1250 Pineapple is the known-bad control: it previously shipped a polished but
# wrong report, so it must now come back stamped INCOMPLETE.
DEMO_ADDRESSES = [
    "1250 Pineapple Ave, Melbourne, FL 32935",
    "1600 Sarno Rd, Melbourne, FL 32935",
    "3100 N Wickham Rd, Melbourne, FL 32935",
    "1502 S Harbor City Blvd, Melbourne, FL 32901",
    "2725 Judge Fran Jamieson Way, Viera, FL 32940",
    "1350 S Patrick Dr, Satellite Beach, FL 32937",
    "755 E Eau Gallie Blvd, Indian Harbour Beach, FL 32937",
]


def _facts_from_pdf(pdf_path):
    """Headline measurements, read back out of the delivered PDF."""
    import re
    try:
        text = subprocess.run(["pdftotext", "-layout", str(pdf_path), "-"],
                              capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return {}
    keys = ["Total roof area", "Total pitched area", "Total flat area",
            "Total roof facets", "Predominant pitch", "Total ridges",
            "Total hips", "Total eaves", "Total valleys"]
    out = {}
    for key in keys:
        m = re.search(re.escape(key) + r"\s{2,}([^\n]+)", text)
        out[key] = m.group(1).strip() if m else "-"
    return out


def report_facts(rows):
    """Pull the headline measurements straight out of each delivered PDF."""
    fmt = "{a:<40}{ar:>10}{pa:>10}{fa:>10}{pp:>9}{rg:>10}{hp:>10}"
    header = fmt.format(a="address", ar="area", pa="pitched", fa="flat",
                        pp="pitch", rg="ridges", hp="hips")
    print(header)
    print("-" * len(header))
    for row in rows:
        if not row.get("pdf"):
            continue
        got = _facts_from_pdf(row["pdf"])
        row["facts"] = got
        print(fmt.format(
            a=row["address"][:39],
            ar=got.get("Total roof area", "-").replace(" sqft", ""),
            pa=got.get("Total pitched area", "-").replace(" sqft", ""),
            fa=got.get("Total flat area", "-").replace(" sqft", ""),
            pp=got.get("Predominant pitch", "-"),
            rg=got.get("Total ridges", "-"),
            hp=got.get("Total hips", "-"),
        ))
    return rows


def live_report(address, predict_facets, predict_outline, state=None,
                out_root="/content/live", use_lidar=True, dpi=110):
    """One address in, a rendered report out. Never raises in front of a client."""
    import io
    import logging
    from contextlib import redirect_stdout, redirect_stderr
    from IPython.display import display, HTML

    address = (address or "").strip()
    if not address:
        display(HTML("<h3 style='color:#b00'>Enter an address first.</h3>"))
        return None

    display(HTML(
        f"<h2 style='margin:6px 0 2px'>{address}</h2>"
        "<div style='color:#777;margin-bottom:10px'>geocode &rarr; NAIP imagery &rarr; "
        "SAM3 facets &rarr; 3DEP LiDAR pitch &rarr; report &rarr; quality gate</div>"
    ))

    # Client-facing: silence library chatter so the output pane stays clean.
    # NOTE: this also hides our own WARNING-level safeguards (LiDAR attribution,
    # under-segmentation, HF cache misconfiguration). That is acceptable ONLY
    # because the delivered PDF carries the verdict on its cover. When debugging,
    # call run_sweep() directly instead of live_report().
    logging.getLogger().setLevel(logging.ERROR)
    for name in list(logging.root.manager.loggerDict):
        logging.getLogger(name).setLevel(logging.ERROR)

    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            rows = run_sweep([address], predict_facets, predict_outline,
                             state=state, out_root=out_root, use_lidar=use_lidar)
    except Exception as exc:
        display(HTML(
            "<h3 style='color:#b00'>Could not complete this address</h3>"
            f"<div style='color:#777'>{type(exc).__name__}</div>"
        ))
        return None
    return _render_live(rows[0] if rows else None, dpi)


def _render_live(row, dpi=110):
    """Client-facing rendering: verdict banner, headline numbers, aerial, report."""
    from IPython.display import display, HTML, Image
    from pdf2image import convert_from_path

    if row is None or "error" in row:
        detail = row.get("error", "no result") if row else "no result"
        display(HTML(
            "<div style='border-left:5px solid #b00;padding:10px 14px;background:#fff4f4'>"
            "<b style='color:#b00'>Could not complete this address.</b>"
            f"<div style='color:#777;margin-top:4px'>{detail}</div></div>"))
        return row

    facts = _facts_from_pdf(row.get("pdf"))
    row["facts"] = facts

    if row["passed"]:
        banner = ("<div style='border-left:5px solid #197a5a;padding:10px 14px;"
                  "background:#f2fbf7'><b style='color:#197a5a'>QUALITY GATE: PASS</b>"
                  "<div style='color:#777;margin-top:4px'>Every check cleared. "
                  f"Completed in {row['secs']}s.</div></div>")
    else:
        why = ", ".join(row["fails"]) or "quality checks"
        banner = ("<div style='border-left:5px solid #b36b00;padding:10px 14px;"
                  "background:#fff9f0'><b style='color:#b36b00'>INCOMPLETE &mdash; "
                  "MANUAL REVIEW REQUIRED</b><div style='color:#777;margin-top:4px'>"
                  f"Failed: {why}. The report is still produced, but it is stamped "
                  f"rather than shipped as finished. Completed in {row['secs']}s."
                  "</div></div>")
    display(HTML(banner))

    order = ["Total roof area", "Predominant pitch", "Total roof facets",
             "Total pitched area", "Total flat area", "Total ridges",
             "Total hips", "Total eaves"]
    cells = ""
    for key in order:
        cells += ("<div style='display:inline-block;min-width:158px;margin:8px 20px 8px 0'>"
                  f"<div style='color:#888;font-size:12px'>{key}</div>"
                  f"<div style='font-size:19px'>{facts.get(key, '-')}</div></div>")
    display(HTML(f"<div style='margin:14px 0'>{cells}</div>"))

    chip = row.get("chip")
    if chip and pathlib.Path(chip).exists():
        display(HTML("<div style='color:#888;font-size:12px;margin-top:6px'>aerial</div>"))
        display(Image(filename=str(chip), width=560))

    pdf = row.get("pdf")
    if pdf and pathlib.Path(pdf).exists():
        display(HTML("<div style='color:#888;font-size:12px;margin-top:12px'>report</div>"))
        try:
            for page in convert_from_path(str(pdf), dpi=dpi):
                display(page)
        except Exception:
            display(HTML(f"<div style='color:#777'>Report saved at {pdf}</div>"))
    return row
