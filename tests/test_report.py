"""Tests for the roof report: units, aggregation model, and PDF/JSON/CSV export."""
import json
import math
import os

from src.output.units import m2_to_sqft, m_to_ftin, sqft_to_squares
from src.output.report_data import build_report_model
from src.output.pdf_report import generate_report
from src.output.json_export import export_report_json
from src.output.csv_export import export_facets_csv, export_edges_csv


def _report_input():
    return {
        "address": "110 Test Lane, Lutz, FL",
        "building_id": "bldg_test",
        "aerial_image_path": None,
        "north_angle_deg": 0.0,
        "facets": [
            {"facet_id": 0, "polygon_xy": [[0, 0], [10, 0], [10, 5], [0, 5]],
             "plan_area_m2": 50.0, "slope_deg": 18.0, "pitch_string": "4:12",
             "aspect_bin": "S", "is_flat": False},
            {"facet_id": 1, "polygon_xy": [[0, 5], [10, 5], [10, 10], [0, 10]],
             "plan_area_m2": 50.0, "slope_deg": 18.0, "pitch_string": "4:12",
             "aspect_bin": "N", "is_flat": False},
            {"facet_id": 2, "polygon_xy": [[10, 0], [20, 0], [20, 10], [10, 10]],
             "plan_area_m2": 40.0, "slope_deg": 0.0, "pitch_string": "0:12",
             "aspect_bin": "flat", "is_flat": True},
        ],
        "edges": [
            {"edge_type": "ridge", "length_m": 10.0, "geometry_xy": [[0, 5], [10, 5]]},
            {"edge_type": "eave", "length_m": 10.0, "geometry_xy": [[0, 0], [10, 0]]},
        ],
    }


def test_units():
    assert math.isclose(m2_to_sqft(100), 1076.39, abs_tol=0.1)
    assert m_to_ftin(155.135) == "509ft 0in" or m_to_ftin(155.135).endswith("in")
    assert sqft_to_squares(6086) == 60.9


def test_report_model_totals():
    m = build_report_model(_report_input())
    expected_total = m2_to_sqft(50 + 50 + 40)
    assert math.isclose(m.total_area_sqft, expected_total, abs_tol=0.1)
    # pitched + flat == total
    assert math.isclose(m.pitched_area_sqft + m.flat_area_sqft, m.total_area_sqft, abs_tol=0.1)
    # predominant pitch is 4:12 (100 m2 vs 40 m2 flat)
    assert m.predominant_pitch == "4:12"
    assert math.isclose(m.predominant_pitch_area_sqft, m2_to_sqft(100), abs_tol=0.1)
    # waste table: 0% == total, 10% == total*1.1, squares == area/100
    w0 = m.waste_table[0]
    assert math.isclose(w0["area_sqft"], m.total_area_sqft, abs_tol=0.1)
    w10 = next(w for w in m.waste_table if w["pct"] == 10)
    assert math.isclose(w10["area_sqft"], m.total_area_sqft * 1.1, abs_tol=0.1)
    assert math.isclose(w10["squares"], round(w10["area_sqft"] / 100, 1), abs_tol=0.05)
    # edge totals present for ridge + eave
    assert m.edge_totals_ftin["ridge"].endswith("in")
    assert m.edge_totals_ftin["eave"].endswith("in")


def test_generate_pdf_and_exports(tmp_path):
    ri = _report_input()
    pdf = str(tmp_path / "report.pdf")
    generate_report(ri, pdf)
    assert os.path.getsize(pdf) > 1000

    js = str(tmp_path / "report.json")
    export_report_json(ri, js)
    data = json.load(open(js))
    assert data["summary"]["num_facets"] == 3
    assert len(data["facets"]) == 3

    fcsv = str(tmp_path / "facets.csv")
    export_facets_csv(ri["facets"], fcsv)
    assert len(open(fcsv).read().strip().splitlines()) == 1 + 3  # header + 3 rows

    ecsv = str(tmp_path / "edges.csv")
    export_edges_csv(ri["edges"], ecsv)
    assert len(open(ecsv).read().strip().splitlines()) == 1 + 2
