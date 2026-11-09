"""
Tests for src/utils/report_ops.py, the three format builders (DOCX, PDF,
XLSX), and the Report Generation Agent end-to-end. Files are actually opened
back up and inspected (python-docx, openpyxl) rather than just checking they
exist on disk.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from docx import Document
from openpyxl import load_workbook

from src.agents.report_generation_agent import ReportGenerationAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.bi_bundle import BIBundle
from src.utils.profiling import profile_dataframe
from src.utils.report_docx_builder import build_docx
from src.utils.report_ops import ReportSection, build_report_sections, split_inline_bold
from src.utils.report_pdf_builder import build_pdf
from src.utils.report_xlsx_builder import build_xlsx

# ----------------------------------------------------------------------
# split_inline_bold
# ----------------------------------------------------------------------


def test_split_inline_bold_basic():
    result = split_inline_bold("A **B** C")
    assert result == [("A ", False), ("B", True), (" C", False)]


def test_split_inline_bold_no_markers():
    assert split_inline_bold("plain text") == [("plain text", False)]


def test_split_inline_bold_multiple_bold_segments():
    result = split_inline_bold("**X** and **Y**")
    bold_segments = [seg for seg, is_bold in result if is_bold]
    assert bold_segments == ["X", "Y"]


def test_split_inline_bold_priority_tag_style():
    # the exact real-world case that was visibly broken: "**[HIGH]** text"
    result = split_inline_bold("**[HIGH]** Increase spend.")
    assert result[0] == ("[HIGH]", True)


# ----------------------------------------------------------------------
# build_report_sections
# ----------------------------------------------------------------------


def _sample_bundle() -> BIBundle:
    bundle = BIBundle()
    bundle.tables["insights"] = pd.DataFrame({"text": ["Revenue up 10%"], "metric": ["revenue"], "pct_change": [10.0], "severity": ["high"]})
    bundle.tables["recommendations"] = pd.DataFrame({"action_text": ["Do X"], "priority": ["high"], "rationale": ["Because Y"]})
    bundle.executive_summary = "## Summary\n\n- Revenue up 10%."
    bundle.ml_model_summary = {"best_model_name": "random_forest", "problem_type": "regression", "best_model_metrics": {"rmse": 1.0}}
    bundle.forecasting_summary = {"best_model_name": "prophet", "best_model_metrics": {"rmse": 2.0}}
    bundle.anomaly_ensemble_summary = {"n_consensus_anomalies": 5, "pct_consensus_anomalies": 2.5, "min_votes_for_consensus": 2, "methods_used": ["a", "b", "c", "d"]}
    return bundle


def test_build_report_sections_includes_kpi_grid_first():
    sections = build_report_sections(_sample_bundle(), {"total_revenue": 100.0})
    assert sections[0].kind == "kpi_grid"


def test_build_report_sections_includes_executive_summary():
    sections = build_report_sections(_sample_bundle(), {})
    titles = [s.title for s in sections]
    assert "Executive Summary" in titles


def test_build_report_sections_respects_max_insights():
    bundle = _sample_bundle()
    bundle.tables["insights"] = pd.DataFrame({"text": [f"insight {i}" for i in range(20)]})
    sections = build_report_sections(bundle, {}, max_insights=5)
    insight_section = next(s for s in sections if s.title == "Key Insights")
    assert len(insight_section.content) == 5


def test_build_report_sections_skips_empty_tables():
    bundle = BIBundle()  # nothing populated
    sections = build_report_sections(bundle, {"total_revenue": 1.0})
    titles = [s.title for s in sections]
    assert "Key Insights" not in titles
    assert "Recommended Actions" not in titles


def test_build_report_sections_includes_anomaly_summary_text():
    sections = build_report_sections(_sample_bundle(), {})
    anomaly_section = next((s for s in sections if s.title == "Anomaly Summary"), None)
    assert anomaly_section is not None
    assert "5" in anomaly_section.content


# ----------------------------------------------------------------------
# DOCX builder
# ----------------------------------------------------------------------


def test_build_docx_creates_readable_file(tmp_path):
    sections = [
        ReportSection("KPIs", "kpi_grid", {"total_revenue": 100.0, "total_orders": 5}),
        ReportSection("Summary", "text", "## Heading\n\n- **[HIGH]** bullet point"),
        ReportSection("A Table", "table", pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})),
    ]
    path = build_docx(sections, tmp_path / "report.docx", "Test", "2024-01-01")
    assert path.exists()

    doc = Document(str(path))
    full_text = "\n".join(p.text for p in doc.paragraphs)
    assert "Test" in full_text
    assert "Heading" in full_text
    assert "[HIGH]" in full_text
    assert len(doc.tables) >= 2  # KPI grid table + the data table


def test_build_docx_bold_run_for_priority_tag(tmp_path):
    sections = [ReportSection("S", "text", "- **[HIGH]** urgent thing")]
    path = build_docx(sections, tmp_path / "r.docx", "T", "now")
    doc = Document(str(path))
    bold_runs = [r.text for p in doc.paragraphs for r in p.runs if r.bold]
    assert "[HIGH]" in bold_runs


def test_build_docx_handles_missing_image_gracefully(tmp_path):
    sections = [ReportSection("", "image", "/nonexistent/path.png")]
    path = build_docx(sections, tmp_path / "r.docx", "T", "now")
    assert path.exists()  # doesn't crash


def test_build_docx_truncates_long_tables(tmp_path):
    df = pd.DataFrame({"x": range(50)})
    sections = [ReportSection("Big Table", "table", df)]
    path = build_docx(sections, tmp_path / "r.docx", "T", "now")
    doc = Document(str(path))
    full_text = "\n".join(p.text for p in doc.paragraphs)
    assert "first 20 of 50" in full_text


# ----------------------------------------------------------------------
# PDF builder
# ----------------------------------------------------------------------


def test_build_pdf_creates_nonempty_file(tmp_path):
    sections = [
        ReportSection("KPIs", "kpi_grid", {"total_revenue": 100.0}),
        ReportSection("Summary", "text", "## Heading\n\n- **[HIGH]** bullet point"),
    ]
    path = build_pdf(sections, tmp_path / "report.pdf", "Test", "2024-01-01")
    assert path.exists()
    assert path.stat().st_size > 500  # a real PDF has real bytes, not a stub


def test_build_pdf_handles_missing_image_gracefully(tmp_path):
    sections = [ReportSection("", "image", "/nonexistent/path.png")]
    path = build_pdf(sections, tmp_path / "r.pdf", "T", "now")
    assert path.exists()


def test_build_pdf_handles_empty_table(tmp_path):
    sections = [ReportSection("Empty", "table", pd.DataFrame())]
    path = build_pdf(sections, tmp_path / "r.pdf", "T", "now")
    assert path.exists()


# ----------------------------------------------------------------------
# XLSX builder
# ----------------------------------------------------------------------


def test_build_xlsx_creates_expected_sheets(tmp_path):
    sheets = {
        "insights": pd.DataFrame({"text": ["a"]}),
        "empty": pd.DataFrame(),
    }
    path = build_xlsx(tmp_path / "r.xlsx", {"total_revenue": 100.0}, sheets, "## Summary")
    wb = load_workbook(path)
    assert "Summary" in wb.sheetnames
    assert "KPIs" in wb.sheetnames
    assert "Insights" in wb.sheetnames
    assert "Empty" not in wb.sheetnames  # empty tables are skipped


def test_build_xlsx_handles_list_valued_cells_after_parquet_roundtrip(tmp_path):
    # the exact real-world bug: a list column becomes a numpy array after a
    # parquet round-trip, and pd.isna() on an array raises ValueError
    df = pd.DataFrame({"top_features": [["price", "freight"]]})
    df.to_parquet(tmp_path / "t.parquet")
    reloaded = pd.read_parquet(tmp_path / "t.parquet")
    path = build_xlsx(tmp_path / "r.xlsx", {}, {"features": reloaded}, "")
    wb = load_workbook(path)
    ws = wb["Features"]
    assert ws.cell(row=2, column=1).value == "price, freight"


def test_build_xlsx_handles_numpy_scalar_types(tmp_path):
    df = pd.DataFrame({"a": [np.int64(5), np.int64(6)], "b": [np.float64(1.5), np.nan]})
    path = build_xlsx(tmp_path / "r.xlsx", {}, {"nums": df}, "")
    wb = load_workbook(path)
    ws = wb["Nums"]
    assert ws.cell(row=2, column=1).value == 5
    assert ws.cell(row=3, column=2).value is None


def test_build_xlsx_dedupes_long_sheet_names(tmp_path):
    sheets = {
        "a_very_long_sheet_name_one": pd.DataFrame({"x": [1]}),
        "a_very_long_sheet_name_two": pd.DataFrame({"x": [2]}),
    }
    path = build_xlsx(tmp_path / "r.xlsx", {}, sheets, "")
    wb = load_workbook(path)
    assert len(set(wb.sheetnames)) == len(wb.sheetnames)  # no duplicate names
    assert all(len(name) <= 31 for name in wb.sheetnames)


# ----------------------------------------------------------------------
# Report Generation Agent end-to-end test
# ----------------------------------------------------------------------

FIXTURE_ROOT = Path("tests/_fixture_report")
INSIGHT_DIR = FIXTURE_ROOT / "insight_upstream"
OUTPUT_ROOT = FIXTURE_ROOT / "processed"
CONFIG_PATH = FIXTURE_ROOT / "config.yaml"


@pytest.fixture()
def insight_manifest_path():
    INSIGHT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(20)],
        "item_total_value": np.linspace(50, 150, 20),
        "delivery_days": np.linspace(1, 10, 20),
        "customer_id": [f"c{i % 5}" for i in range(20)],
        "order_purchase_timestamp": pd.date_range("2024-01-01", periods=20, freq="D"),
    })
    rel_path = "unified_cleaned.parquet"
    df.to_parquet(INSIGHT_DIR / rel_path, index=False)
    profile = profile_dataframe("unified_cleaned", df, rel_path)

    insights = pd.DataFrame({"insight_id": [0], "type": ["revenue_trend_up"], "text": ["Revenue increased by 10%."], "severity": ["high"]})
    insights_rel = "insights.parquet"
    insights.to_parquet(INSIGHT_DIR / insights_rel, index=False)

    recommendations = pd.DataFrame({"recommendation_id": [0], "action_text": ["Stock up."], "priority": ["high"], "rationale": ["Because revenue is up."]})
    recs_rel = "recommendations.parquet"
    recommendations.to_parquet(INSIGHT_DIR / recs_rel, index=False)

    (INSIGHT_DIR / "executive_summary.md").write_text("## Business Performance Summary\n\n- **[HIGH]** Test summary.")

    manifest = DatasetManifest(
        agent_name="insight_recommendation",
        status=AgentStatus.SUCCESS,
        output_dir=str(INSIGHT_DIR),
        primary_table="unified_cleaned",
        tables={
            "unified_cleaned": profile,
            "insights": profile_dataframe("insights", insights, insights_rel),
            "recommendations": profile_dataframe("recommendations", recommendations, recs_rel),
        },
        extra_metadata={"chart_paths": []},
    )
    path = manifest.save()

    config = {
        "report_generation": {
            "report_title": "Test Business Report",
            "kpi_date_column": "order_purchase_timestamp",
            "kpi_value_column": "item_total_value",
            "formats": ["docx", "pdf", "xlsx"],
            "max_insights": 15,
            "max_recommendations": 10,
            "max_table_rows": 20,
            "max_charts": 6,
        }
    }
    CONFIG_PATH.write_text(yaml.dump(config))

    yield path

    shutil.rmtree(FIXTURE_ROOT, ignore_errors=True)


def test_report_agent_requires_upstream_manifest():
    agent = ReportGenerationAgent(config_path="config/config.yaml", output_root=str(OUTPUT_ROOT))
    with pytest.raises(ValueError):
        agent.run(upstream_manifest_path=None)


def test_report_agent_generates_all_three_formats(insight_manifest_path):
    agent = ReportGenerationAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=insight_manifest_path)
    assert set(manifest.extra_metadata["formats_generated"]) == {"docx", "pdf", "xlsx"}
    for fmt, path in manifest.extra_metadata["report_paths"].items():
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0


def test_report_agent_respects_formats_config(insight_manifest_path):
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["report_generation"]["formats"] = ["xlsx"]
    CONFIG_PATH.write_text(yaml.dump(config))

    agent = ReportGenerationAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=insight_manifest_path)
    assert manifest.extra_metadata["formats_generated"] == ["xlsx"]


def test_report_agent_docx_contains_title_and_bold_priority(insight_manifest_path):
    agent = ReportGenerationAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=insight_manifest_path)
    docx_path = manifest.extra_metadata["report_paths"]["docx"]
    doc = Document(docx_path)
    full_text = "\n".join(p.text for p in doc.paragraphs)
    assert "Test Business Report" in full_text
    bold_runs = [r.text for p in doc.paragraphs for r in p.runs if r.bold]
    assert "[HIGH]" in bold_runs  # the executive summary's priority tag rendered bold, not literal


def test_report_agent_passes_through_primary_table(insight_manifest_path):
    agent = ReportGenerationAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=insight_manifest_path)
    df = pd.read_parquet(manifest.resolve_table_path("unified_cleaned"))
    assert len(df) == 20
