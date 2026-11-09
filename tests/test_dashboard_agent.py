"""
Tests for src/utils/dashboard_ops.py and the Dashboard Agent end-to-end,
including actually executing the generated Streamlit app via Streamlit's
AppTest harness — not just checking that files were written.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.agents.dashboard_agent import DashboardAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.dashboard_ops import compute_kpis, resolve_chart_paths, to_json_records
from src.utils.profiling import profile_dataframe


def _write_tiny_png(path: Path) -> None:
    """Writes a real, minimal valid PNG — a fixture using fake bytes here
    would make a test that exercises st.image() fail for a reason that has
    nothing to do with the code under test (the real pipeline always writes
    genuine PNGs via matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([0, 1], [0, 1])
    fig.savefig(path)
    plt.close(fig)

# ----------------------------------------------------------------------
# Pure function tests
# ----------------------------------------------------------------------


def test_compute_kpis_basic_aggregates():
    df = pd.DataFrame({
        "item_total_value": [100.0, 200.0, 300.0],
        "order_id": ["o1", "o1", "o2"],
        "delivery_days": [5.0, np.nan, 10.0],
        "review_score_avg": [4.0, 5.0, np.nan],
        "customer_id": ["c1", "c1", "c2"],
        "order_purchase_timestamp": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-02-01"]),
    })
    kpis = compute_kpis(df, "item_total_value", "order_purchase_timestamp")
    assert kpis["total_revenue"] == pytest.approx(600.0)
    assert kpis["total_orders"] == 2
    assert kpis["total_order_items"] == 3
    assert kpis["avg_delivery_days"] == pytest.approx(7.5)
    assert kpis["unique_customers"] == 2
    assert kpis["date_range_start"] == "2024-01-01"
    assert kpis["date_range_end"] == "2024-02-01"


def test_compute_kpis_handles_missing_columns_gracefully():
    df = pd.DataFrame({"item_total_value": [10.0, 20.0]})
    kpis = compute_kpis(df, "item_total_value", "nonexistent_date_col")
    assert kpis["total_revenue"] == pytest.approx(30.0)
    assert "date_range_start" not in kpis  # column absent -> silently skipped, not an error


def test_to_json_records_handles_nan_and_timestamps_and_numpy_types():
    df = pd.DataFrame({
        "a": [1, np.nan],
        "b": pd.to_datetime(["2024-01-01", None]),
        "c": [np.int64(5), np.int64(6)],
    })
    records = to_json_records(df)
    json.dumps(records)  # must not raise
    assert records[1]["a"] is None
    assert records[1]["b"] is None
    assert isinstance(records[0]["c"], int)


def test_to_json_records_handles_list_columns_after_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({"top_features": [["price", "freight"], ["delivery_days"]]})
    path = tmp_path / "t.parquet"
    df.to_parquet(path)
    reloaded = pd.read_parquet(path)
    assert isinstance(reloaded["top_features"].iloc[0], np.ndarray)  # confirms the real round-trip behavior
    records = to_json_records(reloaded)
    json.dumps(records)  # must not raise — this is the exact bug that was fixed
    assert records[0]["top_features"] == ["price", "freight"]


def test_to_json_records_empty_dataframe_returns_empty_list():
    assert to_json_records(pd.DataFrame()) == []


def test_resolve_chart_paths_joins_output_dir():
    paths = resolve_chart_paths("data/processed/eda", ["charts/hist_price.png", "charts/corr.png"])
    assert paths == ["data/processed/eda/charts/hist_price.png", "data/processed/eda/charts/corr.png"]


def test_resolve_chart_paths_handles_none():
    assert resolve_chart_paths("some/dir", None) == []


# ----------------------------------------------------------------------
# Dashboard Agent end-to-end test
# ----------------------------------------------------------------------

FIXTURE_ROOT = Path("tests/_fixture_dashboard")
INSIGHT_DIR = FIXTURE_ROOT / "insight_upstream"
FORECAST_DIR = FIXTURE_ROOT / "forecasting_upstream"
OUTPUT_ROOT = FIXTURE_ROOT / "processed"
CONFIG_PATH = FIXTURE_ROOT / "config.yaml"


@pytest.fixture()
def insight_manifest_path():
    INSIGHT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(20)],
        "item_total_value": np.linspace(50, 150, 20),
        "delivery_days": np.linspace(1, 10, 20),
        "review_score_avg": np.linspace(1, 5, 20),
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

    (INSIGHT_DIR / "executive_summary.md").write_text("## Business Performance Summary\n\nTest summary.")
    (INSIGHT_DIR / "charts").mkdir(exist_ok=True)
    _write_tiny_png(INSIGHT_DIR / "charts" / "segment_change_region.png")

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
        extra_metadata={"chart_paths": ["charts/segment_change_region.png"]},
    )
    path = manifest.save()

    config = {
        "dashboard": {
            "kpi_date_column": "order_purchase_timestamp",
            "kpi_value_column": "item_total_value",
            "app_root_copy_path": str(FIXTURE_ROOT / "test_dashboard_app.py"),
            "generate_chart": True,
        }
    }
    CONFIG_PATH.write_text(yaml.dump(config))

    yield path

    shutil.rmtree(FIXTURE_ROOT, ignore_errors=True)


@pytest.fixture()
def forecasting_manifest_path():
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame({"period": pd.date_range("2024-01-01", periods=5, freq="W").astype(str), "value": [100.0, 110.0, 105.0, 120.0, 130.0]})
    future = pd.DataFrame({"period": pd.date_range("2024-02-05", periods=3, freq="W").astype(str), "yhat": [135.0, 140.0, 138.0], "yhat_lower": [120.0, 122.0, 118.0], "yhat_upper": [150.0, 158.0, 158.0]})
    history.to_parquet(FORECAST_DIR / "forecast_history.parquet", index=False)
    future.to_parquet(FORECAST_DIR / "forecast_future.parquet", index=False)

    manifest = DatasetManifest(
        agent_name="forecasting",
        status=AgentStatus.SUCCESS,
        output_dir=str(FORECAST_DIR),
        primary_table="unified_cleaned",
        tables={
            "forecast_history": profile_dataframe("forecast_history", history, "forecast_history.parquet"),
            "forecast_future": profile_dataframe("forecast_future", future, "forecast_future.parquet"),
        },
        extra_metadata={"best_model_name": "prophet", "best_model_metrics": {"rmse": 5.0}, "chart_paths": []},
    )
    return manifest.save()


def test_dashboard_requires_upstream_manifest():
    agent = DashboardAgent(config_path="config/config.yaml", output_root=str(OUTPUT_ROOT))
    with pytest.raises(ValueError):
        agent.run(upstream_manifest_path=None)


def test_dashboard_writes_bundle_and_app(insight_manifest_path):
    agent = DashboardAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=insight_manifest_path)
    bundle_path = Path(manifest.extra_metadata["bundle_path"])
    app_path = Path(manifest.extra_metadata["app_path"])
    assert bundle_path.exists()
    assert app_path.exists()

    bundle = json.loads(bundle_path.read_text())
    assert bundle["kpis"]["total_orders"] == 20
    assert len(bundle["insights"]) == 1
    assert "Business Performance Summary" in bundle["executive_summary"]


def test_dashboard_includes_secondary_manifest_data(insight_manifest_path, forecasting_manifest_path):
    agent = DashboardAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(
        upstream_manifest_path=insight_manifest_path,
        manifest_paths={"forecasting": forecasting_manifest_path},
    )
    bundle = json.loads(Path(manifest.extra_metadata["bundle_path"]).read_text())
    assert len(bundle["forecast_future"]) == 3
    assert bundle["forecasting_summary"]["best_model_name"] == "prophet"


def test_dashboard_skips_missing_secondary_manifest_gracefully(insight_manifest_path):
    agent = DashboardAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(
        upstream_manifest_path=insight_manifest_path,
        manifest_paths={"forecasting": "does/not/exist/manifest.json"},
    )
    assert manifest.status in (AgentStatus.SUCCESS, AgentStatus.PARTIAL)  # doesn't crash
    bundle = json.loads(Path(manifest.extra_metadata["bundle_path"]).read_text())
    assert "forecast_future" not in bundle


def test_generated_app_runs_without_exceptions_via_apptest(insight_manifest_path, forecasting_manifest_path):
    """The real end-to-end check: actually EXECUTE the generated Streamlit
    app's script (via Streamlit's own testing harness) and confirm it
    produces no runtime exceptions and renders the expected widgets — not
    just that the .py file was written to disk."""
    from streamlit.testing.v1 import AppTest

    agent = DashboardAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(
        upstream_manifest_path=insight_manifest_path,
        manifest_paths={"forecasting": forecasting_manifest_path},
    )
    app_path = str(Path(manifest.extra_metadata["app_path"]).resolve())

    at = AppTest.from_file(app_path)
    at.run(timeout=60)

    assert not at.exception, f"Generated app raised: {at.exception}"
    metric_labels = [m.label for m in at.metric]
    assert "Total revenue" in metric_labels
    assert "Total orders" in metric_labels


def test_dashboard_passes_through_primary_table(insight_manifest_path):
    agent = DashboardAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=insight_manifest_path)
    df = pd.read_parquet(manifest.resolve_table_path("unified_cleaned"))
    assert len(df) == 20
