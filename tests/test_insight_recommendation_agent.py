"""
Tests for src/utils/insight_ops.py and the Insight & Recommendation Agent
end-to-end. Uses synthetic data with a KNOWN, planted revenue swing (one
segment clearly up, one clearly down) so insight/recommendation-content
assertions are meaningful, not just "something was produced".
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.agents.insight_recommendation_agent import InsightRecommendationAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.insight_ops import (
    build_executive_summary,
    compute_segment_change,
    generate_forecast_insight,
    generate_ml_driver_insight,
    generate_revenue_trend_insight,
    generate_row_anomaly_insight,
    generate_segment_insights,
    generate_time_series_anomaly_insights,
    map_insight_to_recommendation,
)
from src.utils.profiling import profile_dataframe

# ----------------------------------------------------------------------
# Pure function tests
# ----------------------------------------------------------------------


def _segment_dataset():
    """8 weeks of data: 'A' segment steady, 'B' segment drops hard in the
    most recent 4 weeks, 'C' segment rises hard."""
    dates = pd.date_range("2024-01-01", periods=56, freq="D")  # 8 weeks
    rows = []
    for i, d in enumerate(dates):
        week = i // 7
        rows.append({"date": d, "segment": "A", "value": 100.0})
        rows.append({"date": d, "segment": "B", "value": 100.0 if week < 4 else 20.0})
        rows.append({"date": d, "segment": "C", "value": 100.0 if week < 4 else 300.0})
    return pd.DataFrame(rows)


def test_compute_segment_change_detects_planted_drop_and_rise():
    df = _segment_dataset()
    result = compute_segment_change(df, "date", "value", "segment", freq="W", recent_periods=4, prior_periods=4)
    result = result.set_index("segment")
    assert result.loc["B", "pct_change"] < -50
    assert result.loc["B", "direction"] == "decrease"
    assert result.loc["C", "pct_change"] > 100
    assert result.loc["C", "direction"] == "increase"
    assert abs(result.loc["A", "pct_change"]) < 5  # steady segment, near-zero change


def test_compute_segment_change_insufficient_history_returns_empty():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3, freq="D"),
        "value": [1.0, 2.0, 3.0],
        "segment": ["A", "A", "A"],
    })
    result = compute_segment_change(df, "date", "value", "segment", freq="W", recent_periods=4, prior_periods=4)
    assert result.empty


def test_generate_segment_insights_respects_threshold():
    change_df = pd.DataFrame({
        "segment": ["big_drop", "small_change"],
        "recent_value": [50.0, 98.0],
        "prior_value": [100.0, 100.0],
        "pct_change": [-50.0, -2.0],
        "direction": ["decrease", "decrease"],
    })
    insights = generate_segment_insights(change_df, "region", top_n=5, threshold=10.0)
    texts = [i["metric"] for i in insights]
    assert "region:big_drop" in texts
    assert "region:small_change" not in texts  # below threshold, correctly excluded


def test_generate_revenue_trend_insight_none_when_no_data():
    assert generate_revenue_trend_insight(None) is None


def test_generate_revenue_trend_insight_direction_matches_sign():
    pop_change = {"latest_period": "2024-01-08", "previous_period": "2024-01-01", "latest_value": 90.0, "previous_value": 100.0, "pct_change": -10.0}
    insight = generate_revenue_trend_insight(pop_change)
    assert insight["direction"] == "decrease"
    assert "decreased" in insight["text"]


def test_generate_forecast_insight_computes_pct_change():
    forecast_future = pd.DataFrame({"yhat": [120.0, 130.0]})
    insight = generate_forecast_insight(forecast_future, history_last_value=100.0, best_model_name="prophet", horizon=2)
    assert insight["pct_change"] == pytest.approx(25.0)
    assert insight["direction"] == "increase"


def test_generate_forecast_insight_none_on_zero_history():
    forecast_future = pd.DataFrame({"yhat": [10.0]})
    assert generate_forecast_insight(forecast_future, history_last_value=0.0, best_model_name="x", horizon=1) is None


def test_generate_time_series_anomaly_insights_only_flagged_rows():
    anomalous_periods = pd.DataFrame({
        "period": ["2024-01-01", "2024-01-08"],
        "value": [100.0, 500.0],
        "rolling_mean": [90.0, 95.0],
        "z_score": [0.5, 4.0],
        "is_anomalous": [False, True],
        "direction": ["normal", "spike"],
    })
    insights = generate_time_series_anomaly_insights(anomalous_periods, "revenue")
    assert len(insights) == 1
    assert insights[0]["direction"] == "increase"


def test_generate_row_anomaly_insight_none_when_zero_consensus():
    assert generate_row_anomaly_insight(0, 0.0, pd.DataFrame()) is None


def test_generate_row_anomaly_insight_includes_top_feature():
    reasons = pd.DataFrame({
        "row_index": [1, 2, 3],
        "rank": [1, 1, 1],
        "feature": ["price", "price", "freight_value"],
    })
    insight = generate_row_anomaly_insight(3, 1.5, reasons)
    assert insight["top_contributing_feature"] == "price"
    assert "price" in insight["text"]


def test_generate_ml_driver_insight_strips_preprocessor_prefixes():
    shap_summary = pd.DataFrame({
        "feature": ["numeric__price", "categorical__state_SP"],
        "mean_abs_shap": [0.5, 0.2],
    })
    insight = generate_ml_driver_insight(shap_summary, "delivery_days", top_n=2)
    assert "price" in insight["top_features"]
    assert "numeric__" not in insight["text"]


def test_map_insight_to_recommendation_segment_decrease():
    insight = {"type": "segment_decrease", "metric": "region:X", "pct_change": -20.0, "text": "...", "severity": "high"}
    rec = map_insight_to_recommendation(insight)
    assert "promotion" in rec["action_text"].lower() or "discount" in rec["action_text"].lower()
    assert rec["priority"] == "high"


def test_map_insight_to_recommendation_ml_drivers_returns_none():
    # informational insight type — no single action implied
    insight = {"type": "ml_drivers", "text": "...", "severity": "medium"}
    assert map_insight_to_recommendation(insight) is None


def test_build_executive_summary_orders_high_severity_first():
    insights = [
        {"type": "a", "text": "Small change", "severity": "medium", "pct_change": 5.0},
        {"type": "b", "text": "Big change", "severity": "high", "pct_change": 80.0},
    ]
    summary = build_executive_summary(insights, [], max_items=5)
    assert summary.index("Big change") < summary.index("Small change")


def test_build_executive_summary_handles_empty_input():
    summary = build_executive_summary([], [], max_items=5)
    assert "No significant insights" in summary


# ----------------------------------------------------------------------
# Insight & Recommendation Agent end-to-end test
# ----------------------------------------------------------------------

FIXTURE_ROOT = Path("tests/_fixture_insight")
EXPL_DIR = FIXTURE_ROOT / "explainability_upstream"
FORECAST_DIR = FIXTURE_ROOT / "forecasting_upstream"
ANOMALY_DIR = FIXTURE_ROOT / "anomaly_upstream"
OUTPUT_ROOT = FIXTURE_ROOT / "processed"
CONFIG_PATH = FIXTURE_ROOT / "config.yaml"


@pytest.fixture()
def explainability_manifest_path():
    EXPL_DIR.mkdir(parents=True, exist_ok=True)
    df = _segment_dataset().rename(columns={"date": "order_purchase_timestamp", "value": "item_total_value", "segment": "product_category_name_english"})
    df["customer_state"] = "SP"
    rel_path = "unified_cleaned.parquet"
    df.to_parquet(EXPL_DIR / rel_path, index=False)
    profile = profile_dataframe("unified_cleaned", df, rel_path)

    shap_summary = pd.DataFrame({"feature": ["numeric__price", "numeric__freight_value"], "mean_abs_shap": [0.6, 0.3]})
    shap_rel = "shap_global_summary.parquet"
    shap_summary.to_parquet(EXPL_DIR / shap_rel, index=False)
    shap_profile = profile_dataframe("shap_global_summary", shap_summary, shap_rel)

    manifest = DatasetManifest(
        agent_name="explainability",
        status=AgentStatus.SUCCESS,
        output_dir=str(EXPL_DIR),
        primary_table="unified_cleaned",
        tables={"unified_cleaned": profile, "shap_global_summary": shap_profile},
        extra_metadata={"explained_target": "delivery_days"},
    )
    path = manifest.save()

    config = {
        "insight_recommendation": {
            "date_column": "order_purchase_timestamp",
            "value_column": "item_total_value",
            "freq": "W",
            "segment_columns": ["product_category_name_english"],
            "recent_periods": 4,
            "prior_periods": 4,
            "significant_pct_change_threshold": 10.0,
            "top_n_segments": 3,
            "top_n_shap_drivers": 5,
            "generate_chart": True,
        }
    }
    CONFIG_PATH.write_text(yaml.dump(config))

    yield path

    shutil.rmtree(FIXTURE_ROOT, ignore_errors=True)


def test_insight_agent_requires_upstream_manifest():
    agent = InsightRecommendationAgent(config_path="config/config.yaml", output_root=str(OUTPUT_ROOT))
    with pytest.raises(ValueError):
        agent.run(upstream_manifest_path=None)


def test_insight_agent_produces_expected_tables(explainability_manifest_path):
    agent = InsightRecommendationAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=explainability_manifest_path)
    assert "insights" in manifest.tables
    assert "recommendations" in manifest.tables
    assert manifest.extra_metadata["n_insights"] > 0


def test_insight_agent_detects_planted_segment_swings(explainability_manifest_path):
    agent = InsightRecommendationAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=explainability_manifest_path)
    insights = pd.read_parquet(manifest.resolve_table_path("insights"))
    texts = " ".join(insights["text"].tolist())
    assert "B" in texts  # the planted-drop segment
    assert "C" in texts  # the planted-rise segment


def test_insight_agent_works_without_optional_manifests(explainability_manifest_path):
    agent = InsightRecommendationAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=explainability_manifest_path)
    # no forecasting_manifest_path or anomaly_manifest_path passed — should
    # still succeed with revenue trend + segment + ML driver insights
    assert manifest.status == AgentStatus.SUCCESS
    insight_types = manifest.extra_metadata["insight_types"]
    assert not any(t.startswith("forecast_") for t in insight_types)
    assert not any(t.startswith("time_series_") for t in insight_types)


def test_insight_agent_executive_summary_in_metadata(explainability_manifest_path):
    agent = InsightRecommendationAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=explainability_manifest_path)
    assert "Business Performance Summary" in manifest.extra_metadata["executive_summary"]


def test_insight_agent_chart_generated(explainability_manifest_path):
    agent = InsightRecommendationAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=explainability_manifest_path)
    chart_paths = manifest.extra_metadata["chart_paths"]
    assert len(chart_paths) >= 1
    for rel in chart_paths:
        assert (Path(manifest.output_dir) / rel).exists()


def test_insight_agent_passes_through_primary_table(explainability_manifest_path):
    agent = InsightRecommendationAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=explainability_manifest_path)
    df = pd.read_parquet(manifest.resolve_table_path("unified_cleaned"))
    assert len(df) == 168  # 56 days x 3 segments
