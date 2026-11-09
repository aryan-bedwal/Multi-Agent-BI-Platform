"""
Tests for src/utils/eda_ops.py (pure functions) and the EDA Agent end-to-end.
Chart generation is exercised but only checked for file existence — pixel
content isn't worth asserting on.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.agents.eda_agent import EDAAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.eda_ops import (
    compute_categorical_distributions,
    compute_correlation_matrix,
    compute_missing_value_summary,
    compute_numeric_summary,
    compute_time_series_trend,
    period_over_period_change,
    top_correlated_pairs,
)
from src.utils.profiling import profile_dataframe

# ----------------------------------------------------------------------
# Pure function tests (no agent, no disk I/O)
# ----------------------------------------------------------------------


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "price": [10.0, 20.0, 30.0, 40.0, np.nan],
        "freight": [1.0, 2.0, 3.0, 4.0, 5.0],
        "category": ["a", "a", "b", "c", "a"],
        "purchase_date": pd.to_datetime(
            ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29"]
        ),
    })


def test_numeric_summary_basic_stats():
    df = _sample_df()
    summary = compute_numeric_summary(df, ["price", "freight"])
    price_row = summary[summary["column"] == "price"].iloc[0]
    assert price_row["count"] == 4
    assert price_row["n_missing"] == 1
    assert price_row["min"] == 10.0
    assert price_row["max"] == 40.0


def test_numeric_summary_ignores_non_numeric_columns():
    df = _sample_df()
    summary = compute_numeric_summary(df, ["price", "category"])
    assert "category" not in summary["column"].values


def test_correlation_matrix_perfectly_correlated_columns():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8]})
    corr = compute_correlation_matrix(df, ["a", "b"])
    row = corr[corr["column"] == "a"].iloc[0]
    assert row["b"] == pytest.approx(1.0)


def test_top_correlated_pairs_returns_sorted_pairs():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [4, 1, 3, 2]})
    corr = compute_correlation_matrix(df, ["a", "b", "c"])
    pairs = top_correlated_pairs(corr, n=1)
    assert len(pairs) == 1
    assert {pairs[0]["column"], pairs[0]["column_b"]} == {"a", "b"}


def test_categorical_distributions_counts_and_pct():
    df = _sample_df()
    dist = compute_categorical_distributions(df, ["category"], top_n=10)
    a_row = dist[(dist["column"] == "category") & (dist["value"] == "a")].iloc[0]
    assert a_row["count"] == 3
    assert a_row["pct"] == pytest.approx(60.0)


def test_categorical_distributions_buckets_long_tail_as_other():
    df = pd.DataFrame({"cat": ["a"] * 5 + ["b", "c", "d", "e", "f"]})
    dist = compute_categorical_distributions(df, ["cat"], top_n=2)
    assert "(other)" in dist["value"].values
    # top_n=2 keeps the 2 highest-count values (a=5, plus one of the 5 ties);
    # everything else collapses into a single "(other)" row that accounts for
    # the remaining rows — the exact split among ties isn't asserted, just
    # that the totals reconcile.
    assert dist[dist["column"] == "cat"]["count"].sum() == len(df)


def test_missing_value_summary_sorted_descending():
    df = pd.DataFrame({"a": [1, None, None], "b": [1, 2, 3]})
    summary = compute_missing_value_summary(df)
    assert summary.iloc[0]["column"] == "a"
    assert summary.iloc[0]["pct_missing"] == pytest.approx(66.67)


def test_time_series_trend_weekly_buckets():
    df = _sample_df()
    trend = compute_time_series_trend(df, "purchase_date", "price", freq="W")
    assert len(trend) >= 1
    assert "value_sum" in trend.columns


def test_time_series_trend_missing_columns_returns_empty():
    df = _sample_df()
    trend = compute_time_series_trend(df, "nonexistent", "price", freq="W")
    assert trend.empty


def test_period_over_period_change():
    trend = pd.DataFrame({"period": ["2024-01-01", "2024-01-08"], "value_sum": [100.0, 120.0]})
    change = period_over_period_change(trend)
    assert change["pct_change"] == pytest.approx(20.0)


def test_period_over_period_change_needs_two_periods():
    trend = pd.DataFrame({"period": ["2024-01-01"], "value_sum": [100.0]})
    assert period_over_period_change(trend) is None


# ----------------------------------------------------------------------
# EDA Agent end-to-end test (small upstream manifest, real disk I/O)
# ----------------------------------------------------------------------

FIXTURE_ROOT = Path("tests/_fixture_eda")
UPSTREAM_DIR = FIXTURE_ROOT / "upstream"
OUTPUT_ROOT = FIXTURE_ROOT / "processed"
CONFIG_PATH = FIXTURE_ROOT / "config.yaml"


@pytest.fixture()
def upstream_manifest():
    UPSTREAM_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "price": rng.uniform(10, 200, n),
        "freight_value": rng.uniform(1, 30, n),
        "order_status": rng.choice(["delivered", "shipped", "canceled"], n),
        "customer_state": rng.choice(["SP", "RJ", "MG"], n),
        "order_purchase_timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
    })
    rel_path = "unified_cleaned.parquet"
    df.to_parquet(UPSTREAM_DIR / rel_path, index=False)
    profile = profile_dataframe("unified_cleaned", df, rel_path)

    manifest = DatasetManifest(
        agent_name="data_cleaning",
        status=AgentStatus.SUCCESS,
        output_dir=str(UPSTREAM_DIR),
        primary_table="unified_cleaned",
        tables={"unified_cleaned": profile},
    )
    manifest_path = manifest.save()

    config = {
        "eda": {
            "numeric_columns": ["price", "freight_value"],
            "categorical_columns": ["order_status", "customer_state"],
            "top_n_categories": 5,
            "time_series": {
                "date_column": "order_purchase_timestamp",
                "value_column": "price",
                "freq": "W",
            },
            "correlation_method": "pearson",
            "generate_charts": True,
            "max_histogram_columns": 6,
        }
    }
    CONFIG_PATH.write_text(yaml.dump(config))

    yield manifest_path

    shutil.rmtree(FIXTURE_ROOT, ignore_errors=True)


def test_eda_requires_upstream_manifest():
    agent = EDAAgent(config_path="config/config.yaml", output_root=str(OUTPUT_ROOT))
    with pytest.raises(ValueError):
        agent.run(upstream_manifest_path=None)


def test_eda_produces_all_expected_summary_tables(upstream_manifest):
    agent = EDAAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest)
    expected = {
        "numeric_summary",
        "correlation_matrix",
        "categorical_distributions",
        "missing_value_summary",
        "time_series_trend",
        "unified_cleaned",
    }
    assert expected.issubset(set(manifest.tables.keys()))
    assert manifest.status == AgentStatus.SUCCESS


def test_eda_passes_through_primary_table_unchanged(upstream_manifest):
    agent = EDAAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest)
    df = pd.read_parquet(manifest.resolve_table_path("unified_cleaned"))
    assert len(df) == 60
    assert manifest.primary_table == "unified_cleaned"


def test_eda_generates_chart_files_on_disk(upstream_manifest):
    agent = EDAAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest)
    chart_paths = manifest.extra_metadata["chart_paths"]
    assert len(chart_paths) > 0
    for rel in chart_paths:
        assert (Path(manifest.output_dir) / rel).exists()


def test_eda_skips_charts_when_disabled(upstream_manifest):
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["eda"]["generate_charts"] = False
    CONFIG_PATH.write_text(yaml.dump(config))

    agent = EDAAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest)
    assert manifest.extra_metadata["chart_paths"] == []
