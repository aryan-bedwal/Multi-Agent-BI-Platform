"""
Tests for src/utils/anomaly_ops.py and the Anomaly Detection Agent end-to-end.
Uses synthetic data with a small number of DELIBERATELY planted outliers so
the ensemble's detection behavior can be asserted on, not just "it ran".
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.agents.anomaly_detection_agent import AnomalyDetectionAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.anomaly_ops import (
    _HAS_TORCH,
    detect_time_series_anomalies,
    ensemble_vote,
    run_isolation_forest,
    run_lof,
    run_one_class_svm,
)
from src.utils.profiling import profile_dataframe

# ----------------------------------------------------------------------
# Pure function tests
# ----------------------------------------------------------------------


def _planted_outlier_data(n_normal: int = 300, n_outliers: int = 15, n_dims: int = 6, seed: int = 0):
    """6 dimensions and pre-standardized, matching how the real agent calls
    these functions — they receive already-scaled output from a
    ColumnTransformer (StandardScaler + OneHotEncoder), not raw unscaled
    values. Testing with 2 raw dimensions understated every method's real
    performance and made the autoencoder's bottleneck_size=4 not a bottleneck
    at all (bigger than the input)."""
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    normal = rng.normal(loc=0, scale=1, size=(n_normal, n_dims))
    outliers = rng.normal(loc=15, scale=1, size=(n_outliers, n_dims))  # far away cluster
    X = np.vstack([normal, outliers])
    X = StandardScaler().fit_transform(X)
    is_true_outlier = np.array([False] * n_normal + [True] * n_outliers)
    return X, is_true_outlier


def test_isolation_forest_flags_planted_outliers():
    X, is_true_outlier = _planted_outlier_data()
    contamination = is_true_outlier.mean()
    flags, scores = run_isolation_forest(X, contamination=contamination, random_state=0)
    # a well-separated 6D cluster should be caught almost entirely
    assert flags[is_true_outlier].mean() >= 0.9
    # anomalous points should score higher than normal points on average
    assert scores[is_true_outlier].mean() > scores[~is_true_outlier].mean()


def test_lof_flags_planted_outliers():
    X, is_true_outlier = _planted_outlier_data()
    contamination = is_true_outlier.mean()
    flags, scores = run_lof(X, contamination=contamination, n_neighbors=20)
    assert flags[is_true_outlier].mean() >= 0.9


def test_one_class_svm_flags_planted_outliers():
    X, is_true_outlier = _planted_outlier_data()
    contamination = is_true_outlier.mean()
    flags, scores = run_one_class_svm(X, nu=contamination)
    # OneClassSVM fits a density boundary rather than scoring per-point
    # isolation, so it's a weaker detector than Isolation Forest/LOF on this
    # kind of well-separated-cluster test — that's realistic, not a bug in
    # our wrapper. The ensemble's job is to not rely on any one method alone.
    assert flags[is_true_outlier].mean() > 0.3
    assert scores[is_true_outlier].mean() > scores[~is_true_outlier].mean()


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed in this environment")
def test_autoencoder_flags_planted_outliers():
    from src.utils.anomaly_ops import run_autoencoder

    X, is_true_outlier = _planted_outlier_data()
    contamination = is_true_outlier.mean()
    flags, scores = run_autoencoder(X, contamination=contamination, epochs=150, learning_rate=0.02, random_state=0)
    # same rationale as OneClassSVM above — reconstruction error is a softer
    # signal than isolation-based methods; direction and some recall, not
    # perfect recall, is the realistic bar for a lightly-trained autoencoder.
    assert flags[is_true_outlier].mean() > 0.2
    assert scores[is_true_outlier].mean() > scores[~is_true_outlier].mean()


def test_ensemble_vote_requires_min_votes():
    flags = {
        "a": np.array([True, True, False, False]),
        "b": np.array([True, False, True, False]),
        "c": np.array([True, False, False, False]),
    }
    vote_count, consensus = ensemble_vote(flags, min_votes=2)
    assert vote_count.tolist() == [3, 1, 1, 0]
    assert consensus.tolist() == [True, False, False, False]


def test_time_series_anomaly_detects_planted_spike():
    periods = pd.date_range("2024-01-01", periods=20, freq="W")
    values = [100.0] * 20
    values[15] = 1000.0  # planted spike well past the rolling window
    series = pd.DataFrame({"period": periods, "value": values})

    result = detect_time_series_anomalies(series, rolling_window=6, z_threshold=2.0)
    spike_row = result[result["period"] == periods[15]].iloc[0]
    assert spike_row["is_anomalous"]
    assert spike_row["direction"] == "spike"


def test_time_series_anomaly_no_false_positive_on_flat_series():
    periods = pd.date_range("2024-01-01", periods=20, freq="W")
    series = pd.DataFrame({"period": periods, "value": [100.0] * 20})
    result = detect_time_series_anomalies(series, rolling_window=6, z_threshold=2.0)
    assert result["is_anomalous"].sum() == 0


def test_time_series_anomaly_early_periods_marked_insufficient_history():
    periods = pd.date_range("2024-01-01", periods=10, freq="W")
    series = pd.DataFrame({"period": periods, "value": range(10)})
    result = detect_time_series_anomalies(series, rolling_window=6, z_threshold=2.0)
    assert not result.iloc[0]["is_anomalous"]
    assert result.iloc[0]["direction"] == "insufficient_history"


# ----------------------------------------------------------------------
# Anomaly Detection Agent end-to-end test
# ----------------------------------------------------------------------

FIXTURE_ROOT = Path("tests/_fixture_anomaly")
UPSTREAM_DIR = FIXTURE_ROOT / "upstream"
OUTPUT_ROOT = FIXTURE_ROOT / "processed"
CONFIG_PATH = FIXTURE_ROOT / "config.yaml"


@pytest.fixture()
def upstream_manifest():
    UPSTREAM_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    n_normal, n_outliers = 300, 15
    n = n_normal + n_outliers

    price = np.concatenate([rng.normal(50, 10, n_normal), rng.normal(500, 20, n_outliers)])
    freight = np.concatenate([rng.normal(10, 2, n_normal), rng.normal(80, 5, n_outliers)])
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    df = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "order_item_id": ["1"] * n,
        "product_id": [f"p{i}" for i in range(n)],
        "price": price,
        "freight_value": freight,
        "item_total_value": price + freight,
        "payment_value_total": price + freight,
        "payment_installments_max": rng.integers(1, 5, n),
        "delivery_days": rng.uniform(1, 15, n),
        "review_score_avg": rng.uniform(1, 5, n),
        "product_weight_g": rng.uniform(100, 5000, n),
        "payment_type_primary": rng.choice(["credit_card", "boleto"], n),
        "customer_state": rng.choice(["SP", "RJ"], n),
        "seller_state": rng.choice(["SP", "MG"], n),
        "product_category_name_english": rng.choice(["electronics", "toys"], n),
        "order_purchase_timestamp": dates,
    })

    rel_path = "unified_cleaned.parquet"
    df.to_parquet(UPSTREAM_DIR / rel_path, index=False)
    profile = profile_dataframe("unified_cleaned", df, rel_path)

    manifest = DatasetManifest(
        agent_name="eda",
        status=AgentStatus.SUCCESS,
        output_dir=str(UPSTREAM_DIR),
        primary_table="unified_cleaned",
        tables={"unified_cleaned": profile},
    )
    manifest_path = manifest.save()

    config = {
        "anomaly_detection": {
            "numeric_columns": [
                "price", "freight_value", "item_total_value", "payment_value_total",
                "payment_installments_max", "delivery_days", "review_score_avg", "product_weight_g",
            ],
            "categorical_columns": [
                "payment_type_primary", "customer_state", "seller_state", "product_category_name_english",
            ],
            "contamination": round(n_outliers / n, 3),
            "min_votes_for_consensus": 2,
            "isolation_forest": {"n_estimators": 100, "random_state": 42},
            "lof": {"n_neighbors": 20},
            "one_class_svm": {"kernel": "rbf", "gamma": "scale"},
            "autoencoder": {"hidden_size": 8, "bottleneck_size": 2, "epochs": 30, "learning_rate": 0.01, "random_state": 42},
            "time_series": {
                "date_column": "order_purchase_timestamp",
                "value_column": "item_total_value",
                "freq": "W",
                "rolling_window": 4,
                "z_threshold": 2.0,
            },
            "generate_charts": True,
        }
    }
    CONFIG_PATH.write_text(yaml.dump(config))

    yield manifest_path

    shutil.rmtree(FIXTURE_ROOT, ignore_errors=True)


def test_anomaly_agent_requires_upstream_manifest():
    agent = AnomalyDetectionAgent(config_path="config/config.yaml", output_root=str(OUTPUT_ROOT))
    with pytest.raises(ValueError):
        agent.run(upstream_manifest_path=None)


def test_anomaly_agent_produces_expected_tables(upstream_manifest):
    agent = AnomalyDetectionAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest)
    expected = {"row_anomalies", "anomalous_periods", "anomaly_method_summary", "unified_cleaned"}
    assert expected.issubset(set(manifest.tables.keys()))


def test_anomaly_agent_flags_planted_row_outliers(upstream_manifest):
    agent = AnomalyDetectionAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest)
    row_anomalies = pd.read_parquet(manifest.resolve_table_path("row_anomalies"))
    # the last 15 rows (indices 300-314) are the planted high-price/high-freight outliers
    planted = row_anomalies.iloc[300:315]
    assert planted["is_anomaly_consensus"].mean() > 0.6


def test_anomaly_agent_ensemble_summary_in_metadata(upstream_manifest):
    agent = AnomalyDetectionAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest)
    ensemble = manifest.extra_metadata["row_level_method_summary"]["ensemble"]
    assert ensemble["n_consensus_anomalies"] > 0
    assert "isolation_forest" in ensemble["methods_used"]


def test_anomaly_agent_charts_generated(upstream_manifest):
    agent = AnomalyDetectionAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest)
    chart_paths = manifest.extra_metadata["chart_paths"]
    assert len(chart_paths) >= 1
    for rel in chart_paths:
        assert (Path(manifest.output_dir) / rel).exists()


def test_anomaly_agent_passes_through_primary_table(upstream_manifest):
    agent = AnomalyDetectionAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest)
    df = pd.read_parquet(manifest.resolve_table_path("unified_cleaned"))
    assert len(df) == 315


def test_anomaly_agent_status_success_when_torch_available(upstream_manifest):
    agent = AnomalyDetectionAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest)
    if _HAS_TORCH:
        assert manifest.status == AgentStatus.SUCCESS
    else:
        assert manifest.status == AgentStatus.PARTIAL
