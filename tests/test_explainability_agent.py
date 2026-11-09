"""
Tests for src/utils/explainability_ops.py and the Explainability Agent
end-to-end. Uses a synthetic regression problem with a KNOWN dominant feature
so SHAP-ranking assertions are meaningful, and a manually-planted anomalous
row so the z-score reasoning can be checked against ground truth.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.agents.explainability_agent import ExplainabilityAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.explainability_ops import (
    build_shap_explainer,
    compute_anomaly_reasons,
    compute_shap_values,
    explain_row_shap,
    select_example_rows,
    shap_global_summary,
)
from src.utils.ml_ops import build_preprocessor
from src.utils.profiling import profile_dataframe

# ----------------------------------------------------------------------
# Pure function tests
# ----------------------------------------------------------------------


def _fit_known_signal_pipeline(seed: int = 0):
    """y depends heavily on x_dominant and barely on x_noise — a trained
    tree model's SHAP values should reflect that asymmetry clearly."""
    rng = np.random.default_rng(seed)
    n = 300
    x_dominant = rng.uniform(0, 10, n)
    x_noise = rng.uniform(0, 10, n)
    y = 5 * x_dominant + rng.normal(0, 0.5, n)

    df = pd.DataFrame({"x_dominant": x_dominant, "x_noise": x_noise})
    preprocessor = build_preprocessor(["x_dominant", "x_noise"], [])
    model = RandomForestRegressor(n_estimators=100, random_state=seed)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(df, y)
    return pipeline, df, y


def test_shap_summary_ranks_dominant_feature_first():
    pipeline, df, y = _fit_known_signal_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_transformed = np.asarray(preprocessor.transform(df))
    feature_names = list(preprocessor.get_feature_names_out())

    explainer = build_shap_explainer(model, X_transformed)
    shap_values = compute_shap_values(explainer, X_transformed)
    summary = shap_global_summary(shap_values, feature_names)

    assert summary.iloc[0]["feature"] == "numeric__x_dominant"


def test_explain_row_shap_returns_sorted_by_magnitude():
    row_shap = np.array([0.1, -0.9, 0.3])
    row_values = np.array([1.0, 2.0, 3.0])
    reasons = explain_row_shap(row_shap, ["a", "b", "c"], row_values, top_n=2)
    assert reasons[0]["feature"] == "b"  # largest |shap value|
    assert reasons[0]["direction"] == "decreases"
    assert len(reasons) == 2


def test_select_example_rows_includes_extremes():
    y_pred = np.array([10.0, 1.0, 5.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0])
    positions = select_example_rows(y_pred, n_total=6, n_random=2, random_state=0)
    assert np.argmax(y_pred) in positions
    assert np.argmin(y_pred) in positions


def test_compute_anomaly_reasons_identifies_planted_deviation():
    df = pd.DataFrame({
        "price": [50.0] * 20 + [5000.0],  # one wildly high price
        "freight": [10.0] * 21,  # constant — shouldn't show up as a reason
    })
    mask = pd.Series([False] * 20 + [True])
    reasons = compute_anomaly_reasons(df, ["price", "freight"], mask, top_n=2)
    assert len(reasons) == 1  # freight has zero variance -> NaN z-score -> excluded
    assert reasons.iloc[0]["feature"] == "price"
    assert reasons.iloc[0]["z_score"] > 0


def test_compute_anomaly_reasons_empty_mask_returns_empty():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    mask = pd.Series([False, False, False])
    reasons = compute_anomaly_reasons(df, ["x"], mask, top_n=3)
    assert reasons.empty


# ----------------------------------------------------------------------
# Explainability Agent end-to-end test
# ----------------------------------------------------------------------

FIXTURE_ROOT = Path("tests/_fixture_explainability")
ML_DIR = FIXTURE_ROOT / "ml_upstream"
ANOMALY_DIR = FIXTURE_ROOT / "anomaly_upstream"
OUTPUT_ROOT = FIXTURE_ROOT / "processed"
CONFIG_PATH = FIXTURE_ROOT / "config.yaml"


@pytest.fixture()
def ml_manifest_path():
    ML_DIR.mkdir(parents=True, exist_ok=True)
    pipeline, X, y = _fit_known_signal_pipeline(seed=1)

    df = X.copy()
    df["order_id"] = [f"o{i}" for i in range(len(df))]
    df["target_value"] = y
    rel_path = "unified_cleaned.parquet"
    df.to_parquet(ML_DIR / rel_path, index=False)
    profile = profile_dataframe("unified_cleaned", df, rel_path)

    joblib.dump(pipeline, ML_DIR / "best_model.joblib")
    model_card = {
        "target_column": "target_value",
        "problem_type": "regression",
        "best_model_name": "random_forest",
        "best_model_metrics": {"rmse": 1.0, "mae": 0.5, "r2": 0.95},
        "numeric_features": ["x_dominant", "x_noise"],
        "categorical_features": [],
        "n_train_rows": 240,
        "n_test_rows": 60,
    }
    (ML_DIR / "model_card.json").write_text(json.dumps(model_card))

    manifest = DatasetManifest(
        agent_name="ml",
        status=AgentStatus.SUCCESS,
        output_dir=str(ML_DIR),
        primary_table="unified_cleaned",
        tables={"unified_cleaned": profile},
    )
    path = manifest.save()

    config = {
        "explainability": {
            "n_example_rows": 6,
            "shap_sample_size": 100,
            "n_lime_rows": 3,
            "top_n_reasons": 3,
            "random_state": 42,
            "anomaly_top_n_reasons": 3,
            "generate_charts": True,
        }
    }
    CONFIG_PATH.write_text(yaml.dump(config))

    yield path

    shutil.rmtree(FIXTURE_ROOT, ignore_errors=True)


@pytest.fixture()
def anomaly_manifest_path():
    ANOMALY_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)
    n = 100
    row_anomalies = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "price": rng.normal(50, 5, n),
        "freight_value": rng.normal(10, 1, n),
        "isolation_forest_flag": [False] * (n - 3) + [True] * 3,
        "isolation_forest_score": rng.uniform(0, 1, n),
        "vote_count": [0] * (n - 3) + [3] * 3,
        "is_anomaly_consensus": [False] * (n - 3) + [True] * 3,
    })
    # make the planted anomalies genuinely deviate
    row_anomalies.loc[row_anomalies.index[-3:], "price"] = [500.0, 520.0, 480.0]

    rel_path = "row_anomalies.parquet"
    row_anomalies.to_parquet(ANOMALY_DIR / rel_path, index=False)
    profile = profile_dataframe("row_anomalies", row_anomalies, rel_path)

    manifest = DatasetManifest(
        agent_name="anomaly_detection",
        status=AgentStatus.SUCCESS,
        output_dir=str(ANOMALY_DIR),
        primary_table="row_anomalies",
        tables={"row_anomalies": profile},
    )
    return manifest.save()


def test_explainability_requires_upstream_manifest():
    agent = ExplainabilityAgent(config_path="config/config.yaml", output_root=str(OUTPUT_ROOT))
    with pytest.raises(ValueError):
        agent.run(upstream_manifest_path=None)


def test_explainability_produces_shap_tables(ml_manifest_path):
    agent = ExplainabilityAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=ml_manifest_path)
    assert "shap_global_summary" in manifest.tables
    assert "shap_row_explanations" in manifest.tables
    assert manifest.extra_metadata["explained_target"] == "target_value"


def test_explainability_shap_identifies_dominant_feature(ml_manifest_path):
    agent = ExplainabilityAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=ml_manifest_path)
    top_features = manifest.extra_metadata["top_shap_features"]
    assert top_features[0]["feature"] == "numeric__x_dominant"


def test_explainability_produces_lime_tables(ml_manifest_path):
    agent = ExplainabilityAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=ml_manifest_path)
    assert "lime_row_explanations" in manifest.tables
    assert manifest.extra_metadata["n_lime_example_rows"] > 0


def test_explainability_without_anomaly_manifest_skips_anomaly_tables(ml_manifest_path):
    agent = ExplainabilityAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=ml_manifest_path)
    assert "anomaly_reasons" not in manifest.tables
    assert manifest.extra_metadata["n_anomalies_explained"] == 0
    assert manifest.status == AgentStatus.SUCCESS


def test_explainability_with_anomaly_manifest_produces_reasons(ml_manifest_path, anomaly_manifest_path):
    agent = ExplainabilityAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=ml_manifest_path, anomaly_manifest_path=anomaly_manifest_path)
    assert "anomaly_reasons" in manifest.tables
    assert manifest.extra_metadata["n_anomalies_explained"] == 3

    reasons = pd.read_parquet(manifest.resolve_table_path("anomaly_reasons"))
    # every planted anomaly's #1 reason should be 'price' — the feature we deviated
    top_reasons = reasons[reasons["rank"] == 1]
    assert (top_reasons["feature"] == "price").all()


def test_explainability_chart_generated(ml_manifest_path):
    agent = ExplainabilityAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=ml_manifest_path)
    chart_paths = manifest.extra_metadata["chart_paths"]
    assert len(chart_paths) == 1
    assert (Path(manifest.output_dir) / chart_paths[0]).exists()


def test_explainability_passes_through_primary_table(ml_manifest_path):
    agent = ExplainabilityAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=ml_manifest_path)
    df = pd.read_parquet(manifest.resolve_table_path("unified_cleaned"))
    assert len(df) == 300
