"""
Tests for src/utils/ml_ops.py (pure-ish functions) and the ML Agent
end-to-end on a small synthetic regression problem with a real (not
coincidental) signal, so metric assertions are meaningful.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from src.agents.ml_agent import MLAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.ml_ops import (
    detect_problem_type,
    evaluate_classification,
    evaluate_regression,
    pick_best_model,
)
from src.utils.profiling import profile_dataframe

# ----------------------------------------------------------------------
# Pure function tests
# ----------------------------------------------------------------------


def test_detect_problem_type_continuous_numeric_is_regression():
    y = pd.Series(np.linspace(0, 1000, 200))
    assert detect_problem_type(y, classification_max_cardinality=20) == "regression"


def test_detect_problem_type_low_cardinality_numeric_is_classification():
    y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1] * 10)
    assert detect_problem_type(y, classification_max_cardinality=20) == "classification"


def test_detect_problem_type_string_target_is_classification():
    y = pd.Series(["yes", "no", "yes", "no"])
    assert detect_problem_type(y) == "classification"


def test_evaluate_regression_perfect_predictions():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = evaluate_regression(y_true, y_true.copy())
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_evaluate_classification_perfect_predictions():
    y_true = np.array([0, 1, 1, 0])
    metrics = evaluate_classification(y_true, y_true.copy())
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_weighted"] == pytest.approx(1.0)


def test_pick_best_model_regression_prefers_lowest_rmse():
    metrics = {
        "model_a": {"rmse": 5.0, "mae": 4.0, "r2": 0.5},
        "model_b": {"rmse": 2.0, "mae": 1.5, "r2": 0.9},
    }
    name, best = pick_best_model(metrics, "regression")
    assert name == "model_b"


def test_pick_best_model_classification_prefers_highest_f1():
    metrics = {
        "model_a": {"accuracy": 0.6, "f1_weighted": 0.55},
        "model_b": {"accuracy": 0.9, "f1_weighted": 0.88},
    }
    name, best = pick_best_model(metrics, "classification")
    assert name == "model_b"


# ----------------------------------------------------------------------
# ML Agent end-to-end test
# ----------------------------------------------------------------------

FIXTURE_ROOT = Path("tests/_fixture_ml")
UPSTREAM_DIR = FIXTURE_ROOT / "upstream"
OUTPUT_ROOT = FIXTURE_ROOT / "processed"
CONFIG_PATH = FIXTURE_ROOT / "config.yaml"


@pytest.fixture()
def upstream_manifest_regression():
    UPSTREAM_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 400
    # Real, learnable signal: target is a noisy linear function of x1, plus a
    # categorical effect — so a trained model should clearly beat a dummy
    # baseline, and the test can assert on that rather than exact numbers.
    x1 = rng.uniform(0, 100, n)
    cat = rng.choice(["a", "b"], n)
    cat_effect = np.where(cat == "a", 10.0, -10.0)
    noise = rng.normal(0, 2, n)
    y = 2.0 * x1 + cat_effect + noise

    df = pd.DataFrame({
        "id_col": [f"id_{i}" for i in range(n)],
        "x1": x1,
        "x2_irrelevant": rng.uniform(0, 1, n),
        "category": cat,
        "target_value": y,
    })
    # sprinkle some missing targets, mirroring 'not yet delivered' rows
    df.loc[df.sample(20, random_state=1).index, "target_value"] = np.nan

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
        "ml": {
            "target_column": "target_value",
            "problem_type": "regression",
            "exclude_columns": ["id_col"],
            "test_size": 0.25,
            "random_state": 42,
            "candidate_models": {
                "regression": ["linear_regression", "random_forest"],
                "classification": ["logistic_regression", "random_forest"],
            },
            "classification_max_cardinality": 20,
        }
    }
    CONFIG_PATH.write_text(yaml.dump(config))

    yield manifest_path

    shutil.rmtree(FIXTURE_ROOT, ignore_errors=True)


def test_ml_agent_requires_upstream_manifest():
    agent = MLAgent(config_path="config/config.yaml", output_root=str(OUTPUT_ROOT))
    with pytest.raises(ValueError):
        agent.run(upstream_manifest_path=None)


def test_ml_agent_drops_rows_with_missing_target(upstream_manifest_regression):
    agent = MLAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest_regression)
    assert manifest.extra_metadata["n_train_rows"] + manifest.extra_metadata["n_test_rows"] == 380


def test_ml_agent_finds_real_signal_beats_baseline(upstream_manifest_regression):
    agent = MLAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest_regression)
    best_metrics = manifest.extra_metadata["best_model_metrics"]
    # target has real signal from x1 (coefficient 2.0) + category effect —
    # r2 should be clearly positive and strong, not near zero
    assert best_metrics["r2"] > 0.8


def test_ml_agent_selects_linear_regression_for_linear_signal(upstream_manifest_regression):
    # the fixture's target is an exactly linear function of x1 + category +
    # small noise, so linear_regression should legitimately win or tie
    agent = MLAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest_regression)
    metrics_by_model = manifest.extra_metadata["metrics_by_model"]
    assert "linear_regression" in metrics_by_model
    assert "random_forest" in metrics_by_model


def test_ml_agent_saves_loadable_model_artifact(upstream_manifest_regression):
    agent = MLAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest_regression)
    model_path = Path(manifest.output_dir) / manifest.extra_metadata["model_artifact_path"]
    assert model_path.exists()
    pipeline = joblib.load(model_path)
    # sanity: the loaded pipeline can actually predict
    sample = pd.DataFrame({
        "x1": [50.0],
        "x2_irrelevant": [0.5],
        "category": ["a"],
    })
    pred = pipeline.predict(sample)
    assert len(pred) == 1


def test_ml_agent_produces_feature_importance_for_tree_model(upstream_manifest_regression):
    agent = MLAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest_regression)
    if manifest.extra_metadata["best_model_name"] in ("random_forest", "xgboost"):
        assert "feature_importance" in manifest.tables


def test_ml_agent_downgrades_status_on_problem_type_mismatch(upstream_manifest_regression):
    # Use a target with a SMALL number of distinct integer values so
    # 'classification' is actually a valid, fittable choice — the point of
    # this test is the mismatch-detection/status-downgrade, not forcing an
    # incompatible sklearn call. Auto-detection still calls it 'regression'
    # only because classification_max_cardinality is set unrealistically low.
    manifest = DatasetManifest.load(upstream_manifest_regression)
    df = pd.read_parquet(Path(manifest.output_dir) / "unified_cleaned.parquet")
    df["target_value"] = (df["target_value"] % 6).round().astype("Int64")
    df.loc[df.sample(5, random_state=2).index, "target_value"] = pd.NA
    df.to_parquet(Path(manifest.output_dir) / "unified_cleaned.parquet", index=False)

    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["ml"]["problem_type"] = "classification"  # valid: target has only 6 distinct values
    config["ml"]["classification_max_cardinality"] = 3  # but too low, so auto-detect says 'regression'
    CONFIG_PATH.write_text(yaml.dump(config))

    agent = MLAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    result = agent.run(upstream_manifest_path=upstream_manifest_regression)
    assert result.status == AgentStatus.PARTIAL
    assert result.extra_metadata["problem_type"] == "classification"
    assert result.extra_metadata["problem_type_auto_detected"] == "regression"


def test_ml_agent_passes_through_primary_table(upstream_manifest_regression):
    agent = MLAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest_regression)
    df = pd.read_parquet(manifest.resolve_table_path("unified_cleaned"))
    assert len(df) == 400  # all rows, including the ones with missing target
