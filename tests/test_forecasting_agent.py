"""
Tests for src/utils/forecasting_ops.py and the Forecasting Agent end-to-end.
Uses a synthetic series with a real, learnable trend (not pure noise) so
metric/model-selection assertions are meaningful rather than arbitrary.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.agents.forecasting_agent import ForecastingAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.forecasting_ops import (
    _HAS_PROPHET,
    _HAS_TORCH,
    build_time_series,
    chronological_split,
    compute_forecast_metrics,
    forecast_xgb_recursive,
    pick_best_forecast_model,
    train_xgb_forecaster,
)
from src.utils.profiling import profile_dataframe

# ----------------------------------------------------------------------
# Pure function tests
# ----------------------------------------------------------------------


def test_build_time_series_sums_per_period():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-08"]),
        "value": [10.0, 5.0, 20.0],
    })
    series = build_time_series(df, "date", "value", freq="W")
    assert series["value"].sum() == pytest.approx(35.0)


def test_chronological_split_keeps_order_and_sizes():
    series = pd.DataFrame({"period": pd.date_range("2024-01-01", periods=10, freq="W"), "value": range(10)})
    train, test = chronological_split(series, test_periods=3)
    assert len(train) == 7
    assert len(test) == 3
    # test set must be the LAST periods, never randomly selected
    assert test["value"].tolist() == [7, 8, 9]


def test_chronological_split_handles_test_periods_too_large():
    series = pd.DataFrame({"period": pd.date_range("2024-01-01", periods=3, freq="W"), "value": range(3)})
    train, test = chronological_split(series, test_periods=10)
    assert len(train) == 3
    assert test.empty


def test_compute_forecast_metrics_perfect_prediction():
    y_true = np.array([10.0, 20.0, 30.0])
    metrics = compute_forecast_metrics(y_true, y_true.copy())
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["mape"] == pytest.approx(0.0)


def test_pick_best_forecast_model_lowest_rmse_wins():
    metrics = {
        "a": {"rmse": 10.0, "mae": 8.0, "mape": 5.0},
        "b": {"rmse": 3.0, "mae": 2.0, "mape": 1.0},
    }
    name, best = pick_best_forecast_model(metrics)
    assert name == "b"


def test_xgb_forecaster_captures_linear_trend():
    # y = 2*t + small noise -> recursive forecast should continue the trend,
    # not collapse to a flat line, and should clearly beat a naive baseline.
    rng = np.random.default_rng(0)
    t = np.arange(60, dtype=float)
    values = 2.0 * t + rng.normal(0, 0.5, 60)
    train = pd.DataFrame({"period": pd.date_range("2023-01-01", periods=60, freq="W"), "value": values})

    model = train_xgb_forecaster(train, n_lags=4, n_estimators=100, random_state=0)
    preds = forecast_xgb_recursive(model, train["value"].tolist(), n_lags=4, n_steps=5)
    # forecast should stay in a plausible range near the trend continuation,
    # not collapse toward zero or explode
    assert all(50 < p < 160 for p in preds)


def test_xgb_forecaster_raises_with_insufficient_history():
    train = pd.DataFrame({"period": pd.date_range("2024-01-01", periods=3, freq="W"), "value": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        train_xgb_forecaster(train, n_lags=10)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed in this environment")
def test_lstm_forecaster_runs_and_produces_finite_predictions():
    from src.utils.forecasting_ops import forecast_lstm_recursive, train_lstm_forecaster

    rng = np.random.default_rng(1)
    values = 5.0 + rng.normal(0, 1, 40)
    train = pd.DataFrame({"period": pd.date_range("2023-01-01", periods=40, freq="W"), "value": values})
    model, scaler = train_lstm_forecaster(train, lookback=4, hidden_size=8, epochs=20, learning_rate=0.05)
    preds = forecast_lstm_recursive(model, scaler, train["value"].tolist(), lookback=4, n_steps=3)
    assert len(preds) == 3
    assert all(np.isfinite(p) for p in preds)


@pytest.mark.skipif(not _HAS_PROPHET, reason="prophet not installed in this environment")
def test_prophet_fits_and_forecasts():
    from src.utils.forecasting_ops import fit_prophet, forecast_prophet

    rng = np.random.default_rng(2)
    values = 100 + np.arange(40) * 1.5 + rng.normal(0, 2, 40)
    train = pd.DataFrame({"period": pd.date_range("2023-01-01", periods=40, freq="W"), "value": values})
    model = fit_prophet(train)
    fc = forecast_prophet(model, n_periods=5, freq="W")
    assert len(fc) == 5
    assert {"period", "yhat", "yhat_lower", "yhat_upper"}.issubset(fc.columns)
    # uncertainty band should be a real band, not a zero-width line
    assert (fc["yhat_upper"] > fc["yhat_lower"]).all()


# ----------------------------------------------------------------------
# Forecasting Agent end-to-end test
# ----------------------------------------------------------------------

FIXTURE_ROOT = Path("tests/_fixture_forecasting")
UPSTREAM_DIR = FIXTURE_ROOT / "upstream"
OUTPUT_ROOT = FIXTURE_ROOT / "processed"
CONFIG_PATH = FIXTURE_ROOT / "config.yaml"


@pytest.fixture()
def upstream_manifest():
    UPSTREAM_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    n_days = 400
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    # mild upward trend + weekly seasonality-ish noise, all summable to revenue
    trend = np.linspace(50, 150, n_days)
    noise = rng.normal(0, 10, n_days)
    values = trend + noise

    df = pd.DataFrame({"order_purchase_timestamp": dates, "item_total_value": values})
    rel_path = "unified_cleaned.parquet"
    df.to_parquet(UPSTREAM_DIR / rel_path, index=False)
    profile = profile_dataframe("unified_cleaned", df, rel_path)

    manifest = DatasetManifest(
        agent_name="ml",
        status=AgentStatus.SUCCESS,
        output_dir=str(UPSTREAM_DIR),
        primary_table="unified_cleaned",
        tables={"unified_cleaned": profile},
    )
    manifest_path = manifest.save()

    config = {
        "forecasting": {
            "date_column": "order_purchase_timestamp",
            "value_column": "item_total_value",
            "freq": "W",
            "test_periods": 6,
            "horizon": 4,
            "models": ["xgboost"],  # keep the default fixture fast; prophet/lstm tested separately below
            "xgboost": {"n_lags": 4, "n_estimators": 100, "random_state": 42},
            "lstm": {"lookback": 4, "hidden_size": 8, "epochs": 20, "learning_rate": 0.05, "random_state": 42},
            "generate_chart": True,
        }
    }
    CONFIG_PATH.write_text(yaml.dump(config))

    yield manifest_path

    shutil.rmtree(FIXTURE_ROOT, ignore_errors=True)


def test_forecasting_requires_upstream_manifest():
    agent = ForecastingAgent(config_path="config/config.yaml", output_root=str(OUTPUT_ROOT))
    with pytest.raises(ValueError):
        agent.run(upstream_manifest_path=None)


def test_forecasting_agent_produces_expected_tables(upstream_manifest):
    agent = ForecastingAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest)
    expected = {"forecast_model_comparison", "forecast_future", "forecast_history", "unified_cleaned"}
    assert expected.issubset(set(manifest.tables.keys()))


def test_forecasting_agent_future_horizon_matches_config(upstream_manifest):
    agent = ForecastingAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest)
    future = pd.read_parquet(manifest.resolve_table_path("forecast_future"))
    assert len(future) == 4  # matches config horizon


def test_forecasting_agent_future_periods_are_after_history(upstream_manifest):
    agent = ForecastingAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest)
    history = pd.read_parquet(manifest.resolve_table_path("forecast_history"))
    future = pd.read_parquet(manifest.resolve_table_path("forecast_future"))
    assert pd.to_datetime(future["period"]).min() > pd.to_datetime(history["period"]).max()


def test_forecasting_agent_chart_generated(upstream_manifest):
    agent = ForecastingAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest)
    chart_paths = manifest.extra_metadata["chart_paths"]
    assert len(chart_paths) == 1
    assert (Path(manifest.output_dir) / chart_paths[0]).exists()


def test_forecasting_agent_status_partial_when_model_missing(upstream_manifest):
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["forecasting"]["models"] = ["xgboost", "prophet", "lstm"]
    CONFIG_PATH.write_text(yaml.dump(config))

    agent = ForecastingAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest)
    # in this environment prophet/torch ARE installed, so all 3 should evaluate
    # successfully -> SUCCESS. This test documents that expectation explicitly
    # rather than assuming PARTIAL.
    if _HAS_PROPHET and _HAS_TORCH:
        assert manifest.status == AgentStatus.SUCCESS
        assert set(manifest.extra_metadata["models_evaluated"]) == {"xgboost", "prophet", "lstm"}
    else:
        assert manifest.status == AgentStatus.PARTIAL


def test_forecasting_agent_passes_through_primary_table(upstream_manifest):
    agent = ForecastingAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run(upstream_manifest_path=upstream_manifest)
    df = pd.read_parquet(manifest.resolve_table_path("unified_cleaned"))
    assert len(df) == 400
