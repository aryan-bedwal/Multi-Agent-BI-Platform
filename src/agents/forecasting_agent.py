"""
Forecasting Agent
===================
Fifth agent in the pipeline. Consumes the ML Agent's manifest and forecasts
future values of a business time series (default: weekly revenue) using the
three model families named in the original project spec: Prophet, XGBoost,
and LSTM.

Responsibilities (per project spec):
  - Predict future values (sales, inventory, traffic, growth — here: revenue)
  - Models: Prophet, XGBoost, LSTM

Evaluation methodology:
  1. Build the [period, value] series (same bucketing the EDA Agent used)
  2. Chronological split — hold out the most recent `test_periods` periods.
     NEVER a random split: that would let the model train on the future.
  3. Fit each available model on the train portion, forecast exactly
     `test_periods` steps ahead, compare to the actual held-out values
  4. Whichever wins on RMSE gets RETRAINED on the full series (train+test)
     and used to forecast `horizon` genuinely-future periods

Any model whose dependency isn't installed (prophet, torch) is skipped with
a warning rather than failing the whole run — status downgrades to PARTIAL
if fewer than all configured models could be evaluated, but the agent still
produces a usable forecast from whatever did train.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

from src.agents.base_agent import BaseAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.forecasting_charts import plot_forecast
from src.utils.forecasting_ops import (
    _HAS_PROPHET,
    _HAS_TORCH,
    build_time_series,
    chronological_split,
    compute_forecast_metrics,
    fit_prophet,
    forecast_lstm_recursive,
    forecast_prophet,
    forecast_xgb_recursive,
    pick_best_forecast_model,
    train_lstm_forecaster,
    train_xgb_forecaster,
)
from src.utils.profiling import profile_dataframe


class ForecastingAgent(BaseAgent):
    agent_name = "forecasting"

    def __init__(self, config_path: str = "config/config.yaml", output_root: str = "data/processed"):
        super().__init__(output_root=output_root)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["forecasting"]

    # ------------------------------------------------------------------
    # Evaluation: fit each model on train, forecast len(test) steps, score
    # ------------------------------------------------------------------

    def _evaluate_models(
        self, train: pd.DataFrame, test: pd.DataFrame, model_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        metrics_by_model: Dict[str, Dict[str, float]] = {}
        freq = self.config.get("freq", "W")
        n_test = len(test)

        if "prophet" in model_names:
            if not _HAS_PROPHET:
                self.log.warning("prophet not installed — skipping this model (pip install prophet)")
            else:
                try:
                    model = fit_prophet(train)
                    fc = forecast_prophet(model, n_periods=n_test, freq=freq)
                    metrics_by_model["prophet"] = compute_forecast_metrics(test["value"].values, fc["yhat"].values)
                    self.log.info(f"prophet eval metrics: {metrics_by_model['prophet']}")
                except Exception as exc:  # noqa: BLE001
                    self.log.warning(f"prophet evaluation failed: {exc}")

        if "xgboost" in model_names:
            xgb_cfg = self.config.get("xgboost", {})
            n_lags = xgb_cfg.get("n_lags", 4)
            try:
                model = train_xgb_forecaster(
                    train, n_lags=n_lags,
                    n_estimators=xgb_cfg.get("n_estimators", 200),
                    random_state=xgb_cfg.get("random_state", 42),
                )
                preds = forecast_xgb_recursive(model, train["value"].tolist(), n_lags=n_lags, n_steps=n_test)
                metrics_by_model["xgboost"] = compute_forecast_metrics(test["value"].values, np.array(preds))
                self.log.info(f"xgboost eval metrics: {metrics_by_model['xgboost']}")
            except Exception as exc:  # noqa: BLE001
                self.log.warning(f"xgboost evaluation failed: {exc}")

        if "lstm" in model_names:
            if not _HAS_TORCH:
                self.log.warning("torch not installed — skipping LSTM (pip install torch)")
            else:
                lstm_cfg = self.config.get("lstm", {})
                lookback = lstm_cfg.get("lookback", 4)
                try:
                    model, scaler = train_lstm_forecaster(
                        train, lookback=lookback,
                        hidden_size=lstm_cfg.get("hidden_size", 16),
                        epochs=lstm_cfg.get("epochs", 150),
                        learning_rate=lstm_cfg.get("learning_rate", 0.01),
                        random_state=lstm_cfg.get("random_state", 42),
                    )
                    preds = forecast_lstm_recursive(
                        model, scaler, train["value"].tolist(), lookback=lookback, n_steps=n_test
                    )
                    metrics_by_model["lstm"] = compute_forecast_metrics(test["value"].values, np.array(preds))
                    self.log.info(f"lstm eval metrics: {metrics_by_model['lstm']}")
                except Exception as exc:  # noqa: BLE001
                    self.log.warning(f"lstm evaluation failed: {exc}")

        return metrics_by_model

    # ------------------------------------------------------------------
    # Final forecast: retrain the winner on the FULL series
    # ------------------------------------------------------------------

    def _forecast_future(
        self, best_name: str, full_series: pd.DataFrame, horizon: int, in_sample_metrics: Dict[str, float]
    ) -> pd.DataFrame:
        freq = self.config.get("freq", "W")
        last_period = full_series["period"].iloc[-1]
        future_periods = pd.date_range(
            start=last_period, periods=horizon + 1, freq=freq
        )[1:]

        if best_name == "prophet":
            model = fit_prophet(full_series)
            fc = forecast_prophet(model, n_periods=horizon, freq=freq)
            return fc  # already has yhat_lower / yhat_upper from Prophet directly

        if best_name == "xgboost":
            xgb_cfg = self.config.get("xgboost", {})
            n_lags = xgb_cfg.get("n_lags", 4)
            model = train_xgb_forecaster(
                full_series, n_lags=n_lags,
                n_estimators=xgb_cfg.get("n_estimators", 200),
                random_state=xgb_cfg.get("random_state", 42),
            )
            preds = forecast_xgb_recursive(model, full_series["value"].tolist(), n_lags=n_lags, n_steps=horizon)
            half_width = in_sample_metrics.get("mae", 0.0) * 1.28
            return pd.DataFrame({
                "period": future_periods,
                "yhat": preds,
                "yhat_lower": np.array(preds) - half_width,
                "yhat_upper": np.array(preds) + half_width,
            })

        if best_name == "lstm":
            lstm_cfg = self.config.get("lstm", {})
            lookback = lstm_cfg.get("lookback", 4)
            model, scaler = train_lstm_forecaster(
                full_series, lookback=lookback,
                hidden_size=lstm_cfg.get("hidden_size", 16),
                epochs=lstm_cfg.get("epochs", 150),
                learning_rate=lstm_cfg.get("learning_rate", 0.01),
                random_state=lstm_cfg.get("random_state", 42),
            )
            preds = forecast_lstm_recursive(
                model, scaler, full_series["value"].tolist(), lookback=lookback, n_steps=horizon
            )
            half_width = in_sample_metrics.get("mae", 0.0) * 1.28 if in_sample_metrics else 0.0
            return pd.DataFrame({
                "period": future_periods,
                "yhat": preds,
                "yhat_lower": np.array(preds) - half_width,
                "yhat_upper": np.array(preds) + half_width,
            })

        raise ValueError(f"Unknown model name: {best_name}")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self, upstream: Optional[DatasetManifest]) -> DatasetManifest:
        if upstream is None:
            raise ValueError(
                "ForecastingAgent requires an upstream manifest — "
                "run MLAgent first and pass its manifest.json path."
            )
        if upstream.status == "failed":
            raise ValueError("Cannot forecast from a failed upstream run.")

        primary_path = upstream.resolve_table_path(upstream.primary_table)
        df = pd.read_parquet(primary_path)
        self.log.info(f"Loaded upstream primary table '{upstream.primary_table}': {len(df)} rows")

        date_col = self.config["date_column"]
        value_col = self.config["value_column"]
        freq = self.config.get("freq", "W")
        series = build_time_series(df, date_col, value_col, freq=freq)
        self.log.info(f"Built {freq}-frequency series on '{value_col}': {len(series)} periods")

        test_periods = self.config.get("test_periods", 8)
        train, test = chronological_split(series, test_periods)
        if test.empty:
            raise RuntimeError(
                f"Series has only {len(series)} periods — not enough to hold out "
                f"{test_periods} test periods. Reduce forecasting.test_periods."
            )
        self.log.info(f"Chronological split: {len(train)} train periods / {len(test)} test periods")

        model_names = self.config.get("models", ["prophet", "xgboost", "lstm"])
        metrics_by_model = self._evaluate_models(train, test, model_names)
        if not metrics_by_model:
            raise RuntimeError(
                "No forecasting model could be evaluated — check that at least one of "
                "prophet/xgboost/torch is installed."
            )

        status = AgentStatus.SUCCESS if len(metrics_by_model) == len(model_names) else AgentStatus.PARTIAL
        if status == AgentStatus.PARTIAL:
            self.log.warning(
                f"Only {len(metrics_by_model)}/{len(model_names)} configured models evaluated successfully"
            )

        best_name, best_metrics = pick_best_forecast_model(metrics_by_model)
        self.log.info(f"Best forecasting model: '{best_name}' with metrics {best_metrics}")

        horizon = self.config.get("horizon", 8)
        future_forecast = self._forecast_future(best_name, series, horizon, best_metrics)
        self.log.info(f"Forecasted {horizon} future periods with '{best_name}'")

        # -- persist outputs ---------------------------------------------
        profiles: Dict[str, object] = {}

        def _save_table(name: str, table: pd.DataFrame) -> None:
            rel_path = f"{name}.parquet"
            table.to_parquet(self.output_dir / rel_path, index=False)
            profiles[name] = profile_dataframe(name, table, rel_path)

        comparison_rows = [
            {"model": name, **metrics, "is_best": name == best_name}
            for name, metrics in metrics_by_model.items()
        ]
        _save_table("forecast_model_comparison", pd.DataFrame(comparison_rows))
        _save_table("forecast_future", future_forecast)
        _save_table("forecast_history", series)

        # pass the dataset through unchanged, same convention as EDA/ML agents
        passthrough_rel_path = "unified_passthrough.parquet"
        df.to_parquet(self.output_dir / passthrough_rel_path, index=False)
        profiles[upstream.primary_table] = profile_dataframe(
            upstream.primary_table, df, passthrough_rel_path
        )

        chart_paths: List[str] = []
        if self.config.get("generate_chart", True):
            chart_paths = plot_forecast(series, future_forecast, self.output_dir, value_col, best_name)
            self.log.info(f"Saved forecast chart under {self.output_dir}/charts/")

        return DatasetManifest(
            agent_name=self.agent_name,
            status=status,
            output_dir=str(self.output_dir),
            primary_table=upstream.primary_table,
            tables=profiles,
            extra_metadata={
                "date_column": date_col,
                "value_column": value_col,
                "freq": freq,
                "test_periods": test_periods,
                "horizon": horizon,
                "models_configured": model_names,
                "models_evaluated": list(metrics_by_model.keys()),
                "metrics_by_model": metrics_by_model,
                "best_model_name": best_name,
                "best_model_metrics": best_metrics,
                "chart_paths": chart_paths,
            },
        )


if __name__ == "__main__":
    from src.agents.data_cleaning_agent import DataCleaningAgent
    from src.agents.data_ingestion_agent import DataIngestionAgent
    from src.agents.eda_agent import EDAAgent
    from src.agents.ml_agent import MLAgent

    ml_manifest_path = "data/processed/ml/manifest.json"
    if not Path(ml_manifest_path).exists():
        print("No ML manifest found — running Ingestion + Cleaning + EDA + ML first.")
        ingestion_manifest = DataIngestionAgent().run()
        cleaning_manifest = DataCleaningAgent().run(
            upstream_manifest_path=f"{ingestion_manifest.output_dir}/manifest.json"
        )
        eda_manifest = EDAAgent().run(
            upstream_manifest_path=f"{cleaning_manifest.output_dir}/manifest.json"
        )
        ml_manifest = MLAgent().run(
            upstream_manifest_path=f"{eda_manifest.output_dir}/manifest.json"
        )
        ml_manifest_path = f"{ml_manifest.output_dir}/manifest.json"

    agent = ForecastingAgent()
    manifest = agent.run(upstream_manifest_path=ml_manifest_path)
    print("\n" + manifest.summary())
