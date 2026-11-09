"""
Reusable forecasting operations, same pattern as cleaning_ops.py / eda_ops.py /
ml_ops.py: small, testable functions the Forecasting Agent orchestrates.

Three model families, per the original project spec's Forecasting Agent
section: Prophet, XGBoost (via lag features + recursive multi-step
prediction), and LSTM (via a lookback window + recursive multi-step
prediction). Each is optional at import time — if a person's environment
doesn't have `prophet` or `torch` installed, the corresponding functions
raise a clear ImportError-derived message rather than crashing the whole
agent; the agent itself decides whether to skip or fail (see forecasting_agent.py).

All three models are evaluated the same way: fit on a chronological train
split, forecast exactly as many periods as were held out, compare to the
actual held-out values. Whichever wins gets retrained on the FULL series to
produce the actual future forecast.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.eda_ops import compute_time_series_trend

try:
    from prophet import Prophet

    _HAS_PROPHET = True
except ImportError:  # pragma: no cover - environment-dependent
    _HAS_PROPHET = False

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - environment-dependent
    _HAS_TORCH = False

# Prophet/cmdstanpy are chatty at INFO level by default — quiet them down so
# pipeline logs stay readable.
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


# ----------------------------------------------------------------------
# Series construction / splitting
# ----------------------------------------------------------------------

def build_time_series(
    df: pd.DataFrame, date_column: str, value_column: str, freq: str = "W"
) -> pd.DataFrame:
    """Returns a clean [period(datetime), value(float)] series, summed per
    bucket. Reuses the EDA Agent's bucketing logic (compute_time_series_trend)
    so the Forecasting Agent's notion of 'weekly revenue' is identical to what
    the EDA Agent already showed the person — no second, subtly different
    definition of the same series."""
    trend = compute_time_series_trend(df, date_column, value_column, freq=freq)
    if trend.empty:
        return pd.DataFrame(columns=["period", "value"])
    out = trend[["period", "value_sum"]].rename(columns={"value_sum": "value"})
    out["period"] = pd.to_datetime(out["period"])
    return out.sort_values("period").reset_index(drop=True)


def chronological_split(series: pd.DataFrame, test_periods: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time series splits must never be random — the test set has to be the
    most recent periods, or the model gets to 'see the future' during
    training."""
    if test_periods <= 0 or test_periods >= len(series):
        return series.copy(), pd.DataFrame(columns=series.columns)
    return series.iloc[:-test_periods].reset_index(drop=True), series.iloc[-test_periods:].reset_index(drop=True)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def compute_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    nonzero = y_true != 0
    mape = float(np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else float("nan")
    return {"rmse": rmse, "mae": mae, "mape": mape}


def pick_best_forecast_model(metrics_by_model: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    return min(metrics_by_model.items(), key=lambda kv: kv[1]["rmse"])


# ----------------------------------------------------------------------
# Prophet
# ----------------------------------------------------------------------

def fit_prophet(train: pd.DataFrame) -> "Prophet":
    if not _HAS_PROPHET:
        raise ImportError("prophet is not installed — `pip install prophet` to enable this model")
    prophet_df = train.rename(columns={"period": "ds", "value": "y"})
    model = Prophet(interval_width=0.8)
    model.fit(prophet_df)
    return model


def forecast_prophet(model: "Prophet", n_periods: int, freq: str) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=n_periods, freq=freq)
    forecast = model.predict(future)
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(n_periods).reset_index(drop=True)
    return result.rename(columns={"ds": "period"})


# ----------------------------------------------------------------------
# XGBoost (lag features + recursive multi-step forecasting)
# ----------------------------------------------------------------------

def _make_lag_features(values: List[float], n_lags: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(n_lags, len(values)):
        X.append(values[i - n_lags:i])
        y.append(values[i])
    return np.array(X), np.array(y)


def train_xgb_forecaster(train: pd.DataFrame, n_lags: int, n_estimators: int = 200, random_state: int = 42):
    from xgboost import XGBRegressor

    values = train["value"].tolist()
    if len(values) <= n_lags:
        raise ValueError(
            f"Not enough training periods ({len(values)}) for n_lags={n_lags} — "
            "reduce forecasting.xgboost.n_lags or provide more history."
        )
    X, y = _make_lag_features(values, n_lags)
    model = XGBRegressor(n_estimators=n_estimators, random_state=random_state, verbosity=0)
    model.fit(X, y)
    return model


def forecast_xgb_recursive(model, history: List[float], n_lags: int, n_steps: int) -> List[float]:
    """Recursive multi-step forecasting: predict one step ahead, append that
    prediction to the history as if it were observed, predict the next step
    from the new window, repeat. Standard technique when future exogenous lag
    values genuinely don't exist yet."""
    window = list(history[-n_lags:])
    predictions = []
    for _ in range(n_steps):
        x = np.array(window[-n_lags:]).reshape(1, -1)
        pred = float(model.predict(x)[0])
        predictions.append(pred)
        window.append(pred)
    return predictions


# ----------------------------------------------------------------------
# LSTM (lookback window + recursive multi-step forecasting)
# ----------------------------------------------------------------------

if _HAS_TORCH:

    class LSTMForecaster(nn.Module):
        def __init__(self, hidden_size: int = 16):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

else:  # pragma: no cover - environment-dependent
    LSTMForecaster = None  # type: ignore[assignment]


class _MinMaxScaler1D:
    """Tiny hand-rolled scaler so this module doesn't need sklearn for one
    column — keeps the LSTM path self-contained and easy to persist alongside
    the torch state_dict."""

    def __init__(self):
        self.min_ = 0.0
        self.max_ = 1.0

    def fit(self, values: np.ndarray) -> "_MinMaxScaler1D":
        self.min_, self.max_ = float(values.min()), float(values.max())
        if self.max_ == self.min_:
            self.max_ = self.min_ + 1.0  # avoid divide-by-zero on a constant series
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.min_) / (self.max_ - self.min_)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * (self.max_ - self.min_) + self.min_


def train_lstm_forecaster(
    train: pd.DataFrame,
    lookback: int,
    hidden_size: int = 16,
    epochs: int = 150,
    learning_rate: float = 0.01,
    random_state: int = 42,
) -> Tuple["LSTMForecaster", _MinMaxScaler1D]:
    if not _HAS_TORCH:
        raise ImportError("torch is not installed — `pip install torch` to enable this model")

    values = train["value"].to_numpy(dtype=float)
    if len(values) <= lookback:
        raise ValueError(
            f"Not enough training periods ({len(values)}) for lookback={lookback} — "
            "reduce forecasting.lstm.lookback or provide more history."
        )

    torch.manual_seed(random_state)
    scaler = _MinMaxScaler1D().fit(values)
    scaled = scaler.transform(values)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i])
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)  # (n, lookback, 1)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)  # (n, 1)

    model = LSTMForecaster(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

    return model, scaler


def forecast_lstm_recursive(
    model: "LSTMForecaster", scaler: _MinMaxScaler1D, history: List[float], lookback: int, n_steps: int
) -> List[float]:
    model.eval()
    window = list(scaler.transform(np.array(history[-lookback:], dtype=float)))
    predictions_scaled = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.tensor(window[-lookback:], dtype=torch.float32).reshape(1, lookback, 1)
            pred_scaled = float(model(x).item())
            predictions_scaled.append(pred_scaled)
            window.append(pred_scaled)
    return list(scaler.inverse_transform(np.array(predictions_scaled)))


# ----------------------------------------------------------------------
# Residual-based uncertainty band for models without native intervals
# ----------------------------------------------------------------------

def residual_interval(y_true: np.ndarray, y_pred: np.ndarray, z: float = 1.28) -> float:
    """80%-ish interval half-width from in-sample residual std (z=1.28 ~ 80%
    under a normal approximation), for XGBoost/LSTM which don't produce
    intervals natively the way Prophet does. Deliberately simple — this is
    not a substitute for proper quantile/conformal forecasting, just enough
    to put a visible uncertainty band on the chart."""
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.std(residuals) * z) if len(residuals) else 0.0
