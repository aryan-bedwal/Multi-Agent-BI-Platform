"""
Reusable anomaly detection operations, same pattern as the other agents'
*_ops.py modules: small, testable functions the Anomaly Detection Agent
orchestrates.

Two distinct kinds of anomaly, per the project spec:
  1. Row-level (fraud-like) — four independent algorithms score every row;
     a row only counts as a consensus anomaly if enough of them agree.
  2. Time-series (sudden revenue drop/spike) — rolling z-score on the same
     weekly revenue series the EDA/Forecasting Agents already use.

The autoencoder needs torch; the other three are pure scikit-learn. Torch is
optional at import time (same pattern as forecasting_ops.py) — if it's
missing, the ensemble just runs with 3 methods instead of 4 and the agent
logs that rather than crashing.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - environment-dependent
    _HAS_TORCH = False


# ----------------------------------------------------------------------
# Row-level: four algorithms, each returning (is_anomaly: bool array, score: float array)
# Score convention across all four: HIGHER score = MORE anomalous, so they're
# directly comparable/rankable regardless of each algorithm's native convention.
# ----------------------------------------------------------------------

def run_isolation_forest(
    X: np.ndarray, contamination: float, n_estimators: int = 200, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    model = IsolationForest(
        n_estimators=n_estimators, contamination=contamination, random_state=random_state
    )
    labels = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    raw_score = model.decision_function(X)  # HIGHER = more normal (native convention)
    score = -raw_score  # flip so higher = more anomalous, matching our shared convention
    return labels == -1, score


def run_lof(X: np.ndarray, contamination: float, n_neighbors: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    n_neighbors = min(n_neighbors, max(1, len(X) - 1))
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    score = -model.negative_outlier_factor_  # native: more negative = more anomalous -> flip
    return labels == -1, score


def run_one_class_svm(X: np.ndarray, nu: float, kernel: str = "rbf", gamma: str = "scale") -> Tuple[np.ndarray, np.ndarray]:
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    labels = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    raw_score = model.decision_function(X)  # HIGHER = more normal (native convention)
    score = -raw_score
    return labels == -1, score


if _HAS_TORCH:

    class AnomalyAutoencoder(nn.Module):
        def __init__(self, input_dim: int, hidden_size: int = 16, bottleneck_size: int = 4):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, bottleneck_size),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

else:  # pragma: no cover - environment-dependent
    AnomalyAutoencoder = None  # type: ignore[assignment]


def run_autoencoder(
    X: np.ndarray,
    contamination: float,
    hidden_size: int = 16,
    bottleneck_size: int = 4,
    epochs: int = 100,
    learning_rate: float = 0.01,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trains on the FULL dataset (unsupervised, no train/test split needed —
    the model learns to reconstruct 'normal' patterns; rows it reconstructs
    badly are the anomalies). Threshold is the (1-contamination) percentile
    of reconstruction error, so this method targets the same strictness as
    the other three."""
    if not _HAS_TORCH:
        raise ImportError("torch is not installed — `pip install torch` to enable this model")

    torch.manual_seed(random_state)
    X_t = torch.tensor(X, dtype=torch.float32)
    model = AnomalyAutoencoder(input_dim=X.shape[1], hidden_size=hidden_size, bottleneck_size=bottleneck_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        reconstruction = model(X_t)
        loss = loss_fn(reconstruction, X_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        reconstruction = model(X_t)
        per_row_error = torch.mean((reconstruction - X_t) ** 2, dim=1).numpy()

    threshold = np.percentile(per_row_error, 100 * (1 - contamination))
    return per_row_error > threshold, per_row_error


# ----------------------------------------------------------------------
# Ensemble voting
# ----------------------------------------------------------------------

def ensemble_vote(flags_by_method: Dict[str, np.ndarray], min_votes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (vote_count per row, consensus boolean per row)."""
    vote_matrix = np.vstack(list(flags_by_method.values()))  # (n_methods, n_rows)
    vote_count = vote_matrix.sum(axis=0)
    consensus = vote_count >= min_votes
    return vote_count, consensus


# ----------------------------------------------------------------------
# Time-series anomalies (rolling z-score on a period series)
# ----------------------------------------------------------------------

def detect_time_series_anomalies(
    series: pd.DataFrame, rolling_window: int = 6, z_threshold: float = 2.0
) -> pd.DataFrame:
    """series: [period, value]. Flags periods whose value deviates more than
    `z_threshold` standard deviations from a trailing rolling mean/std — this
    adapts to trend/seasonality better than a single global mean/std would,
    which matters for a business series that's generally growing over time."""
    out = series.copy().sort_values("period").reset_index(drop=True)
    out["rolling_mean"] = out["value"].rolling(window=rolling_window, min_periods=2).mean().shift(1)
    out["rolling_std"] = out["value"].rolling(window=rolling_window, min_periods=2).std().shift(1)

    diff = out["value"] - out["rolling_mean"]
    safe_std = out["rolling_std"].replace(0, np.nan)
    out["z_score"] = diff / safe_std

    # A perfectly flat baseline (rolling_std == 0) isn't a "can't tell" case —
    # it's the easiest case: ANY deviation from a constant baseline is
    # meaningful. Give it a z-score that clears the threshold rather than
    # dividing by zero into NaN (which would silently mean 'never flaggable').
    sentinel_magnitude = z_threshold + 1.0  # guaranteed to exceed the threshold
    flat_baseline = (out["rolling_std"] == 0) & diff.notna()
    flat_and_different = flat_baseline & (diff != 0)
    out.loc[flat_and_different, "z_score"] = np.sign(diff[flat_and_different]) * sentinel_magnitude
    out.loc[flat_baseline & (diff == 0), "z_score"] = 0.0

    out["is_anomalous"] = out["z_score"].abs() > z_threshold
    out["direction"] = np.select(
        [out["z_score"] > z_threshold, out["z_score"] < -z_threshold],
        ["spike", "drop"],
        default="normal",
    )
    # first `rolling_window` periods have no baseline yet — not flaggable
    out.loc[out["rolling_mean"].isna(), ["is_anomalous", "direction"]] = [False, "insufficient_history"]
    return out
