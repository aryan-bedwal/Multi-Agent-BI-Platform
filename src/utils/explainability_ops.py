"""
Reusable explainability operations, same pattern as the other agents'
*_ops.py modules.

Two genuinely different explanation problems, two different techniques:
  - `explain_predictions_shap` / `explain_predictions_lime` explain a
    SUPERVISED model's output — the ML Agent's trained pipeline.
  - `compute_anomaly_reasons` explains an UNSUPERVISED ensemble VOTE (the
    Anomaly Detection Agent's consensus flags), which has no single
    differentiable output for SHAP/LIME to attribute against. It uses
    feature z-score ranking instead — simpler, but actually well-defined
    for "why did these 4 methods flag this row", unlike forcing SHAP onto
    a vote count.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import Pipeline


# ----------------------------------------------------------------------
# SHAP
# ----------------------------------------------------------------------

def build_shap_explainer(model, X_background: np.ndarray):
    """Picks the fast, exact explainer for the model type actually used by
    the ML Agent (tree ensembles, linear models) rather than falling back to
    the slow, approximate generic KernelExplainer — this only needs to work
    for the models get_candidate_models() in ml_ops.py can produce."""
    if hasattr(model, "feature_importances_"):  # RandomForest, XGBoost
        return shap.TreeExplainer(model)
    if hasattr(model, "coef_"):  # Linear/LogisticRegression
        return shap.LinearExplainer(model, X_background)
    # Fallback for anything else — slower, but correct for arbitrary models.
    return shap.Explainer(model.predict, X_background)


def compute_shap_values(explainer, X: np.ndarray) -> np.ndarray:
    raw = explainer.shap_values(X)
    # Classifiers with >1 class return a list/3D array (one set per class) —
    # collapse to the positive-class attribution for binary classification,
    # which is the common case this agent will see.
    if isinstance(raw, list):
        raw = raw[-1]
    arr = np.asarray(raw)
    if arr.ndim == 3:  # (n_classes, n_rows, n_features) or (n_rows, n_features, n_classes)
        arr = arr[..., -1] if arr.shape[-1] < arr.shape[0] else arr[-1]
    return arr


def shap_global_summary(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def explain_row_shap(
    shap_values_row: np.ndarray, feature_names: List[str], feature_values_row: np.ndarray, top_n: int = 5
) -> List[Dict]:
    order = np.argsort(-np.abs(shap_values_row))[:top_n]
    return [
        {
            "feature": feature_names[i],
            "feature_value": float(feature_values_row[i]),
            "shap_value": float(shap_values_row[i]),
            "direction": "increases" if shap_values_row[i] > 0 else "decreases",
        }
        for i in order
    ]


# ----------------------------------------------------------------------
# LIME
# ----------------------------------------------------------------------

def build_lime_explainer(
    X_background: np.ndarray, feature_names: List[str], mode: str = "regression", random_state: int = 42
) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        X_background,
        feature_names=feature_names,
        mode=mode,
        random_state=random_state,
        discretize_continuous=True,
    )


def explain_row_lime(
    lime_explainer: LimeTabularExplainer, predict_fn, x_row: np.ndarray, top_n: int = 5
) -> List[Dict]:
    explanation = lime_explainer.explain_instance(x_row, predict_fn, num_features=top_n)
    return [
        {"feature_condition": desc, "weight": float(weight)}
        for desc, weight in explanation.as_list()
    ]


# ----------------------------------------------------------------------
# Anomaly reasons (deviation-based, not SHAP — see module docstring)
# ----------------------------------------------------------------------

def compute_anomaly_reasons(
    df: pd.DataFrame, numeric_columns: List[str], anomaly_mask: pd.Series, top_n: int = 5
) -> pd.DataFrame:
    """For every row where anomaly_mask is True, rank numeric features by
    |z-score| against the FULL population's mean/std, and report the top_n —
    i.e. 'why does this row look unusual': which specific numbers are most
    out of line with everyone else's."""
    means = df[numeric_columns].mean()
    stds = df[numeric_columns].std().replace(0, np.nan)
    z_scores = (df[numeric_columns] - means) / stds

    rows = []
    flagged_idx = df.index[anomaly_mask]
    for idx in flagged_idx:
        row_z = z_scores.loc[idx].abs().sort_values(ascending=False)
        for rank, (feature, z) in enumerate(row_z.head(top_n).items(), start=1):
            if pd.isna(z):
                continue
            rows.append({
                "row_index": int(idx),
                "rank": rank,
                "feature": feature,
                "value": float(df.loc[idx, feature]),
                "population_mean": float(means[feature]),
                "z_score": float(z_scores.loc[idx, feature]),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Row selection for illustrative examples
# ----------------------------------------------------------------------

def select_example_rows(
    y_pred: np.ndarray, n_total: int, n_random: int = 3, random_state: int = 42
) -> np.ndarray:
    """Picks the highest-predicted, lowest-predicted, and a random sample of
    row POSITIONS (0-indexed into the array y_pred was computed from) — so
    the illustrative explanations cover a spread of cases, not just whichever
    rows happened to come first."""
    rng = np.random.default_rng(random_state)
    n_extremes = max(0, (n_total - n_random) // 2)
    order = np.argsort(y_pred)
    lowest = order[:n_extremes]
    highest = order[-n_extremes:] if n_extremes else np.array([], dtype=int)
    remaining = np.setdiff1d(np.arange(len(y_pred)), np.concatenate([lowest, highest]))
    random_pick = rng.choice(remaining, size=min(n_random, len(remaining)), replace=False)
    return np.unique(np.concatenate([lowest, highest, random_pick]))
