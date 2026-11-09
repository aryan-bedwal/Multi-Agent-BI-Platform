"""
Reusable cleaning operations, kept as pure functions (df in, df + report out)
so they're easy to unit test in isolation and easy for a future agent (e.g.
EDA, which will want 'what did cleaning change' context) to reuse without
importing the whole agent.

Each function returns (transformed_df, report_dict) — reports get merged into
the Cleaning Agent's manifest so the transformation is auditable, not silent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. Duplicates
# ----------------------------------------------------------------------

def remove_duplicates(
    df: pd.DataFrame, subset: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    n_before = len(df)
    subset = [c for c in (subset or []) if c in df.columns] or None
    out = df.drop_duplicates(subset=subset)
    n_removed = n_before - len(out)
    return out.reset_index(drop=True), {
        "n_before": n_before,
        "n_removed": n_removed,
        "subset_used": subset,
    }


# ----------------------------------------------------------------------
# 2. Dtype conversion
# ----------------------------------------------------------------------

def apply_dtype_overrides(
    df: pd.DataFrame, dtype_map: Dict[str, str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    applied: Dict[str, str] = {}
    failed: Dict[str, str] = {}
    for col, dtype in dtype_map.items():
        if col not in out.columns:
            continue
        before = str(out[col].dtype)
        try:
            if dtype == "category":
                out[col] = out[col].astype("category")
            elif dtype in ("string", "str"):
                out[col] = out[col].astype("string")
            else:
                out[col] = out[col].astype(dtype)
            applied[col] = f"{before} -> {dtype}"
        except (ValueError, TypeError) as exc:
            failed[col] = f"{before} -> {dtype} failed: {exc}"
    return out, {"applied": applied, "failed": failed}


# ----------------------------------------------------------------------
# 3. String standardization
# ----------------------------------------------------------------------

def standardize_strings(
    df: pd.DataFrame, columns_config: Dict[str, str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """columns_config: {column: 'lower' | 'upper' | 'title' | 'strip_only'}"""
    out = df.copy()
    touched: List[str] = []
    for col, mode in columns_config.items():
        if col not in out.columns:
            continue
        series = out[col].astype("string")
        series = series.str.strip()
        series = series.str.replace(r"\s+", " ", regex=True)
        if mode == "lower":
            series = series.str.lower()
        elif mode == "upper":
            series = series.str.upper()
        elif mode == "title":
            series = series.str.title()
        # 'strip_only' -> no case change
        out[col] = series
        touched.append(col)
    return out, {"columns_standardized": touched}


# ----------------------------------------------------------------------
# 4. Missing values
# ----------------------------------------------------------------------

def impute_missing(
    df: pd.DataFrame,
    strategy_map: Dict[str, str],
    default_numeric: str = "median",
    default_categorical: str = "mode",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    strategy per column: 'median' | 'mean' | 'mode' | 'constant:<value>' |
    'flag_only' (leave null, add <col>_was_missing bool column) | 'skip'
    (leave completely untouched — for columns like delivery_days where NaN
    is a legitimate business meaning: 'not yet delivered', not a data error).

    Columns not in strategy_map fall back to default_numeric/default_categorical
    by dtype; datetime columns default to 'flag_only' since imputing a date is
    rarely sound and downstream agents (Forecasting) need to know it was missing.
    """
    out = df.copy()
    report: Dict[str, Dict[str, Any]] = {}

    for col in out.columns:
        n_missing_before = int(out[col].isna().sum())
        if n_missing_before == 0:
            continue

        strategy = strategy_map.get(col)
        if strategy is None:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                strategy = "flag_only"
            elif pd.api.types.is_numeric_dtype(out[col]):
                strategy = default_numeric
            else:
                strategy = default_categorical

        if strategy == "skip":
            report[col] = {"n_missing_before": n_missing_before, "strategy": "skip", "n_missing_after": n_missing_before}
            continue

        if strategy == "flag_only":
            out[f"{col}_was_missing"] = out[col].isna()
            report[col] = {
                "n_missing_before": n_missing_before,
                "strategy": "flag_only",
                "n_missing_after": n_missing_before,
                "flag_column": f"{col}_was_missing",
            }
            continue

        if strategy == "median":
            fill_value = out[col].median()
        elif strategy == "mean":
            fill_value = out[col].mean()
        elif strategy == "mode":
            mode_vals = out[col].mode()
            fill_value = mode_vals.iat[0] if not mode_vals.empty else None
        elif strategy.startswith("constant:"):
            fill_value = strategy.split(":", 1)[1]
        else:
            fill_value = None

        if fill_value is not None:
            out[col] = out[col].fillna(fill_value)

        n_missing_after = int(out[col].isna().sum())
        report[col] = {
            "n_missing_before": n_missing_before,
            "strategy": strategy,
            "fill_value": str(fill_value),
            "n_missing_after": n_missing_after,
        }

    return out, report


# ----------------------------------------------------------------------
# 5. Outliers
# ----------------------------------------------------------------------

def handle_outliers(
    df: pd.DataFrame,
    columns_config: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    columns_config: {column: {"action": "cap" | "flag", "iqr_multiplier": float}}

    Deliberately conservative: this is a data-quality step, not the Anomaly
    Detection Agent. Default action is 'flag' (add a boolean column, don't
    touch the value) so legitimate business outliers (e.g. a genuinely huge
    order) survive for the Anomaly Detection Agent to reason about later.
    'cap' (winsorize to the IQR bound) is opt-in per column for cases where
    extreme values are almost certainly data-entry errors.
    """
    out = df.copy()
    report: Dict[str, Any] = {}

    for col, cfg in columns_config.items():
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue

        multiplier = cfg.get("iqr_multiplier", 3.0)
        action = cfg.get("action", "flag")

        q1, q3 = out[col].quantile(0.25), out[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        is_outlier = (out[col] < lower) | (out[col] > upper)
        n_outliers = int(is_outlier.sum())

        if action == "cap":
            out[col] = out[col].clip(lower=lower, upper=upper)
        else:  # flag
            out[f"{col}_is_outlier"] = is_outlier

        report[col] = {
            "action": action,
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "n_outliers": n_outliers,
            "pct_outliers": round(n_outliers / len(out) * 100, 2) if len(out) else 0.0,
        }

    return out, report
