"""
Reusable EDA computations, kept as pure functions (df in, summary df out) —
same pattern as cleaning_ops.py, so they're testable in isolation and reusable
by a future agent (e.g. the Recommendation agent may want the same category
distribution logic to phrase "Electronics sales dropped 18% in the South").
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def compute_numeric_summary(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return pd.DataFrame(
            columns=["column", "count", "mean", "std", "min", "p25", "p50", "p75", "max", "n_missing"]
        )
    rows = []
    for c in cols:
        s = df[c]
        rows.append({
            "column": c,
            "count": int(s.count()),
            "mean": float(s.mean()) if s.count() else None,
            "std": float(s.std()) if s.count() else None,
            "min": float(s.min()) if s.count() else None,
            "p25": float(s.quantile(0.25)) if s.count() else None,
            "p50": float(s.quantile(0.50)) if s.count() else None,
            "p75": float(s.quantile(0.75)) if s.count() else None,
            "max": float(s.max()) if s.count() else None,
            "n_missing": int(s.isna().sum()),
        })
    return pd.DataFrame(rows)


def compute_correlation_matrix(
    df: pd.DataFrame, columns: List[str], method: str = "pearson"
) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2:
        return pd.DataFrame()
    corr = df[cols].corr(method=method)
    corr = corr.reset_index().rename(columns={"index": "column"})
    return corr


def top_correlated_pairs(corr: pd.DataFrame, n: int = 5) -> List[Dict]:
    """corr is the wide matrix from compute_correlation_matrix (with a
    'column' index column). Returns the n strongest non-self correlations."""
    if corr.empty or "column" not in corr.columns:
        return []
    long = corr.melt(id_vars="column", var_name="column_b", value_name="correlation")
    long = long[long["column"] != long["column_b"]]
    long["abs_correlation"] = long["correlation"].abs()
    long = long.drop_duplicates(subset=["abs_correlation"]).sort_values(
        "abs_correlation", ascending=False
    )
    return long.head(n)[["column", "column_b", "correlation"]].to_dict("records")


def compute_categorical_distributions(
    df: pd.DataFrame, columns: List[str], top_n: int = 10
) -> pd.DataFrame:
    """Long-format table: one row per (column, value), with count/pct. Values
    beyond top_n per column are bucketed into a single 'other' row."""
    rows = []
    for c in columns:
        if c not in df.columns:
            continue
        counts = df[c].astype("string").fillna("(missing)").value_counts()
        total = int(counts.sum())
        top = counts.head(top_n)
        for value, count in top.items():
            rows.append({
                "column": c,
                "value": value,
                "count": int(count),
                "pct": round(count / total * 100, 2) if total else 0.0,
            })
        remainder = counts.iloc[top_n:]
        if len(remainder):
            rem_count = int(remainder.sum())
            rows.append({
                "column": c,
                "value": "(other)",
                "count": rem_count,
                "pct": round(rem_count / total * 100, 2) if total else 0.0,
            })
    return pd.DataFrame(rows)


def compute_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        n_missing = int(df[c].isna().sum())
        rows.append({
            "column": c,
            "n_missing": n_missing,
            "pct_missing": round(n_missing / len(df) * 100, 2) if len(df) else 0.0,
        })
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False).reset_index(drop=True)


def compute_time_series_trend(
    df: pd.DataFrame, date_column: str, value_column: str, freq: str = "W"
) -> pd.DataFrame:
    if date_column not in df.columns or value_column not in df.columns:
        return pd.DataFrame(columns=["period", "order_count", "value_sum", "value_mean"])

    working = df[[date_column, value_column]].dropna(subset=[date_column]).copy()
    if working.empty:
        return pd.DataFrame(columns=["period", "order_count", "value_sum", "value_mean"])

    working["period"] = pd.to_datetime(working[date_column]).dt.to_period(freq).dt.start_time
    grouped = (
        working.groupby("period")
        .agg(order_count=(value_column, "count"), value_sum=(value_column, "sum"), value_mean=(value_column, "mean"))
        .reset_index()
        .sort_values("period")
    )
    grouped["period"] = grouped["period"].astype(str)
    return grouped


def period_over_period_change(trend: pd.DataFrame, value_col: str = "value_sum") -> Optional[Dict]:
    """Simple last-vs-previous-period % change, the kind of number the
    Recommendation Agent will want to phrase as 'Revenue increased 12% this
    period.' Returns None if there aren't at least 2 periods."""
    if len(trend) < 2:
        return None
    last, prev = trend[value_col].iloc[-1], trend[value_col].iloc[-2]
    if prev == 0 or pd.isna(prev):
        return None
    pct_change = round((last - prev) / prev * 100, 2)
    return {
        "latest_period": trend["period"].iloc[-1],
        "previous_period": trend["period"].iloc[-2],
        "latest_value": float(last),
        "previous_value": float(prev),
        "pct_change": pct_change,
    }
