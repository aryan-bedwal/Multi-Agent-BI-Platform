"""
Reusable dashboard helpers, same pattern as the other agents' *_ops.py
modules. Two concerns:
  1. Computing top-line KPIs from the passthrough dataset — the numbers a
     dashboard's header row shows before a person even picks a tab.
  2. Making pandas data JSON-safe (Timestamps, NaN, numpy scalar types)
     so it can go into the bundle file the generated Streamlit app reads —
     `json.dumps` chokes on all three without this.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def compute_kpis(df: pd.DataFrame, value_column: str, date_column: str) -> Dict[str, Any]:
    kpis: Dict[str, Any] = {}

    if value_column in df.columns:
        kpis["total_revenue"] = float(df[value_column].sum())
        kpis["avg_order_item_value"] = float(df[value_column].mean()) if len(df) else 0.0

    if "order_id" in df.columns:
        kpis["total_orders"] = int(df["order_id"].nunique())
    kpis["total_order_items"] = int(len(df))

    if "delivery_days" in df.columns:
        valid = df["delivery_days"].dropna()
        kpis["avg_delivery_days"] = float(valid.mean()) if len(valid) else None

    if "review_score_avg" in df.columns:
        valid = df["review_score_avg"].dropna()
        kpis["avg_review_score"] = float(valid.mean()) if len(valid) else None

    for id_col, label in [("customer_unique_id", "unique_customers"), ("customer_id", "unique_customers"),
                           ("product_id", "unique_products"), ("seller_id", "unique_sellers")]:
        if id_col in df.columns and label not in kpis:
            kpis[label] = int(df[id_col].nunique())

    if date_column in df.columns:
        dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
        if len(dates):
            kpis["date_range_start"] = str(dates.min().date())
            kpis["date_range_end"] = str(dates.max().date())

    return kpis


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if isinstance(value, np.ndarray):
        # Parquet round-trips a Python list into a numpy array on read back —
        # convert it back to a plain list, recursing in case elements need
        # the same NaN/Timestamp handling.
        return [_json_safe_value(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass  # pd.isna() itself can raise on some scalar-ish-but-not-quite types
    return value


def to_json_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Converts a DataFrame to a list of plain-Python-typed dicts, safe to
    pass to json.dumps — handles pandas/numpy scalar types and NaN/NaT that
    the standard library's json module doesn't know how to serialize."""
    if df is None or df.empty:
        return []
    records = df.to_dict("records")
    return [{k: _json_safe_value(v) for k, v in record.items()} for record in records]


def resolve_chart_paths(output_dir: str, chart_paths: Optional[List[str]]) -> List[str]:
    """Turns an agent's relative chart_paths (e.g. 'charts/forecast.png')
    into absolute paths the dashboard app can st.image() directly, given
    that agent's own output_dir."""
    if not chart_paths:
        return []
    from pathlib import Path
    return [str(Path(output_dir) / rel) for rel in chart_paths]
