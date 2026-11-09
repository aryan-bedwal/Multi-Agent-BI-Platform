"""
Chart generation for the EDA Agent. Kept separate from eda_ops.py because
these functions do I/O (write PNG files) rather than returning data — makes
eda_ops.py's pure functions independently testable without a display backend.

Every function takes the agent's `output_dir` (not a charts subdirectory) and
handles the `charts/` subfolder itself, returning paths relative to
`output_dir` — the same convention TableProfile.file_path uses, so chart
paths and table paths resolve the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless — no display available in this environment
import matplotlib.pyplot as plt
import pandas as pd


def _charts_dir(output_dir: Path) -> Path:
    d = output_dir / "charts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_histograms(df: pd.DataFrame, columns: List[str], output_dir: Path, max_columns: int = 6) -> List[str]:
    charts_dir = _charts_dir(output_dir)
    written = []
    cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])][:max_columns]
    for col in cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[col].dropna().plot(kind="hist", bins=30, ax=ax, color="#4C72B0", edgecolor="white")
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        fig.tight_layout()
        rel = f"charts/hist_{col}.png"
        fig.savefig(charts_dir / f"hist_{col}.png")
        plt.close(fig)
        written.append(rel)
    return written


def plot_correlation_heatmap(corr: pd.DataFrame, output_dir: Path) -> List[str]:
    if corr.empty or "column" not in corr.columns:
        return []
    charts_dir = _charts_dir(output_dir)
    matrix = corr.set_index("column")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(charts_dir / "correlation_heatmap.png")
    plt.close(fig)
    return ["charts/correlation_heatmap.png"]


def plot_categorical_bars(dist_table: pd.DataFrame, output_dir: Path) -> List[str]:
    if dist_table.empty:
        return []
    charts_dir = _charts_dir(output_dir)
    written = []
    for col in dist_table["column"].unique():
        sub = dist_table[dist_table["column"] == col].sort_values("count", ascending=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(sub["value"], sub["count"], color="#55A868")
        ax.set_title(f"Top categories: {col}")
        ax.set_xlabel("count")
        fig.tight_layout()
        rel = f"charts/categories_{col}.png"
        fig.savefig(charts_dir / f"categories_{col}.png")
        plt.close(fig)
        written.append(rel)
    return written


def plot_time_series(trend: pd.DataFrame, output_dir: Path, value_column_label: str) -> List[str]:
    if trend.empty:
        return []
    charts_dir = _charts_dir(output_dir)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trend["period"], trend["value_sum"], marker="o", color="#C44E52")
    ax.set_title(f"{value_column_label} over time")
    ax.set_xlabel("period")
    ax.set_ylabel(value_column_label)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout()
    fig.savefig(charts_dir / "time_series_trend.png")
    plt.close(fig)
    return ["charts/time_series_trend.png"]
