"""Chart generation for the Anomaly Detection Agent. Same convention as the
other *_charts.py modules: takes output_dir, writes into output_dir/charts/,
returns paths relative to output_dir."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_row_anomaly_scatter(
    df: pd.DataFrame, x_col: str, y_col: str, consensus_col: str, output_dir: Path
) -> List[str]:
    if x_col not in df.columns or y_col not in df.columns:
        return []
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    normal = df[~df[consensus_col]]
    anomalous = df[df[consensus_col]]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(normal[x_col], normal[y_col], s=10, alpha=0.4, color="#4C72B0", label="Normal")
    ax.scatter(anomalous[x_col], anomalous[y_col], s=25, color="#C44E52", label="Consensus anomaly", edgecolors="black", linewidths=0.5)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Row-level anomalies: {x_col} vs {y_col}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "row_anomaly_scatter.png")
    plt.close(fig)
    return ["charts/row_anomaly_scatter.png"]


def plot_time_series_anomalies(anomalous_periods: pd.DataFrame, output_dir: Path, value_label: str) -> List[str]:
    if anomalous_periods.empty:
        return []
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(anomalous_periods["period"], anomalous_periods["value"], color="#4C72B0", marker="o", markersize=3, label="Actual")
    flagged = anomalous_periods[anomalous_periods["is_anomalous"]]
    drops = flagged[flagged["direction"] == "drop"]
    spikes = flagged[flagged["direction"] == "spike"]
    if not drops.empty:
        ax.scatter(drops["period"], drops["value"], color="#C44E52", s=60, zorder=5, label="Drop", marker="v")
    if not spikes.empty:
        ax.scatter(spikes["period"], spikes["value"], color="#55A868", s=60, zorder=5, label="Spike", marker="^")
    ax.set_title(f"{value_label}: flagged periods")
    ax.set_xlabel("period")
    ax.set_ylabel(value_label)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "time_series_anomalies.png")
    plt.close(fig)
    return ["charts/time_series_anomalies.png"]
