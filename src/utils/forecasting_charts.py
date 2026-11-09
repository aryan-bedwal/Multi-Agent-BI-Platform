"""Chart generation for the Forecasting Agent. Same convention as
eda_charts.py: takes output_dir, writes into output_dir/charts/, returns
paths relative to output_dir."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    output_dir: Path,
    value_label: str,
    best_model_name: str,
) -> List[str]:
    """history: [period, value]. forecast: [period, yhat, yhat_lower, yhat_upper]."""
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(history["period"], history["value"], marker="o", markersize=3, color="#4C72B0", label="Actual")
    ax.plot(forecast["period"], forecast["yhat"], marker="o", markersize=3, color="#C44E52", label=f"Forecast ({best_model_name})")
    if "yhat_lower" in forecast.columns and "yhat_upper" in forecast.columns:
        ax.fill_between(
            forecast["period"], forecast["yhat_lower"], forecast["yhat_upper"],
            color="#C44E52", alpha=0.2, label="Uncertainty band",
        )
    ax.set_title(f"{value_label}: history and forecast")
    ax.set_xlabel("period")
    ax.set_ylabel(value_label)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "forecast.png")
    plt.close(fig)
    return ["charts/forecast.png"]
