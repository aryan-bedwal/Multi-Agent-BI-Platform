"""Chart generation for the Explainability Agent. Same convention as the
other *_charts.py modules: takes output_dir, writes into output_dir/charts/,
returns paths relative to output_dir."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_shap_summary(shap_summary: pd.DataFrame, output_dir: Path, top_n: int = 15) -> List[str]:
    if shap_summary.empty:
        return []
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    top = shap_summary.head(top_n).sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(top))))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Global feature importance (SHAP)")
    fig.tight_layout()
    fig.savefig(charts_dir / "shap_summary.png")
    plt.close(fig)
    return ["charts/shap_summary.png"]
