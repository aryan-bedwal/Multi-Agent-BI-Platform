"""Chart generation for the Insight & Recommendation Agent. Same convention
as the other *_charts.py modules: takes output_dir, writes into
output_dir/charts/, returns paths relative to output_dir."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_segment_changes(segment_change: pd.DataFrame, output_dir: Path, segment_label: str, top_n: int = 10) -> List[str]:
    if segment_change.empty:
        return []
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    top = segment_change.reindex(segment_change["pct_change"].abs().sort_values(ascending=False).index).head(top_n)
    top = top.sort_values("pct_change")
    colors = ["#C44E52" if v < 0 else "#55A868" for v in top["pct_change"]]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(top))))
    ax.barh(top["segment"], top["pct_change"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("% change (recent vs. prior period)")
    ax.set_title(f"Biggest revenue swings by {segment_label}")
    fig.tight_layout()
    safe_label = segment_label.replace(" ", "_")
    fig.savefig(charts_dir / f"segment_change_{safe_label}.png")
    plt.close(fig)
    return [f"charts/segment_change_{safe_label}.png"]
