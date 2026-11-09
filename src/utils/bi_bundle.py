"""
Shared "pull everything together" logic for the two output-tier agents that
both need the same underlying data: the Dashboard Agent (renders it as a
Streamlit app) and the Report Generation Agent (renders it as DOCX/PDF/XLSX).

Both agents share an identical input contract — the Insight & Recommendation
Agent's manifest as primary, plus every other agent's manifest optionally by
name — and need mostly the same aggregated data (KPIs' source tables, chart
paths, model/anomaly/forecast summaries). Extracting that pull-logic here
once means both agents can't silently drift into pulling slightly different
data for what's supposed to be the same underlying picture of the pipeline.

Unlike the Dashboard Agent's bundle (which is JSON, for the generated
Streamlit app to load), this returns plain pandas DataFrames — the natural
shape for a report builder to iterate rows/write table cells, and something
the Dashboard Agent can still JSON-serialize itself via to_json_records()
when it needs to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.schemas.data_contracts import DatasetManifest
from src.utils.dashboard_ops import resolve_chart_paths
from src.utils.logger import ManifestLogCollector

# Which tables to pull from which optional secondary manifest, if present,
# and under what bundle key.
_SECONDARY_TABLE_MAP = {
    "forecasting": [("forecast_history", "forecast_history"), ("forecast_future", "forecast_future"),
                    ("forecast_model_comparison", "forecast_model_comparison")],
    "anomaly_detection": [("anomalous_periods", "anomalous_periods")],
    "ml": [("model_comparison", "model_comparison"), ("feature_importance", "feature_importance")],
}


@dataclass
class BIBundle:
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    executive_summary: str = ""
    chart_paths: Dict[str, List[str]] = field(default_factory=dict)
    anomaly_ensemble_summary: Optional[dict] = None
    ml_model_summary: Optional[dict] = None
    forecasting_summary: Optional[dict] = None
    source_manifests: Dict[str, str] = field(default_factory=dict)

    def table(self, name: str) -> pd.DataFrame:
        """Always returns a DataFrame — empty if the table wasn't available,
        so callers don't need an `if name in bundle.tables` check everywhere."""
        return self.tables.get(name, pd.DataFrame())

    def all_chart_paths(self) -> List[str]:
        return [p for paths in self.chart_paths.values() for p in paths]


def gather_bi_bundle(
    upstream: DatasetManifest,
    manifest_paths: Dict[str, str],
    log: Optional[ManifestLogCollector] = None,
) -> BIBundle:
    bundle = BIBundle()
    bundle.source_manifests = {
        "insight_recommendation": upstream.output_dir,
        **{name: str(Path(path).parent) for name, path in manifest_paths.items()},
    }

    # -- from the primary (Insight & Recommendation) manifest ---------------
    for table_name in ("insights", "recommendations", "segment_change"):
        if table_name in upstream.tables:
            bundle.tables[table_name] = pd.read_parquet(upstream.resolve_table_path(table_name))

    exec_summary_path = Path(upstream.output_dir) / "executive_summary.md"
    bundle.executive_summary = exec_summary_path.read_text() if exec_summary_path.exists() else ""
    bundle.chart_paths["insight_recommendation"] = resolve_chart_paths(
        upstream.output_dir, upstream.extra_metadata.get("chart_paths", [])
    )

    # -- from optional secondary manifests -----------------------------------
    for agent_name, path in manifest_paths.items():
        try:
            manifest = DatasetManifest.load(path)
        except FileNotFoundError:
            if log:
                log.warning(f"Could not load '{agent_name}' manifest at {path} — skipping")
            continue

        for table_name, bundle_key in _SECONDARY_TABLE_MAP.get(agent_name, []):
            if table_name in manifest.tables:
                bundle.tables[bundle_key] = pd.read_parquet(manifest.resolve_table_path(table_name))

        bundle.chart_paths[agent_name] = resolve_chart_paths(
            manifest.output_dir, manifest.extra_metadata.get("chart_paths", [])
        )
        if log:
            log.info(f"Included '{agent_name}' manifest: {len(bundle.chart_paths[agent_name])} chart(s)")

        if agent_name == "anomaly_detection":
            bundle.anomaly_ensemble_summary = (
                manifest.extra_metadata.get("row_level_method_summary", {}).get("ensemble")
            )
        if agent_name == "ml":
            bundle.ml_model_summary = {
                "best_model_name": manifest.extra_metadata.get("best_model_name"),
                "problem_type": manifest.extra_metadata.get("problem_type"),
                "best_model_metrics": manifest.extra_metadata.get("best_model_metrics"),
            }
        if agent_name == "forecasting":
            bundle.forecasting_summary = {
                "best_model_name": manifest.extra_metadata.get("best_model_name"),
                "best_model_metrics": manifest.extra_metadata.get("best_model_metrics"),
            }

    return bundle
