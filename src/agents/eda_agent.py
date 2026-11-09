"""
EDA (Exploratory Data Analysis) Agent
=======================================
Third agent in the pipeline. Consumes the Data Cleaning Agent's manifest and
runs automated exploratory analysis over its `primary_table` (unified_cleaned).

Responsibilities (per project spec):
  - Histograms
  - Correlation heatmaps
  - Missing-value analysis
  - Category distributions
  - Time-series trends

Design note: unlike Ingestion/Cleaning, this agent's job isn't to transform
the dataset — it's to characterize it. So it does two things other agents
don't:
  1. Produces multiple *summary tables* (numeric_summary, correlation_matrix,
     categorical_distributions, missing_value_summary, time_series_trend) as
     first-class outputs in the manifest — not just charts — so the ML,
     Recommendation, and Dashboard agents can consume EDA findings
     programmatically instead of re-deriving them or parsing images.
  2. Passes the cleaned `unified_cleaned` table through unchanged as
     `primary_table`, so the ML Agent (next) can read *this* agent's
     manifest and get the modeling-ready table without reaching back to the
     Cleaning Agent's output directory.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import yaml

from src.agents.base_agent import BaseAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.eda_charts import (
    plot_categorical_bars,
    plot_correlation_heatmap,
    plot_histograms,
    plot_time_series,
)
from src.utils.eda_ops import (
    compute_categorical_distributions,
    compute_correlation_matrix,
    compute_missing_value_summary,
    compute_numeric_summary,
    compute_time_series_trend,
    period_over_period_change,
    top_correlated_pairs,
)
from src.utils.profiling import profile_dataframe


class EDAAgent(BaseAgent):
    agent_name = "eda"

    def __init__(self, config_path: str = "config/config.yaml", output_root: str = "data/processed"):
        super().__init__(output_root=output_root)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["eda"]

    # ------------------------------------------------------------------
    # Column auto-detection (falls back to config lists if not present)
    # ------------------------------------------------------------------

    def _resolve_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        configured = self.config.get("numeric_columns") or []
        cols = [c for c in configured if c in df.columns]
        if cols:
            return cols
        auto = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        self.log.info(f"No configured numeric columns found in data — auto-detected {len(auto)}")
        return auto

    def _resolve_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        configured = self.config.get("categorical_columns") or []
        cols = [c for c in configured if c in df.columns]
        if cols:
            return cols
        auto = [
            c for c in df.columns
            if pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)
        ]
        self.log.info(f"No configured categorical columns found in data — auto-detected {len(auto)}")
        return auto[:10]

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self, upstream: Optional[DatasetManifest]) -> DatasetManifest:
        if upstream is None:
            raise ValueError(
                "EDAAgent requires an upstream manifest — "
                "run DataCleaningAgent first and pass its manifest.json path."
            )
        if upstream.status == "failed":
            raise ValueError("Cannot run EDA on a failed upstream cleaning run.")

        primary_path = upstream.resolve_table_path(upstream.primary_table)
        df = pd.read_parquet(primary_path)
        self.log.info(f"Loaded upstream primary table '{upstream.primary_table}': {len(df)} rows")

        numeric_cols = self._resolve_numeric_columns(df)
        categorical_cols = self._resolve_categorical_columns(df)
        self.log.info(f"Numeric columns: {numeric_cols}")
        self.log.info(f"Categorical columns: {categorical_cols}")

        # -- summary tables --------------------------------------------
        numeric_summary = compute_numeric_summary(df, numeric_cols)
        self.log.info(f"Computed numeric summary for {len(numeric_summary)} columns")

        corr_method = self.config.get("correlation_method", "pearson")
        correlation_matrix = compute_correlation_matrix(df, numeric_cols, method=corr_method)
        top_corr = top_correlated_pairs(correlation_matrix, n=5)
        for pair in top_corr:
            self.log.info(
                f"Correlation {pair['column']} vs {pair['column_b']}: {pair['correlation']:.3f}"
            )

        top_n = self.config.get("top_n_categories", 10)
        categorical_distributions = compute_categorical_distributions(df, categorical_cols, top_n=top_n)
        self.log.info(f"Computed category distributions for {df[categorical_cols].shape[1] if categorical_cols else 0} columns")

        missing_value_summary = compute_missing_value_summary(df)
        worst_missing = missing_value_summary[missing_value_summary["n_missing"] > 0].head(3)
        for _, row in worst_missing.iterrows():
            self.log.info(f"Missing values: '{row['column']}' — {row['pct_missing']}%")

        ts_cfg = self.config.get("time_series", {})
        time_series_trend = compute_time_series_trend(
            df,
            date_column=ts_cfg.get("date_column", ""),
            value_column=ts_cfg.get("value_column", ""),
            freq=ts_cfg.get("freq", "W"),
        )
        pop_change = period_over_period_change(time_series_trend)
        if pop_change:
            self.log.info(
                f"Latest period {pop_change['latest_period']}: "
                f"{pop_change['pct_change']:+.2f}% vs previous period"
            )

        # -- persist summary tables --------------------------------------
        profiles: Dict[str, object] = {}

        def _save(name: str, table: pd.DataFrame) -> None:
            rel_path = f"{name}.parquet"
            table.to_parquet(self.output_dir / rel_path, index=False)
            profiles[name] = profile_dataframe(name, table, rel_path)

        _save("numeric_summary", numeric_summary)
        _save("correlation_matrix", correlation_matrix)
        _save("categorical_distributions", categorical_distributions)
        _save("missing_value_summary", missing_value_summary)
        _save("time_series_trend", time_series_trend)

        # -- pass the cleaned dataset through unchanged -------------------
        passthrough_rel_path = "unified_passthrough.parquet"
        df.to_parquet(self.output_dir / passthrough_rel_path, index=False)
        profiles[upstream.primary_table] = profile_dataframe(
            upstream.primary_table, df, passthrough_rel_path
        )

        # -- charts -----------------------------------------------------
        chart_paths: List[str] = []
        if self.config.get("generate_charts", True):
            max_hist = self.config.get("max_histogram_columns", 6)
            chart_paths += plot_histograms(df, numeric_cols, self.output_dir, max_columns=max_hist)
            chart_paths += plot_correlation_heatmap(correlation_matrix, self.output_dir)
            chart_paths += plot_categorical_bars(categorical_distributions, self.output_dir)
            chart_paths += plot_time_series(
                time_series_trend, self.output_dir, ts_cfg.get("value_column", "value")
            )
            self.log.info(f"Generated {len(chart_paths)} chart(s) under {self.output_dir}/charts/")

        return DatasetManifest(
            agent_name=self.agent_name,
            status=AgentStatus.SUCCESS,
            output_dir=str(self.output_dir),
            primary_table=upstream.primary_table,
            tables=profiles,
            extra_metadata={
                "numeric_columns_analyzed": numeric_cols,
                "categorical_columns_analyzed": categorical_cols,
                "top_correlated_pairs": top_corr,
                "period_over_period_change": pop_change,
                "chart_paths": chart_paths,
            },
        )


if __name__ == "__main__":
    from pathlib import Path

    from src.agents.data_cleaning_agent import DataCleaningAgent
    from src.agents.data_ingestion_agent import DataIngestionAgent

    cleaning_manifest_path = "data/processed/data_cleaning/manifest.json"
    if not Path(cleaning_manifest_path).exists():
        print("No cleaning manifest found — running Ingestion + Cleaning first.")
        ingestion_manifest = DataIngestionAgent().run()
        cleaning_manifest = DataCleaningAgent().run(
            upstream_manifest_path=f"{ingestion_manifest.output_dir}/manifest.json"
        )
        cleaning_manifest_path = f"{cleaning_manifest.output_dir}/manifest.json"

    agent = EDAAgent()
    manifest = agent.run(upstream_manifest_path=cleaning_manifest_path)
    print("\n" + manifest.summary())
