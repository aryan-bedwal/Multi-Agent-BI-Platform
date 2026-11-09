"""
Data Cleaning Agent
====================
Second agent in the pipeline. Consumes the Data Ingestion Agent's manifest and
cleans its `primary_table` (the `unified` order-item-grain fact table).

Responsibilities (per project spec):
  - Remove duplicates
  - Handle missing values
  - Detect outliers
  - Convert data types
  - Standardize formats

Pipeline order matters and is deliberate:
  1. Deduplicate               (before any transform touches values)
  2. String standardization     (so downstream missing-value/mode logic operates
                                  on already-normalized text, e.g. " SP " and
                                  "sp" shouldn't be treated as different modes)
  3. Missing-value imputation    (fills nulls, including constant fills like
                                  'unknown', while columns are still plain
                                  object/string dtype)
  4. Dtype conversions          (runs LAST for 'category' columns specifically:
                                  converting to category before imputing would
                                  make a constant-fill like 'unknown' invalid,
                                  since it isn't yet a known category)
  5. Outlier handling            (last, on the finalized numeric values)

Raw tables from Ingestion are passed through unchanged into this agent's
output directory (bronze/silver layering) — EDA may still want the raw,
unjoined per-table distributions.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import yaml

from src.agents.base_agent import BaseAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.cleaning_ops import (
    apply_dtype_overrides,
    handle_outliers,
    impute_missing,
    remove_duplicates,
    standardize_strings,
)
from src.utils.profiling import profile_dataframe


class DataCleaningAgent(BaseAgent):
    agent_name = "data_cleaning"

    def __init__(self, config_path: str = "config/config.yaml", output_root: str = "data/processed"):
        super().__init__(output_root=output_root)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["cleaning"]

    # ------------------------------------------------------------------
    # Pass-through of raw tables (bronze -> silver, unchanged)
    # ------------------------------------------------------------------

    def _passthrough_raw_tables(self, upstream: DatasetManifest) -> Dict[str, dict]:
        profiles: Dict[str, dict] = {}
        raw_dir = self.output_dir / "raw_tables"
        raw_dir.mkdir(parents=True, exist_ok=True)

        for name, table_profile in upstream.tables.items():
            if name == upstream.primary_table:
                continue
            src_path = upstream.resolve_table_path(name)
            df = pd.read_parquet(src_path)
            short_name = name.replace("raw__", "")
            rel_path = f"raw_tables/{short_name}.parquet"
            df.to_parquet(self.output_dir / rel_path, index=False)
            profiles[name] = profile_dataframe(short_name, df, rel_path)
            self.log.info(f"Passed through '{name}' unchanged ({len(df)} rows)")

        return profiles

    # ------------------------------------------------------------------
    # Cleaning pipeline for the primary table
    # ------------------------------------------------------------------

    def _clean_primary_table(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        report: Dict[str, dict] = {}
        n_rows_before = len(df)

        df, dedup_report = remove_duplicates(df, subset=self.config.get("dedup_subset"))
        self.log.info(
            f"Dedup: removed {dedup_report['n_removed']} rows "
            f"(subset={dedup_report['subset_used']})"
        )
        report["deduplication"] = dedup_report

        df, string_report = standardize_strings(df, self.config.get("string_standardize", {}))
        self.log.info(f"Standardized string columns: {string_report['columns_standardized']}")
        report["string_standardization"] = string_report

        df, missing_report = impute_missing(df, self.config.get("missing_value_strategy", {}))
        for col, info in missing_report.items():
            self.log.info(
                f"Missing values '{col}': {info['n_missing_before']} -> "
                f"{info.get('n_missing_after', info['n_missing_before'])} "
                f"(strategy={info['strategy']})"
            )
        report["missing_values"] = missing_report

        df, dtype_report = apply_dtype_overrides(df, self.config.get("dtype_overrides", {}))
        if dtype_report["applied"]:
            self.log.info(f"Dtype conversions applied: {dtype_report['applied']}")
        if dtype_report["failed"]:
            self.log.warning(f"Dtype conversions failed: {dtype_report['failed']}")
        report["dtype_conversions"] = dtype_report

        df, outlier_report = handle_outliers(df, self.config.get("outliers", {}))
        for col, info in outlier_report.items():
            self.log.info(
                f"Outliers '{col}': {info['n_outliers']} rows ({info['pct_outliers']}%) "
                f"outside [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}] — action={info['action']}"
            )
        report["outliers"] = outlier_report

        report["n_rows_before"] = n_rows_before
        report["n_rows_after"] = len(df)
        return df, report

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self, upstream: Optional[DatasetManifest]) -> DatasetManifest:
        if upstream is None:
            raise ValueError(
                "DataCleaningAgent requires an upstream manifest — "
                "run DataIngestionAgent first and pass its manifest.json path."
            )
        if upstream.status == "failed":
            raise ValueError("Cannot clean data from a failed upstream ingestion run.")

        raw_profiles = self._passthrough_raw_tables(upstream)

        primary_path = upstream.resolve_table_path(upstream.primary_table)
        raw_primary = pd.read_parquet(primary_path)
        self.log.info(f"Loaded upstream primary table '{upstream.primary_table}': {len(raw_primary)} rows")

        before_profile = profile_dataframe(
            f"{upstream.primary_table}_before_cleaning", raw_primary, ""
        )

        cleaned, cleaning_report = self._clean_primary_table(raw_primary)

        cleaned_rel_path = "unified_cleaned.parquet"
        cleaned.to_parquet(self.output_dir / cleaned_rel_path, index=False)
        after_profile = profile_dataframe("unified_cleaned", cleaned, cleaned_rel_path)

        profiles = {**raw_profiles, "unified_cleaned": after_profile}

        status = AgentStatus.SUCCESS
        if cleaning_report["dtype_conversions"]["failed"]:
            status = AgentStatus.PARTIAL
            self.log.warning("Some dtype conversions failed — status downgraded to PARTIAL")

        return DatasetManifest(
            agent_name=self.agent_name,
            status=status,
            output_dir=str(self.output_dir),
            primary_table="unified_cleaned",
            tables=profiles,
            extra_metadata={
                "rows_before": before_profile.n_rows,
                "rows_after": after_profile.n_rows,
                "rows_removed_total": before_profile.n_rows - after_profile.n_rows,
                "cleaning_report": cleaning_report,
            },
        )


if __name__ == "__main__":
    from src.agents.data_ingestion_agent import DataIngestionAgent

    ingestion_manifest_path = "data/processed/data_ingestion/manifest.json"
    try:
        upstream_manifest = DatasetManifest.load(ingestion_manifest_path)
    except FileNotFoundError:
        print("No ingestion manifest found — running Data Ingestion Agent first.")
        upstream_manifest = DataIngestionAgent().run()
        ingestion_manifest_path = str(
            __import__("pathlib").Path(upstream_manifest.output_dir) / "manifest.json"
        )

    agent = DataCleaningAgent()
    manifest = agent.run(upstream_manifest_path=ingestion_manifest_path)
    print("\n" + manifest.summary())
