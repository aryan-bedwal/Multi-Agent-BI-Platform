"""
Anomaly Detection Agent
==========================
Sixth agent in the pipeline. Per the project's system architecture, this
branches off the EDA Agent's manifest (not the ML/Forecasting chain) and
converges with Forecasting at the Explainability Agent.

Responsibilities (per project spec):
  - Detect unusual events: fraud, sudden revenue drops, unexpected expenses,
    inventory spikes
  - Algorithms: Isolation Forest, Local Outlier Factor, Autoencoder, One-Class SVM

Two distinct anomaly types, handled by two different mechanisms:
  1. Row-level (fraud-like) — an ENSEMBLE of all four named algorithms scores
     every row independently; a row only becomes a "consensus anomaly" if at
     least `min_votes_for_consensus` of them agree. This is deliberately
     conservative — see the Cleaning Agent's outlier-handling philosophy this
     mirrors — any one algorithm alone is too trigger-happy to act on directly.
  2. Time-series (sudden revenue drop/spike) — rolling z-score on the same
     weekly revenue series the EDA/Forecasting Agents already use.

If torch isn't installed, the autoencoder is skipped (ensemble runs on the
remaining 3 methods) and status downgrades to PARTIAL rather than failing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.agents.base_agent import BaseAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.anomaly_charts import plot_row_anomaly_scatter, plot_time_series_anomalies
from src.utils.anomaly_ops import (
    _HAS_TORCH,
    detect_time_series_anomalies,
    ensemble_vote,
    run_autoencoder,
    run_isolation_forest,
    run_lof,
    run_one_class_svm,
)
from src.utils.ml_ops import build_preprocessor
from src.utils.profiling import profile_dataframe


class AnomalyDetectionAgent(BaseAgent):
    agent_name = "anomaly_detection"

    def __init__(self, config_path: str = "config/config.yaml", output_root: str = "data/processed"):
        super().__init__(output_root=output_root)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["anomaly_detection"]

    # ------------------------------------------------------------------
    # Row-level ensemble
    # ------------------------------------------------------------------

    def _run_row_level_ensemble(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, dict]]:
        numeric_cols = [c for c in self.config.get("numeric_columns", []) if c in df.columns]
        categorical_cols = [c for c in self.config.get("categorical_columns", []) if c in df.columns]
        self.log.info(f"Row-level anomaly features — numeric: {numeric_cols}, categorical: {categorical_cols}")

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        X = preprocessor.fit_transform(df[numeric_cols + categorical_cols])
        X = np.asarray(X)

        contamination = self.config.get("contamination", 0.03)
        flags: Dict[str, np.ndarray] = {}
        scores: Dict[str, np.ndarray] = {}
        summary: Dict[str, dict] = {}

        if_cfg = self.config.get("isolation_forest", {})
        flags["isolation_forest"], scores["isolation_forest"] = run_isolation_forest(
            X, contamination=contamination,
            n_estimators=if_cfg.get("n_estimators", 200), random_state=if_cfg.get("random_state", 42),
        )

        lof_cfg = self.config.get("lof", {})
        flags["lof"], scores["lof"] = run_lof(
            X, contamination=contamination, n_neighbors=lof_cfg.get("n_neighbors", 20)
        )

        svm_cfg = self.config.get("one_class_svm", {})
        flags["one_class_svm"], scores["one_class_svm"] = run_one_class_svm(
            X, nu=contamination, kernel=svm_cfg.get("kernel", "rbf"), gamma=svm_cfg.get("gamma", "scale")
        )

        if _HAS_TORCH:
            ae_cfg = self.config.get("autoencoder", {})
            flags["autoencoder"], scores["autoencoder"] = run_autoencoder(
                X, contamination=contamination,
                hidden_size=ae_cfg.get("hidden_size", 16),
                bottleneck_size=ae_cfg.get("bottleneck_size", 4),
                epochs=ae_cfg.get("epochs", 100),
                learning_rate=ae_cfg.get("learning_rate", 0.01),
                random_state=ae_cfg.get("random_state", 42),
            )
        else:
            self.log.warning("torch not installed — skipping autoencoder (pip install torch)")

        for name, flag_arr in flags.items():
            n_flagged = int(flag_arr.sum())
            summary[name] = {
                "n_flagged": n_flagged,
                "pct_flagged": round(n_flagged / len(df) * 100, 2) if len(df) else 0.0,
            }
            self.log.info(f"'{name}' flagged {n_flagged} rows ({summary[name]['pct_flagged']}%)")

        min_votes = self.config.get("min_votes_for_consensus", 2)
        vote_count, consensus = ensemble_vote(flags, min_votes=min_votes)
        n_consensus = int(consensus.sum())
        self.log.info(
            f"Consensus (>= {min_votes}/{len(flags)} methods agree): "
            f"{n_consensus} rows ({round(n_consensus / len(df) * 100, 2) if len(df) else 0}%)"
        )

        result = df[["order_id", "order_item_id", "product_id"] if "order_item_id" in df.columns else []].copy()
        for col in numeric_cols:
            result[col] = df[col].values
        for name in flags:
            result[f"{name}_flag"] = flags[name]
            result[f"{name}_score"] = scores[name]
        result["vote_count"] = vote_count
        result["is_anomaly_consensus"] = consensus

        summary["ensemble"] = {
            "methods_used": list(flags.keys()),
            "min_votes_for_consensus": min_votes,
            "n_consensus_anomalies": n_consensus,
            "pct_consensus_anomalies": round(n_consensus / len(df) * 100, 2) if len(df) else 0.0,
        }
        return result, summary

    # ------------------------------------------------------------------
    # Time-series-level
    # ------------------------------------------------------------------

    def _run_time_series_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        from src.utils.forecasting_ops import build_time_series

        ts_cfg = self.config.get("time_series", {})
        series = build_time_series(
            df, ts_cfg.get("date_column", ""), ts_cfg.get("value_column", ""), freq=ts_cfg.get("freq", "W")
        )
        if series.empty:
            self.log.warning("Time series is empty — skipping time-series anomaly detection")
            return pd.DataFrame(columns=["period", "value", "rolling_mean", "rolling_std", "z_score", "is_anomalous", "direction"])

        result = detect_time_series_anomalies(
            series,
            rolling_window=ts_cfg.get("rolling_window", 6),
            z_threshold=ts_cfg.get("z_threshold", 2.0),
        )
        n_flagged = int(result["is_anomalous"].sum())
        self.log.info(f"Time-series anomalies: {n_flagged} of {len(result)} periods flagged")
        for _, row in result[result["is_anomalous"]].iterrows():
            self.log.info(
                f"  {row['period']}: value={row['value']:.2f}, z={row['z_score']:.2f}, direction={row['direction']}"
            )
        return result

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self, upstream: Optional[DatasetManifest]) -> DatasetManifest:
        if upstream is None:
            raise ValueError(
                "AnomalyDetectionAgent requires an upstream manifest — "
                "run EDAAgent first and pass its manifest.json path."
            )
        if upstream.status == "failed":
            raise ValueError("Cannot run anomaly detection on a failed upstream run.")

        primary_path = upstream.resolve_table_path(upstream.primary_table)
        df = pd.read_parquet(primary_path)
        self.log.info(f"Loaded upstream primary table '{upstream.primary_table}': {len(df)} rows")

        row_anomalies, method_summary = self._run_row_level_ensemble(df)
        anomalous_periods = self._run_time_series_detection(df)

        status = AgentStatus.SUCCESS if _HAS_TORCH else AgentStatus.PARTIAL
        if not _HAS_TORCH:
            self.log.warning("Autoencoder unavailable — ensemble ran on 3/4 configured methods")

        # -- persist outputs ---------------------------------------------
        profiles: Dict[str, object] = {}

        def _save_table(name: str, table: pd.DataFrame) -> None:
            rel_path = f"{name}.parquet"
            table.to_parquet(self.output_dir / rel_path, index=False)
            profiles[name] = profile_dataframe(name, table, rel_path)

        _save_table("row_anomalies", row_anomalies)
        _save_table("anomalous_periods", anomalous_periods)

        method_summary_rows = [
            {"method": name, **stats}
            for name, stats in method_summary.items()
            if name != "ensemble"
        ]
        _save_table("anomaly_method_summary", pd.DataFrame(method_summary_rows))

        passthrough_rel_path = "unified_passthrough.parquet"
        df.to_parquet(self.output_dir / passthrough_rel_path, index=False)
        profiles[upstream.primary_table] = profile_dataframe(
            upstream.primary_table, df, passthrough_rel_path
        )

        chart_paths: List[str] = []
        if self.config.get("generate_charts", True):
            numeric_cols = self.config.get("numeric_columns", [])
            if len(numeric_cols) >= 2:
                chart_paths += plot_row_anomaly_scatter(
                    row_anomalies, numeric_cols[0], numeric_cols[1], "is_anomaly_consensus", self.output_dir
                )
            value_label = self.config.get("time_series", {}).get("value_column", "value")
            chart_paths += plot_time_series_anomalies(anomalous_periods, self.output_dir, value_label)
            self.log.info(f"Generated {len(chart_paths)} chart(s) under {self.output_dir}/charts/")

        return DatasetManifest(
            agent_name=self.agent_name,
            status=status,
            output_dir=str(self.output_dir),
            primary_table=upstream.primary_table,
            tables=profiles,
            extra_metadata={
                "contamination": self.config.get("contamination"),
                "min_votes_for_consensus": self.config.get("min_votes_for_consensus"),
                "row_level_method_summary": method_summary,
                "n_time_series_anomalies": int(anomalous_periods["is_anomalous"].sum()) if not anomalous_periods.empty else 0,
                "chart_paths": chart_paths,
            },
        )


if __name__ == "__main__":
    from src.agents.data_cleaning_agent import DataCleaningAgent
    from src.agents.data_ingestion_agent import DataIngestionAgent
    from src.agents.eda_agent import EDAAgent

    eda_manifest_path = "data/processed/eda/manifest.json"
    if not Path(eda_manifest_path).exists():
        print("No EDA manifest found — running Ingestion + Cleaning + EDA first.")
        ingestion_manifest = DataIngestionAgent().run()
        cleaning_manifest = DataCleaningAgent().run(
            upstream_manifest_path=f"{ingestion_manifest.output_dir}/manifest.json"
        )
        eda_manifest = EDAAgent().run(
            upstream_manifest_path=f"{cleaning_manifest.output_dir}/manifest.json"
        )
        eda_manifest_path = f"{eda_manifest.output_dir}/manifest.json"

    agent = AnomalyDetectionAgent()
    manifest = agent.run(upstream_manifest_path=eda_manifest_path)
    print("\n" + manifest.summary())
