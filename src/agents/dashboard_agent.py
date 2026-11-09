"""
Dashboard Agent
==================
Ninth agent in the pipeline — called "Visualization Agent" in the
architecture diagram, "Dashboard Agent" in the original spec doc; same role
either way: "creates interactive charts, dashboards and KPIs."

Unlike every prior agent, this one's real output is a RUNNABLE STREAMLIT APP,
not a transformed dataset. It reads the Insight & Recommendation Agent's
manifest as primary upstream, and optionally every other agent's manifest by
name (passed as a dict) for stage-specific KPIs/charts/tables. It assembles
everything into one JSON bundle the generated app loads at runtime, and
passes the dataset through unchanged as `primary_table` like every other
agent, for manifest-chain consistency — even though nothing downstream is
expected to consume it yet.

Design choice: the app does no analysis of its own. Every number, table, and
chart it shows was already computed by an upstream agent; the Dashboard
Agent's job is only to assemble and the Streamlit app's job is only to
render. This keeps real computation out of Streamlit's rerun cycle (which
re-executes the whole script top-to-bottom on every UI interaction) and
avoids a second, possibly-inconsistent implementation of logic an earlier
agent already got right.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from src.agents.base_agent import BaseAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.bi_bundle import gather_bi_bundle
from src.utils.dashboard_app_template import render_app_source
from src.utils.dashboard_ops import compute_kpis, to_json_records
from src.utils.profiling import profile_dataframe


class DashboardAgent(BaseAgent):
    agent_name = "dashboard"

    def __init__(self, config_path: str = "config/config.yaml", output_root: str = "data/processed"):
        super().__init__(output_root=output_root)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["dashboard"]
        self._manifest_paths: Dict[str, str] = {}

    def run(
        self, upstream_manifest_path: Optional[str] = None, manifest_paths: Optional[Dict[str, str]] = None
    ) -> DatasetManifest:
        """upstream_manifest_path is the Insight & Recommendation Agent's
        manifest (required primary). `manifest_paths` is an optional dict of
        {agent_name: manifest_path} for any of: eda, ml, forecasting,
        anomaly_detection, explainability — used for stage-specific KPIs,
        tables, and chart images. Missing entries are skipped gracefully,
        same degrade-gracefully pattern as the Explainability and Insight &
        Recommendation Agents' optional secondary manifests."""
        self._manifest_paths = manifest_paths or {}
        return super().run(upstream_manifest_path=upstream_manifest_path)

    # ------------------------------------------------------------------
    # Bundle assembly (JSON-serializable, for the generated Streamlit app)
    # ------------------------------------------------------------------

    def _build_bundle(self, upstream: DatasetManifest, df: pd.DataFrame) -> dict:
        kpi_date_col = self.config.get("kpi_date_column", "order_purchase_timestamp")
        kpi_value_col = self.config.get("kpi_value_column", "item_total_value")
        kpis = compute_kpis(df, kpi_value_col, kpi_date_col)
        self.log.info(f"Computed KPIs: {kpis}")

        bi = gather_bi_bundle(upstream, self._manifest_paths, log=self.log)

        bundle: dict = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_rows": len(df),
            "kpis": kpis,
            "source_manifests": bi.source_manifests,
            "chart_paths": bi.chart_paths,
            "executive_summary": bi.executive_summary,
        }
        for table_name, table_df in bi.tables.items():
            bundle[table_name] = to_json_records(table_df)
        if bi.anomaly_ensemble_summary is not None:
            bundle["anomaly_ensemble_summary"] = bi.anomaly_ensemble_summary
        if bi.ml_model_summary is not None:
            bundle["ml_model_summary"] = bi.ml_model_summary
        if bi.forecasting_summary is not None:
            bundle["forecasting_summary"] = bi.forecasting_summary

        return bundle

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self, upstream: Optional[DatasetManifest]) -> DatasetManifest:
        if upstream is None:
            raise ValueError(
                "DashboardAgent requires an upstream manifest (the Insight & Recommendation "
                "Agent's) — run InsightRecommendationAgent first and pass its manifest.json path."
            )
        if upstream.status == "failed":
            raise ValueError("Cannot build a dashboard from a failed upstream run.")

        df = pd.read_parquet(upstream.resolve_table_path(upstream.primary_table))
        self.log.info(f"Loaded upstream primary table '{upstream.primary_table}': {len(df)} rows")

        bundle = self._build_bundle(upstream, df)

        bundle_path = self.output_dir / "dashboard_bundle.json"
        bundle_path.write_text(json.dumps(bundle, indent=2))
        self.log.info(f"Wrote dashboard bundle to {bundle_path} ({bundle_path.stat().st_size} bytes)")

        app_source = render_app_source(str(bundle_path))
        manifest_tracked_app_path = self.output_dir / "app.py"
        manifest_tracked_app_path.write_text(app_source, encoding="utf-8")

        root_copy_path = Path(self.config.get("app_root_copy_path", "dashboard_app.py"))
        root_copy_path.write_text(app_source, encoding="utf-8")
        self.log.info(f"Wrote Streamlit app to {manifest_tracked_app_path} and {root_copy_path}")

        # -- pass the dataset through unchanged, same convention as every
        # other agent — even though nothing downstream consumes it yet -------
        passthrough_rel_path = "unified_passthrough.parquet"
        df.to_parquet(self.output_dir / passthrough_rel_path, index=False)
        profiles = {
            upstream.primary_table: profile_dataframe(upstream.primary_table, df, passthrough_rel_path)
        }

        n_charts_included = sum(len(v) for v in bundle["chart_paths"].values())
        n_secondary = len(self._manifest_paths)
        status = AgentStatus.SUCCESS if n_charts_included > 0 else AgentStatus.PARTIAL
        if status == AgentStatus.PARTIAL:
            self.log.warning("No charts were included in the bundle — status downgraded to PARTIAL")

        return DatasetManifest(
            agent_name=self.agent_name,
            status=status,
            output_dir=str(self.output_dir),
            primary_table=upstream.primary_table,
            tables=profiles,
            extra_metadata={
                "bundle_path": str(bundle_path),
                "app_path": str(manifest_tracked_app_path),
                "app_root_copy_path": str(root_copy_path),
                "n_secondary_manifests_included": n_secondary,
                "secondary_manifests_included": list(self._manifest_paths.keys()),
                "n_charts_included": n_charts_included,
                "n_kpis": len(bundle["kpis"]),
                "n_insights": len(bundle.get("insights", [])),
                "n_recommendations": len(bundle.get("recommendations", [])),
            },
        )


if __name__ == "__main__":
    from src.agents.anomaly_detection_agent import AnomalyDetectionAgent
    from src.agents.data_cleaning_agent import DataCleaningAgent
    from src.agents.data_ingestion_agent import DataIngestionAgent
    from src.agents.eda_agent import EDAAgent
    from src.agents.explainability_agent import ExplainabilityAgent
    from src.agents.forecasting_agent import ForecastingAgent
    from src.agents.insight_recommendation_agent import InsightRecommendationAgent
    from src.agents.ml_agent import MLAgent

    insight_manifest_path = "data/processed/insight_recommendation/manifest.json"
    if not Path(insight_manifest_path).exists():
        print("Upstream manifests missing — running the full pipeline first.")
        ingestion_manifest = DataIngestionAgent().run()
        cleaning_manifest = DataCleaningAgent().run(
            upstream_manifest_path=f"{ingestion_manifest.output_dir}/manifest.json"
        )
        eda_manifest = EDAAgent().run(
            upstream_manifest_path=f"{cleaning_manifest.output_dir}/manifest.json"
        )
        ml_manifest = MLAgent().run(
            upstream_manifest_path=f"{eda_manifest.output_dir}/manifest.json"
        )
        forecasting_manifest = ForecastingAgent().run(
            upstream_manifest_path=f"{ml_manifest.output_dir}/manifest.json"
        )
        anomaly_manifest = AnomalyDetectionAgent().run(
            upstream_manifest_path=f"{eda_manifest.output_dir}/manifest.json"
        )
        explainability_manifest = ExplainabilityAgent().run(
            upstream_manifest_path=f"{ml_manifest.output_dir}/manifest.json",
            anomaly_manifest_path=f"{anomaly_manifest.output_dir}/manifest.json",
        )
        insight_manifest = InsightRecommendationAgent().run(
            upstream_manifest_path=f"{explainability_manifest.output_dir}/manifest.json",
            forecasting_manifest_path=f"{forecasting_manifest.output_dir}/manifest.json",
            anomaly_manifest_path=f"{anomaly_manifest.output_dir}/manifest.json",
        )
        insight_manifest_path = f"{insight_manifest.output_dir}/manifest.json"
        manifest_paths = {
            "eda": f"{eda_manifest.output_dir}/manifest.json",
            "ml": f"{ml_manifest.output_dir}/manifest.json",
            "forecasting": f"{forecasting_manifest.output_dir}/manifest.json",
            "anomaly_detection": f"{anomaly_manifest.output_dir}/manifest.json",
            "explainability": f"{explainability_manifest.output_dir}/manifest.json",
        }
    else:
        manifest_paths = {
            "eda": "data/processed/eda/manifest.json",
            "ml": "data/processed/ml/manifest.json",
            "forecasting": "data/processed/forecasting/manifest.json",
            "anomaly_detection": "data/processed/anomaly_detection/manifest.json",
            "explainability": "data/processed/explainability/manifest.json",
        }

    agent = DashboardAgent()
    manifest = agent.run(upstream_manifest_path=insight_manifest_path, manifest_paths=manifest_paths)
    print("\n" + manifest.summary())
    print(f"\nRun: streamlit run {manifest.extra_metadata['app_root_copy_path']}")
