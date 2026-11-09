"""
Report Generation Agent
==========================
Tenth agent in the pipeline. Same input contract as the Dashboard Agent —
reads the Insight & Recommendation Agent's manifest as primary, plus every
other agent's manifest optionally by name — and shares its data-gathering
logic (src/utils/bi_bundle.py) so the report and the dashboard can never
silently disagree about what happened in a given pipeline run.

Responsibilities (per project spec):
  - Automatically generate reports
  - Example: "Business Performance Report — Revenue increased by 8%. Profit
    increased by 11%. ... Recommendations: Increase inventory by 20%.
    Launch campaign in Region B."
  - "Export to PDF or DOCX." (architecture diagram additionally names
    PowerPoint and Excel — this implementation covers DOCX, PDF, and XLSX;
    see config.yaml's report_generation section for why PowerPoint isn't
    built here)

Like the Dashboard Agent, this agent's real output is files, not a
transformed dataset — but it still passes the dataset through unchanged as
`primary_table`, for manifest-chain consistency with every other agent.

Architecture: one shared "content model" (src/utils/report_ops.py's
ReportSection list), built once from the gathered BI data, walked by two
different renderers (DOCX, PDF) — so the two narrative report formats can't
drift into showing different content. XLSX is structurally different (a
workbook of full tables, not a flowing narrative) and is built directly from
the BIBundle's DataFrames instead.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from src.agents.base_agent import BaseAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.bi_bundle import gather_bi_bundle
from src.utils.dashboard_ops import compute_kpis
from src.utils.profiling import profile_dataframe
from src.utils.report_docx_builder import build_docx
from src.utils.report_ops import build_report_sections
from src.utils.report_pdf_builder import build_pdf
from src.utils.report_xlsx_builder import build_xlsx


class ReportGenerationAgent(BaseAgent):
    agent_name = "report_generation"

    def __init__(self, config_path: str = "config/config.yaml", output_root: str = "data/processed"):
        super().__init__(output_root=output_root)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["report_generation"]
        self._manifest_paths: Dict[str, str] = {}

    def run(
        self, upstream_manifest_path: Optional[str] = None, manifest_paths: Optional[Dict[str, str]] = None
    ) -> DatasetManifest:
        """upstream_manifest_path is the Insight & Recommendation Agent's
        manifest (required primary). `manifest_paths` is the same optional
        {agent_name: manifest_path} dict the Dashboard Agent accepts."""
        self._manifest_paths = manifest_paths or {}
        return super().run(upstream_manifest_path=upstream_manifest_path)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self, upstream: Optional[DatasetManifest]) -> DatasetManifest:
        if upstream is None:
            raise ValueError(
                "ReportGenerationAgent requires an upstream manifest (the Insight & Recommendation "
                "Agent's) — run InsightRecommendationAgent first and pass its manifest.json path."
            )
        if upstream.status == "failed":
            raise ValueError("Cannot generate a report from a failed upstream run.")

        df = pd.read_parquet(upstream.resolve_table_path(upstream.primary_table))
        self.log.info(f"Loaded upstream primary table '{upstream.primary_table}': {len(df)} rows")

        kpi_date_col = self.config.get("kpi_date_column", "order_purchase_timestamp")
        kpi_value_col = self.config.get("kpi_value_column", "item_total_value")
        kpis = compute_kpis(df, kpi_value_col, kpi_date_col)
        self.log.info(f"Computed KPIs: {kpis}")

        bundle = gather_bi_bundle(upstream, self._manifest_paths, log=self.log)

        sections = build_report_sections(
            bundle, kpis,
            max_insights=self.config.get("max_insights", 15),
            max_recommendations=self.config.get("max_recommendations", 10),
            max_charts=self.config.get("max_charts", 6),
        )
        self.log.info(f"Built {len(sections)} report section(s)")

        title = self.config.get("report_title", "Business Performance Report")
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        formats = self.config.get("formats", ["docx", "pdf", "xlsx"])

        report_paths: Dict[str, str] = {}

        if "docx" in formats:
            docx_path = build_docx(sections, self.output_dir / "business_report.docx", title, generated_at)
            report_paths["docx"] = str(docx_path)
            self.log.info(f"Wrote DOCX report to {docx_path} ({docx_path.stat().st_size} bytes)")

        if "pdf" in formats:
            pdf_path = build_pdf(sections, self.output_dir / "business_report.pdf", title, generated_at)
            report_paths["pdf"] = str(pdf_path)
            self.log.info(f"Wrote PDF report to {pdf_path} ({pdf_path.stat().st_size} bytes)")

        if "xlsx" in formats:
            xlsx_path = build_xlsx(
                self.output_dir / "business_report.xlsx", kpis, bundle.tables, bundle.executive_summary
            )
            report_paths["xlsx"] = str(xlsx_path)
            self.log.info(f"Wrote XLSX report to {xlsx_path} ({xlsx_path.stat().st_size} bytes)")

        # -- pass the dataset through unchanged, same convention as every
        # other output-tier agent ------------------------------------------
        passthrough_rel_path = "unified_passthrough.parquet"
        df.to_parquet(self.output_dir / passthrough_rel_path, index=False)
        profiles = {
            upstream.primary_table: profile_dataframe(upstream.primary_table, df, passthrough_rel_path)
        }

        status = AgentStatus.SUCCESS if report_paths else AgentStatus.PARTIAL
        if status == AgentStatus.PARTIAL:
            self.log.warning("No report formats were generated — status downgraded to PARTIAL")

        return DatasetManifest(
            agent_name=self.agent_name,
            status=status,
            output_dir=str(self.output_dir),
            primary_table=upstream.primary_table,
            tables=profiles,
            extra_metadata={
                "report_paths": report_paths,
                "formats_generated": list(report_paths.keys()),
                "n_sections": len(sections),
                "n_kpis": len(kpis),
                "n_secondary_manifests_included": len(self._manifest_paths),
                "secondary_manifests_included": list(self._manifest_paths.keys()),
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

    agent = ReportGenerationAgent()
    manifest = agent.run(upstream_manifest_path=insight_manifest_path, manifest_paths=manifest_paths)
    print("\n" + manifest.summary())
    print("\nGenerated reports:", manifest.extra_metadata["report_paths"])
