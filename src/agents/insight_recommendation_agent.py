"""
Insight & Recommendation Agent
=================================
Eighth agent in the pipeline. Reads the Explainability Agent's manifest as
primary upstream (unified_cleaned + SHAP global importances + anomaly
reasons, if those were computed), and optionally the Forecasting Agent's and
Anomaly Detection Agent's manifests for a fuller picture.

Responsibilities (per project spec):
  - "Instead of only predicting, it recommends actions."
  - Example: Prediction: Sales may drop 14%. Recommendation: Launch discount
    campaign, increase advertising budget, stock fast-moving products.
  - "You can use: Rule-based recommendations initially. An LLM ... to
    generate natural-language recommendations."

This implementation is rule-based — the spec's explicitly-named first
option, not a fallback. See config.yaml's `insight_recommendation` section
for why no live LLM call is made here (no API key available in this build
environment) and why the insight objects are structured so an LLM-phrasing
layer could be added later without touching anything upstream.

Two of the spec's example insight types are deliberately NOT produced,
because faking them would be worse than omitting them:
  - "Customer churn is expected to rise" — this dataset has no churn label
  - "Inventory for Product X will run out in 8 days" — this dataset has no
    stock/inventory level data
Every insight this agent DOES produce is computed from data that actually
exists in the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from src.agents.base_agent import BaseAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.eda_ops import compute_time_series_trend, period_over_period_change
from src.utils.insight_charts import plot_segment_changes
from src.utils.insight_ops import (
    build_executive_summary,
    compute_segment_change,
    generate_forecast_insight,
    generate_ml_driver_insight,
    generate_revenue_trend_insight,
    generate_row_anomaly_insight,
    generate_segment_insights,
    generate_time_series_anomaly_insights,
    map_insight_to_recommendation,
)
from src.utils.profiling import profile_dataframe

_SEGMENT_LABELS = {
    "product_category_name_english": "product category",
    "customer_state": "region",
    "seller_state": "seller region",
    "payment_type_primary": "payment method",
}


class InsightRecommendationAgent(BaseAgent):
    agent_name = "insight_recommendation"

    def __init__(self, config_path: str = "config/config.yaml", output_root: str = "data/processed"):
        super().__init__(output_root=output_root)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["insight_recommendation"]
        self._forecasting_manifest_path: Optional[str] = None
        self._anomaly_manifest_path: Optional[str] = None
        self._segment_change_tables: Dict[str, pd.DataFrame] = {}

    def run(
        self,
        upstream_manifest_path: Optional[str] = None,
        forecasting_manifest_path: Optional[str] = None,
        anomaly_manifest_path: Optional[str] = None,
    ) -> DatasetManifest:
        """upstream_manifest_path is the Explainability Agent's manifest
        (required, same as every other agent's primary upstream). The other
        two are optional — this agent degrades gracefully without them,
        producing fewer insight types rather than failing."""
        self._forecasting_manifest_path = forecasting_manifest_path
        self._anomaly_manifest_path = anomaly_manifest_path
        return super().run(upstream_manifest_path=upstream_manifest_path)

    # ------------------------------------------------------------------
    # Insight generation
    # ------------------------------------------------------------------

    def _generate_insights(self, upstream: DatasetManifest, df: pd.DataFrame) -> List[Dict]:
        insights: List[Dict] = []
        date_col = self.config["date_column"]
        value_col = self.config["value_column"]
        freq = self.config.get("freq", "W")

        # -- overall revenue trend: self-sufficient, computed directly from
        # the passthrough data so this agent doesn't strictly need the EDA
        # manifest for its headline number ---------------------------------
        trend = compute_time_series_trend(df, date_col, value_col, freq=freq)
        pop_change = period_over_period_change(trend, value_col="value_sum")
        revenue_insight = generate_revenue_trend_insight(pop_change)
        if revenue_insight:
            insights.append(revenue_insight)
            self.log.info(f"Insight: {revenue_insight['text']}")

        # -- segment-level swings --------------------------------------------
        recent_periods = self.config.get("recent_periods", 4)
        prior_periods = self.config.get("prior_periods", 4)
        top_n_segments = self.config.get("top_n_segments", 3)
        threshold = self.config.get("significant_pct_change_threshold", 10.0)

        self._segment_change_tables = {}
        for seg_col in self.config.get("segment_columns", []):
            if seg_col not in df.columns:
                self.log.warning(f"Segment column '{seg_col}' not found in data — skipping")
                continue
            seg_label = _SEGMENT_LABELS.get(seg_col, seg_col.replace("_", " "))
            change_df = compute_segment_change(
                df, date_col, value_col, seg_col, freq=freq,
                recent_periods=recent_periods, prior_periods=prior_periods,
            )
            self._segment_change_tables[seg_col] = change_df
            seg_insights = generate_segment_insights(change_df, seg_label, top_n=top_n_segments, threshold=threshold)
            insights.extend(seg_insights)
            for si in seg_insights:
                self.log.info(f"Insight: {si['text']}")

        # -- forecast-based insight (optional secondary manifest) -----------
        if self._forecasting_manifest_path:
            forecasting_manifest = DatasetManifest.load(self._forecasting_manifest_path)
            forecast_future = pd.read_parquet(forecasting_manifest.resolve_table_path("forecast_future"))
            forecast_history = pd.read_parquet(forecasting_manifest.resolve_table_path("forecast_history"))
            last_value = float(forecast_history["value"].iloc[-1]) if not forecast_history.empty else None
            best_model = forecasting_manifest.extra_metadata.get("best_model_name", "model")
            horizon = forecasting_manifest.extra_metadata.get("horizon", len(forecast_future))
            forecast_insight = generate_forecast_insight(forecast_future, last_value, best_model, horizon)
            if forecast_insight:
                insights.append(forecast_insight)
                self.log.info(f"Insight: {forecast_insight['text']}")
        else:
            self.log.info("No forecasting_manifest_path provided — skipping forecast-based insight")

        # -- anomaly-based insights (optional secondary manifest) -----------
        if self._anomaly_manifest_path:
            anomaly_manifest = DatasetManifest.load(self._anomaly_manifest_path)
            anomalous_periods = pd.read_parquet(anomaly_manifest.resolve_table_path("anomalous_periods"))
            ts_insights = generate_time_series_anomaly_insights(anomalous_periods, value_col)
            insights.extend(ts_insights)
            for ti in ts_insights:
                self.log.info(f"Insight: {ti['text']}")

            ensemble = anomaly_manifest.extra_metadata["row_level_method_summary"]["ensemble"]
            reasons_df = (
                pd.read_parquet(upstream.resolve_table_path("anomaly_reasons"))
                if "anomaly_reasons" in upstream.tables else pd.DataFrame()
            )
            row_insight = generate_row_anomaly_insight(
                ensemble["n_consensus_anomalies"], ensemble["pct_consensus_anomalies"], reasons_df
            )
            if row_insight:
                insights.append(row_insight)
                self.log.info(f"Insight: {row_insight['text']}")
        else:
            self.log.info("No anomaly_manifest_path provided — skipping anomaly-based insights")

        # -- ML driver insight (from Explainability's own SHAP output) ------
        if "shap_global_summary" in upstream.tables:
            shap_summary = pd.read_parquet(upstream.resolve_table_path("shap_global_summary"))
            target_column = upstream.extra_metadata.get("explained_target", "the target metric")
            top_n_drivers = self.config.get("top_n_shap_drivers", 5)
            driver_insight = generate_ml_driver_insight(shap_summary, target_column, top_n=top_n_drivers)
            if driver_insight:
                insights.append(driver_insight)
                self.log.info(f"Insight: {driver_insight['text']}")

        return insights

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self, upstream: Optional[DatasetManifest]) -> DatasetManifest:
        if upstream is None:
            raise ValueError(
                "InsightRecommendationAgent requires an upstream manifest (the Explainability "
                "Agent's) — run ExplainabilityAgent first and pass its manifest.json path."
            )
        if upstream.status == "failed":
            raise ValueError("Cannot generate insights from a failed upstream run.")

        df = pd.read_parquet(upstream.resolve_table_path(upstream.primary_table))
        self.log.info(f"Loaded upstream primary table '{upstream.primary_table}': {len(df)} rows")

        insights = self._generate_insights(upstream, df)
        for idx, insight in enumerate(insights):
            insight["insight_id"] = idx

        recommendations = []
        for idx, insight in enumerate(insights):
            rec = map_insight_to_recommendation(insight)
            if rec:
                rec["recommendation_id"] = len(recommendations)
                rec["related_insight_id"] = idx
                recommendations.append(rec)
        self.log.info(f"Generated {len(insights)} insight(s), {len(recommendations)} recommendation(s)")

        executive_summary = build_executive_summary(insights, recommendations, max_items=5)

        # -- persist outputs ---------------------------------------------
        profiles: Dict[str, object] = {}

        def _save_table(name: str, table: pd.DataFrame) -> None:
            rel_path = f"{name}.parquet"
            table.to_parquet(self.output_dir / rel_path, index=False)
            profiles[name] = profile_dataframe(name, table, rel_path)

        _save_table("insights", pd.DataFrame(insights))
        _save_table("recommendations", pd.DataFrame(recommendations))

        non_empty_segments = {c: t for c, t in self._segment_change_tables.items() if not t.empty}
        if non_empty_segments:
            combined = pd.concat(
                [t.assign(segment_type=col) for col, t in non_empty_segments.items()],
                ignore_index=True,
            )
            _save_table("segment_change", combined)

        (self.output_dir / "executive_summary.md").write_text(executive_summary)

        passthrough_rel_path = "unified_passthrough.parquet"
        df.to_parquet(self.output_dir / passthrough_rel_path, index=False)
        profiles[upstream.primary_table] = profile_dataframe(
            upstream.primary_table, df, passthrough_rel_path
        )

        chart_paths: List[str] = []
        if self.config.get("generate_chart", True):
            for seg_col, change_df in self._segment_change_tables.items():
                seg_label = _SEGMENT_LABELS.get(seg_col, seg_col.replace("_", " "))
                chart_paths += plot_segment_changes(change_df, self.output_dir, seg_label)
            self.log.info(f"Generated {len(chart_paths)} chart(s) under {self.output_dir}/charts/")

        status = AgentStatus.SUCCESS if insights else AgentStatus.PARTIAL
        if status == AgentStatus.PARTIAL:
            self.log.warning("No insights met the configured thresholds — status downgraded to PARTIAL")

        return DatasetManifest(
            agent_name=self.agent_name,
            status=status,
            output_dir=str(self.output_dir),
            primary_table=upstream.primary_table,
            tables=profiles,
            extra_metadata={
                "n_insights": len(insights),
                "n_recommendations": len(recommendations),
                "insight_types": [i["type"] for i in insights],
                "executive_summary": executive_summary,
                "forecasting_manifest_used": self._forecasting_manifest_path,
                "anomaly_manifest_used": self._anomaly_manifest_path,
                "chart_paths": chart_paths,
            },
        )


if __name__ == "__main__":
    from src.agents.anomaly_detection_agent import AnomalyDetectionAgent
    from src.agents.data_cleaning_agent import DataCleaningAgent
    from src.agents.data_ingestion_agent import DataIngestionAgent
    from src.agents.eda_agent import EDAAgent
    from src.agents.explainability_agent import ExplainabilityAgent
    from src.agents.forecasting_agent import ForecastingAgent
    from src.agents.ml_agent import MLAgent

    explainability_manifest_path = "data/processed/explainability/manifest.json"
    forecasting_manifest_path = "data/processed/forecasting/manifest.json"
    anomaly_manifest_path = "data/processed/anomaly_detection/manifest.json"

    if not Path(explainability_manifest_path).exists():
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
        explainability_manifest_path = f"{explainability_manifest.output_dir}/manifest.json"
        forecasting_manifest_path = f"{forecasting_manifest.output_dir}/manifest.json"
        anomaly_manifest_path = f"{anomaly_manifest.output_dir}/manifest.json"

    agent = InsightRecommendationAgent()
    manifest = agent.run(
        upstream_manifest_path=explainability_manifest_path,
        forecasting_manifest_path=forecasting_manifest_path,
        anomaly_manifest_path=anomaly_manifest_path,
    )
    print("\n" + manifest.summary())
    print("\n" + manifest.extra_metadata["executive_summary"])
