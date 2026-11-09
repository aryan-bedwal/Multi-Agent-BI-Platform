"""Entry point for running the pipeline built so far.

Follows the project's system architecture: Ingestion -> Cleaning -> EDA, then
EDA branches into two parallel paths (ML -> Forecasting, and Anomaly
Detection), which converge at the Explainability Agent, which feeds the
Insight & Recommendation Agent, which feeds the Dashboard Agent — the final
agent, which assembles everything into a runnable Streamlit app.
"""

from src.agents.anomaly_detection_agent import AnomalyDetectionAgent
from src.agents.dashboard_agent import DashboardAgent
from src.agents.data_cleaning_agent import DataCleaningAgent
from src.agents.data_ingestion_agent import DataIngestionAgent
from src.agents.eda_agent import EDAAgent
from src.agents.explainability_agent import ExplainabilityAgent
from src.agents.forecasting_agent import ForecastingAgent
from src.agents.insight_recommendation_agent import InsightRecommendationAgent
from src.agents.ml_agent import MLAgent
from src.agents.report_generation_agent import ReportGenerationAgent


def main() -> None:
    print("=" * 60)
    print("STAGE 1: Data Ingestion")
    print("=" * 60)
    ingestion_manifest = DataIngestionAgent().run()
    print(ingestion_manifest.summary())

    print("\n" + "=" * 60)
    print("STAGE 2: Data Cleaning")
    print("=" * 60)
    cleaning_manifest = DataCleaningAgent().run(
        upstream_manifest_path=f"{ingestion_manifest.output_dir}/manifest.json"
    )
    print(cleaning_manifest.summary())

    print("\n" + "=" * 60)
    print("STAGE 3: Exploratory Data Analysis")
    print("=" * 60)
    eda_manifest = EDAAgent().run(
        upstream_manifest_path=f"{cleaning_manifest.output_dir}/manifest.json"
    )
    print(eda_manifest.summary())

    # --- branch A: ML -> Forecasting ---
    print("\n" + "=" * 60)
    print("STAGE 4A: Machine Learning")
    print("=" * 60)
    ml_manifest = MLAgent().run(
        upstream_manifest_path=f"{eda_manifest.output_dir}/manifest.json"
    )
    print(ml_manifest.summary())

    print("\n" + "=" * 60)
    print("STAGE 5A: Forecasting")
    print("=" * 60)
    forecasting_manifest = ForecastingAgent().run(
        upstream_manifest_path=f"{ml_manifest.output_dir}/manifest.json"
    )
    print(forecasting_manifest.summary())

    # --- branch B: Anomaly Detection (off EDA directly) ---
    print("\n" + "=" * 60)
    print("STAGE 4B: Anomaly Detection")
    print("=" * 60)
    anomaly_manifest = AnomalyDetectionAgent().run(
        upstream_manifest_path=f"{eda_manifest.output_dir}/manifest.json"
    )
    print(anomaly_manifest.summary())

    # --- convergence 1: Explainability reads ML (primary) + Anomaly Detection (secondary) ---
    print("\n" + "=" * 60)
    print("STAGE 6: Explainability")
    print("=" * 60)
    explainability_manifest = ExplainabilityAgent().run(
        upstream_manifest_path=f"{ml_manifest.output_dir}/manifest.json",
        anomaly_manifest_path=f"{anomaly_manifest.output_dir}/manifest.json",
    )
    print(explainability_manifest.summary())

    # --- convergence 2: Insight & Recommendation reads Explainability (primary)
    # + Forecasting + Anomaly Detection (secondary) ---
    print("\n" + "=" * 60)
    print("STAGE 7: Insight & Recommendation")
    print("=" * 60)
    insight_manifest = InsightRecommendationAgent().run(
        upstream_manifest_path=f"{explainability_manifest.output_dir}/manifest.json",
        forecasting_manifest_path=f"{forecasting_manifest.output_dir}/manifest.json",
        anomaly_manifest_path=f"{anomaly_manifest.output_dir}/manifest.json",
    )
    print(insight_manifest.summary())

    # --- final: Dashboard reads Insight & Recommendation (primary)
    # + every other agent by name (secondary, for stage-specific KPIs/charts) ---
    print("\n" + "=" * 60)
    print("STAGE 8: Dashboard")
    print("=" * 60)
    dashboard_manifest = DashboardAgent().run(
        upstream_manifest_path=f"{insight_manifest.output_dir}/manifest.json",
        manifest_paths={
            "eda": f"{eda_manifest.output_dir}/manifest.json",
            "ml": f"{ml_manifest.output_dir}/manifest.json",
            "forecasting": f"{forecasting_manifest.output_dir}/manifest.json",
            "anomaly_detection": f"{anomaly_manifest.output_dir}/manifest.json",
            "explainability": f"{explainability_manifest.output_dir}/manifest.json",
        },
    )
    print(dashboard_manifest.summary())

    # --- final: Report Generation reads Insight & Recommendation (primary)
    # + every other agent by name (secondary) — same contract as Dashboard ---
    print("\n" + "=" * 60)
    print("STAGE 9: Report Generation")
    print("=" * 60)
    report_manifest = ReportGenerationAgent().run(
        upstream_manifest_path=f"{insight_manifest.output_dir}/manifest.json",
        manifest_paths={
            "eda": f"{eda_manifest.output_dir}/manifest.json",
            "ml": f"{ml_manifest.output_dir}/manifest.json",
            "forecasting": f"{forecasting_manifest.output_dir}/manifest.json",
            "anomaly_detection": f"{anomaly_manifest.output_dir}/manifest.json",
            "explainability": f"{explainability_manifest.output_dir}/manifest.json",
        },
    )
    print(report_manifest.summary())

    print("\n" + "=" * 60)
    print(f"Rows before cleaning: {cleaning_manifest.extra_metadata['rows_before']}")
    print(f"Rows after cleaning:  {cleaning_manifest.extra_metadata['rows_after']}")
    print(f"Best ML model: {ml_manifest.extra_metadata['best_model_name']} "
          f"({ml_manifest.extra_metadata['problem_type']})")
    print(f"Best forecasting model: {forecasting_manifest.extra_metadata['best_model_name']}")
    ensemble = anomaly_manifest.extra_metadata["row_level_method_summary"]["ensemble"]
    print(f"Row-level consensus anomalies: {ensemble['n_consensus_anomalies']} "
          f"({ensemble['pct_consensus_anomalies']}%)")
    print(f"Insights generated: {insight_manifest.extra_metadata['n_insights']}")
    print(f"Recommendations generated: {insight_manifest.extra_metadata['n_recommendations']}")
    print(f"Dashboard charts included: {dashboard_manifest.extra_metadata['n_charts_included']}")
    print(f"Report formats generated: {report_manifest.extra_metadata['formats_generated']}")
    print(f"\nDashboard ready — run: streamlit run {dashboard_manifest.extra_metadata['app_root_copy_path']}")
    print(f"Reports saved to: {report_manifest.extra_metadata['report_paths']}")


if __name__ == "__main__":
    main()
