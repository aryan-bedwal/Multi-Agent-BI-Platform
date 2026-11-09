"""
Explainability Agent
=======================
Seventh agent in the pipeline, and the convergence point for the two
branches the architecture diagram shows splitting off EDA: ML -> Forecasting
and Anomaly Detection. This agent reads the ML Agent's manifest as its
primary upstream (the trained model to explain) and, optionally, the
Anomaly Detection Agent's manifest as a secondary input (the consensus
flags to explain).

Responsibilities (per project spec):
  - Answer "why did the model predict this?"
  - Uses SHAP, LIME

Two genuinely different explanation problems get two different techniques,
deliberately — see src/utils/explainability_ops.py's module docstring for
the reasoning:
  1. ML model predictions (supervised) — SHAP (TreeExplainer/LinearExplainer,
     picked by model type) for accurate, fast global + per-row attribution,
     and LIME for a smaller set of rows as an independent cross-check using a
     different technique (a local surrogate model rather than game-theoretic
     attribution).
  2. Anomaly Detection consensus flags (unsupervised ensemble vote) — no
     single differentiable output exists for SHAP/LIME to attribute against,
     so these are explained by ranking each flagged row's features by
     z-score deviation from the population instead.

Forecasting model explanations are NOT covered here — Prophet/LSTM don't have
a standard SHAP/LIME formulation the way a tabular sklearn model does, and
forcing one on would produce a plausible-looking but not-actually-principled
result. Prophet's own trend/seasonality decomposition is a more honest way to
explain a forecast, and is a reasonable candidate for a future agent rather
than something bolted on here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

from src.agents.base_agent import BaseAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.explainability_charts import plot_shap_summary
from src.utils.explainability_ops import (
    build_lime_explainer,
    build_shap_explainer,
    compute_anomaly_reasons,
    compute_shap_values,
    explain_row_lime,
    explain_row_shap,
    select_example_rows,
    shap_global_summary,
)
from src.utils.profiling import profile_dataframe


class ExplainabilityAgent(BaseAgent):
    agent_name = "explainability"

    def __init__(self, config_path: str = "config/config.yaml", output_root: str = "data/processed"):
        super().__init__(output_root=output_root)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["explainability"]
        self._anomaly_manifest_path: Optional[str] = None

    def run(self, upstream_manifest_path: Optional[str] = None, anomaly_manifest_path: Optional[str] = None) -> DatasetManifest:
        """Extends BaseAgent.run() with a second, optional upstream manifest —
        the ML Agent's manifest is still the primary `upstream_manifest_path`
        (required, same as every other agent); the Anomaly Detection Agent's
        manifest is an additional input specific to this convergence point."""
        self._anomaly_manifest_path = anomaly_manifest_path
        return super().run(upstream_manifest_path=upstream_manifest_path)

    # ------------------------------------------------------------------
    # ML model explanations (SHAP + LIME)
    # ------------------------------------------------------------------

    def _explain_ml_model(self, upstream: DatasetManifest) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        model_card_path = Path(upstream.output_dir) / "model_card.json"
        if not model_card_path.exists():
            raise FileNotFoundError(
                f"No model_card.json found in {upstream.output_dir} — "
                "is the upstream manifest really from the ML Agent?"
            )
        model_card = json.loads(model_card_path.read_text())

        model_path = Path(upstream.output_dir) / "best_model.joblib"
        pipeline = joblib.load(model_path)
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        self.log.info(f"Explaining '{model_card['best_model_name']}' ({model_card['problem_type']}) "
                       f"predicting '{model_card['target_column']}'")

        df = pd.read_parquet(upstream.resolve_table_path(upstream.primary_table))
        target = model_card["target_column"]
        df_valid = df.dropna(subset=[target]).copy()

        feature_cols = model_card["numeric_features"] + model_card["categorical_features"]
        X_full = df_valid[feature_cols]

        sample_size = min(self.config.get("shap_sample_size", 200), len(X_full))
        random_state = self.config.get("random_state", 42)
        sample_idx = X_full.sample(n=sample_size, random_state=random_state).index
        X_sample = X_full.loc[sample_idx]

        id_cols = [c for c in ["order_id", "order_item_id", "product_id"] if c in df_valid.columns]
        ids_sample = df_valid.loc[sample_idx, id_cols].reset_index(drop=True) if id_cols else None

        X_transformed = np.asarray(preprocessor.transform(X_sample))
        feature_names = list(preprocessor.get_feature_names_out())
        y_pred_sample = model.predict(X_transformed)
        self.log.info(f"Explanation sample: {len(X_sample)} rows, {len(feature_names)} transformed features")

        # -- SHAP: global summary + per-row for a handful of example rows ---
        explainer = build_shap_explainer(model, X_transformed)
        shap_values = compute_shap_values(explainer, X_transformed)
        shap_summary = shap_global_summary(shap_values, feature_names)
        self.log.info(f"Top SHAP features: {shap_summary.head(5).to_dict('records')}")

        n_example = self.config.get("n_example_rows", 9)
        top_n = self.config.get("top_n_reasons", 5)
        example_positions = select_example_rows(y_pred_sample, n_total=n_example, random_state=random_state)

        shap_rows = []
        for pos in example_positions:
            reasons = explain_row_shap(shap_values[pos], feature_names, X_transformed[pos], top_n=top_n)
            order_id = ids_sample.iloc[pos]["order_id"] if ids_sample is not None and "order_id" in ids_sample.columns else None
            for rank, r in enumerate(reasons, start=1):
                shap_rows.append({
                    "row_position": int(pos), "order_id": order_id,
                    "predicted_value": float(y_pred_sample[pos]), "rank": rank, **r,
                })
        shap_row_explanations = pd.DataFrame(shap_rows)

        # -- LIME: independent cross-check on a subset of the same rows -----
        lime_mode = "classification" if model_card["problem_type"] == "classification" else "regression"
        predict_fn = model.predict_proba if lime_mode == "classification" else model.predict
        lime_explainer = build_lime_explainer(X_transformed, feature_names, mode=lime_mode, random_state=random_state)

        n_lime = self.config.get("n_lime_rows", 5)
        lime_positions = example_positions[:n_lime]
        lime_rows = []
        for pos in lime_positions:
            try:
                reasons = explain_row_lime(lime_explainer, predict_fn, X_transformed[pos], top_n=top_n)
            except Exception as exc:  # noqa: BLE001 - LIME can be finicky on edge-case rows
                self.log.warning(f"LIME failed for row position {pos}: {exc}")
                continue
            order_id = ids_sample.iloc[pos]["order_id"] if ids_sample is not None and "order_id" in ids_sample.columns else None
            for rank, r in enumerate(reasons, start=1):
                lime_rows.append({
                    "row_position": int(pos), "order_id": order_id,
                    "predicted_value": float(y_pred_sample[pos]), "rank": rank, **r,
                })
        lime_row_explanations = pd.DataFrame(lime_rows)

        return model_card, shap_summary, shap_row_explanations, lime_row_explanations, feature_names

    # ------------------------------------------------------------------
    # Anomaly consensus explanations (deviation-based)
    # ------------------------------------------------------------------

    def _explain_anomalies(self) -> pd.DataFrame:
        if not self._anomaly_manifest_path:
            self.log.info("No anomaly_manifest_path provided — skipping anomaly explanations")
            return pd.DataFrame()

        anomaly_manifest = DatasetManifest.load(self._anomaly_manifest_path)
        row_anomalies = pd.read_parquet(anomaly_manifest.resolve_table_path("row_anomalies"))

        exclude = {"order_id", "order_item_id", "product_id", "vote_count", "is_anomaly_consensus"}
        numeric_cols = [
            c for c in row_anomalies.columns
            if c not in exclude and not c.endswith("_flag") and not c.endswith("_score")
            and pd.api.types.is_numeric_dtype(row_anomalies[c])
        ]
        consensus_mask = row_anomalies["is_anomaly_consensus"]
        n_consensus = int(consensus_mask.sum())
        self.log.info(f"Explaining {n_consensus} consensus anomalies using features: {numeric_cols}")

        top_n = self.config.get("anomaly_top_n_reasons", 5)
        reasons = compute_anomaly_reasons(row_anomalies, numeric_cols, consensus_mask, top_n=top_n)
        if not reasons.empty and "order_id" in row_anomalies.columns:
            id_lookup = row_anomalies["order_id"].reset_index().rename(columns={"index": "row_index"})
            reasons = reasons.merge(id_lookup, on="row_index", how="left")
        return reasons

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self, upstream: Optional[DatasetManifest]) -> DatasetManifest:
        if upstream is None:
            raise ValueError(
                "ExplainabilityAgent requires an upstream manifest (the ML Agent's) — "
                "run MLAgent first and pass its manifest.json path as upstream_manifest_path."
            )
        if upstream.status == "failed":
            raise ValueError("Cannot explain a model from a failed upstream ML run.")
        if upstream.agent_name != "ml":
            self.log.warning(
                f"Primary upstream manifest is from '{upstream.agent_name}', not 'ml' — "
                "expected the ML Agent's manifest as the primary input"
            )

        model_card, shap_summary, shap_row_explanations, lime_row_explanations, feature_names = self._explain_ml_model(upstream)
        anomaly_reasons = self._explain_anomalies()

        # -- persist outputs ---------------------------------------------
        profiles: Dict[str, object] = {}

        def _save_table(name: str, table: pd.DataFrame) -> None:
            rel_path = f"{name}.parquet"
            table.to_parquet(self.output_dir / rel_path, index=False)
            profiles[name] = profile_dataframe(name, table, rel_path)

        _save_table("shap_global_summary", shap_summary)
        _save_table("shap_row_explanations", shap_row_explanations)
        if not lime_row_explanations.empty:
            _save_table("lime_row_explanations", lime_row_explanations)
        if not anomaly_reasons.empty:
            _save_table("anomaly_reasons", anomaly_reasons)

        passthrough_rel_path = "unified_passthrough.parquet"
        df = pd.read_parquet(upstream.resolve_table_path(upstream.primary_table))
        df.to_parquet(self.output_dir / passthrough_rel_path, index=False)
        profiles[upstream.primary_table] = profile_dataframe(
            upstream.primary_table, df, passthrough_rel_path
        )

        chart_paths: List[str] = []
        if self.config.get("generate_charts", True):
            chart_paths = plot_shap_summary(shap_summary, self.output_dir)
            self.log.info(f"Generated {len(chart_paths)} chart(s) under {self.output_dir}/charts/")

        status = AgentStatus.SUCCESS
        if self._anomaly_manifest_path and anomaly_reasons.empty:
            status = AgentStatus.PARTIAL
            self.log.warning("Anomaly manifest was provided but produced no anomaly reasons — status downgraded")

        return DatasetManifest(
            agent_name=self.agent_name,
            status=status,
            output_dir=str(self.output_dir),
            primary_table=upstream.primary_table,
            tables=profiles,
            extra_metadata={
                "explained_model_name": model_card["best_model_name"],
                "explained_target": model_card["target_column"],
                "explained_problem_type": model_card["problem_type"],
                "n_features_explained": len(feature_names),
                "top_shap_features": shap_summary.head(10).to_dict("records"),
                "n_shap_example_rows": shap_row_explanations["row_position"].nunique() if not shap_row_explanations.empty else 0,
                "n_lime_example_rows": lime_row_explanations["row_position"].nunique() if not lime_row_explanations.empty else 0,
                "anomaly_manifest_used": self._anomaly_manifest_path,
                "n_anomalies_explained": anomaly_reasons["row_index"].nunique() if not anomaly_reasons.empty else 0,
                "chart_paths": chart_paths,
            },
        )


if __name__ == "__main__":
    from src.agents.anomaly_detection_agent import AnomalyDetectionAgent
    from src.agents.data_cleaning_agent import DataCleaningAgent
    from src.agents.data_ingestion_agent import DataIngestionAgent
    from src.agents.eda_agent import EDAAgent
    from src.agents.ml_agent import MLAgent

    ml_manifest_path = "data/processed/ml/manifest.json"
    anomaly_manifest_path = "data/processed/anomaly_detection/manifest.json"
    if not Path(ml_manifest_path).exists() or not Path(anomaly_manifest_path).exists():
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
        anomaly_manifest = AnomalyDetectionAgent().run(
            upstream_manifest_path=f"{eda_manifest.output_dir}/manifest.json"
        )
        ml_manifest_path = f"{ml_manifest.output_dir}/manifest.json"
        anomaly_manifest_path = f"{anomaly_manifest.output_dir}/manifest.json"

    agent = ExplainabilityAgent()
    manifest = agent.run(upstream_manifest_path=ml_manifest_path, anomaly_manifest_path=anomaly_manifest_path)
    print("\n" + manifest.summary())
