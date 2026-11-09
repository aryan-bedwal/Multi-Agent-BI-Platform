"""
Machine Learning Agent
========================
Fourth agent in the pipeline. Consumes the EDA Agent's manifest, trains
several candidate models on its `primary_table` (unified_cleaned), and keeps
the best-performing one.

Responsibilities (per project spec):
  - Automatically identify the problem type (regression / classification / clustering)
  - Train and compare several candidate models
  - Select the best-performing one

Business task on this dataset: predict `delivery_days` (regression) — flag
orders likely to be slow before they ship. Configurable via
`config.yaml: ml.target_column`.

Leakage is handled explicitly, not incidentally: `config.yaml: ml.exclude_columns`
documents WHY each column is excluded (identifier / leakage / derived-from-target)
rather than just listing them, because silently-correct leakage handling is the
easiest thing for a later editor of this config to accidentally break.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.agents.base_agent import BaseAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.ml_ops import (
    build_preprocessor,
    detect_problem_type,
    evaluate_classification,
    evaluate_regression,
    extract_feature_importance,
    get_candidate_models,
    pick_best_model,
)
from src.utils.profiling import profile_dataframe


class MLAgent(BaseAgent):
    agent_name = "ml"

    def __init__(self, config_path: str = "config/config.yaml", output_root: str = "data/processed"):
        super().__init__(output_root=output_root)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["ml"]

    # ------------------------------------------------------------------
    # Feature/target preparation
    # ------------------------------------------------------------------

    def _prepare_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
        target = self.config["target_column"]
        if target not in df.columns:
            raise ValueError(f"Configured target_column '{target}' not found in the dataset")

        n_before = len(df)
        df = df.dropna(subset=[target]).copy()
        n_dropped = n_before - len(df)
        if n_dropped:
            self.log.info(
                f"Dropped {n_dropped} rows with missing target '{target}' "
                f"(e.g. orders not yet delivered)"
            )

        exclude = set(self.config.get("exclude_columns", [])) | {target}
        feature_cols = [c for c in df.columns if c not in exclude]

        X = df[feature_cols]
        y = df[target]

        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
        categorical_cols = [
            c for c in feature_cols
            if c not in numeric_cols
            and (pd.api.types.is_object_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(X[c]))
        ]
        unused = [c for c in feature_cols if c not in numeric_cols and c not in categorical_cols]
        if unused:
            self.log.warning(f"Dropping {len(unused)} feature column(s) of unsupported dtype: {unused}")
            X = X.drop(columns=unused)

        self.log.info(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
        self.log.info(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

        return X, y, numeric_cols, categorical_cols

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self, upstream: Optional[DatasetManifest]) -> DatasetManifest:
        if upstream is None:
            raise ValueError(
                "MLAgent requires an upstream manifest — "
                "run EDAAgent first and pass its manifest.json path."
            )
        if upstream.status == "failed":
            raise ValueError("Cannot train models on a failed upstream run.")

        primary_path = upstream.resolve_table_path(upstream.primary_table)
        df = pd.read_parquet(primary_path)
        self.log.info(f"Loaded upstream primary table '{upstream.primary_table}': {len(df)} rows")

        X, y, numeric_cols, categorical_cols = self._prepare_xy(df)

        configured_type = self.config.get("problem_type", "auto")
        max_card = self.config.get("classification_max_cardinality", 20)
        detected_type = detect_problem_type(y, classification_max_cardinality=max_card)
        problem_type = detected_type if configured_type == "auto" else configured_type

        status = AgentStatus.SUCCESS
        if configured_type != "auto" and configured_type != detected_type:
            self.log.warning(
                f"Configured problem_type='{configured_type}' but auto-detection says "
                f"'{detected_type}' — proceeding with configured value, status downgraded to PARTIAL"
            )
            status = AgentStatus.PARTIAL
        self.log.info(f"Problem type: {problem_type} (auto-detected: {detected_type})")

        test_size = self.config.get("test_size", 0.2)
        random_state = self.config.get("random_state", 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.log.info(f"Train/test split: {len(X_train)} / {len(X_test)} rows")

        model_names = self.config.get("candidate_models", {}).get(problem_type, [])
        candidates = get_candidate_models(problem_type, model_names, random_state=random_state)
        if not candidates:
            raise RuntimeError(f"No candidate models available for problem_type='{problem_type}'")

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        evaluate_fn = evaluate_regression if problem_type == "regression" else evaluate_classification

        metrics_by_model: Dict[str, Dict[str, float]] = {}
        fitted_pipelines: Dict[str, Pipeline] = {}

        for name, estimator in candidates.items():
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            metrics = evaluate_fn(y_test.values, y_pred)
            metrics_by_model[name] = metrics
            fitted_pipelines[name] = pipeline
            self.log.info(f"Model '{name}' metrics: {metrics}")

        best_name, best_metrics = pick_best_model(metrics_by_model, problem_type)
        best_pipeline = fitted_pipelines[best_name]
        self.log.info(f"Best model: '{best_name}' with metrics {best_metrics}")

        # -- persist outputs ---------------------------------------------
        profiles: Dict[str, object] = {}

        def _save_table(name: str, table: pd.DataFrame) -> None:
            rel_path = f"{name}.parquet"
            table.to_parquet(self.output_dir / rel_path, index=False)
            profiles[name] = profile_dataframe(name, table, rel_path)

        comparison_rows = [
            {"model": name, **metrics, "is_best": name == best_name}
            for name, metrics in metrics_by_model.items()
        ]
        _save_table("model_comparison", pd.DataFrame(comparison_rows))

        y_pred_best = best_pipeline.predict(X_test)
        predictions_sample = pd.DataFrame({
            "y_true": y_test.reset_index(drop=True),
            "y_pred": y_pred_best,
        }).head(500)
        _save_table("predictions_sample", predictions_sample)

        feature_importance = extract_feature_importance(best_pipeline)
        if not feature_importance.empty:
            _save_table("feature_importance", feature_importance)
            self.log.info(f"Top features: {feature_importance.head(5).to_dict('records')}")

        # pass the dataset through unchanged, same convention as the EDA agent
        passthrough_rel_path = "unified_passthrough.parquet"
        df.to_parquet(self.output_dir / passthrough_rel_path, index=False)
        profiles[upstream.primary_table] = profile_dataframe(
            upstream.primary_table, df, passthrough_rel_path
        )

        model_rel_path = "best_model.joblib"
        joblib.dump(best_pipeline, self.output_dir / model_rel_path)
        self.log.info(f"Saved best model pipeline to {model_rel_path}")

        model_card = {
            "target_column": self.config["target_column"],
            "problem_type": problem_type,
            "best_model_name": best_name,
            "best_model_metrics": best_metrics,
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
            "n_train_rows": len(X_train),
            "n_test_rows": len(X_test),
        }
        (self.output_dir / "model_card.json").write_text(json.dumps(model_card, indent=2))

        return DatasetManifest(
            agent_name=self.agent_name,
            status=status,
            output_dir=str(self.output_dir),
            primary_table=upstream.primary_table,
            tables=profiles,
            extra_metadata={
                "target_column": self.config["target_column"],
                "problem_type": problem_type,
                "problem_type_auto_detected": detected_type,
                "candidate_models_tried": list(candidates.keys()),
                "metrics_by_model": metrics_by_model,
                "best_model_name": best_name,
                "best_model_metrics": best_metrics,
                "model_artifact_path": model_rel_path,
                "model_card_path": "model_card.json",
                "n_train_rows": len(X_train),
                "n_test_rows": len(X_test),
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

    agent = MLAgent()
    manifest = agent.run(upstream_manifest_path=eda_manifest_path)
    print("\n" + manifest.summary())
