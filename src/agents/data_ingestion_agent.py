"""
Data Ingestion Agent
=====================
First agent in the pipeline. Responsibilities (per project spec):
  - Read CSV / Excel / SQL / API sources
  - Output a raw dataset other agents can build on

This implementation targets the Olist multi-table dataset specifically (9
relational CSVs) since that's the real target dataset, while keeping the
CSV/Excel/SQL reading logic generic so a future config swap can point it at
a different source without rewriting the agent.

What it does, end to end:
  1. Load every table declared in config.yaml
  2. Validate each against its required-columns schema; drop (don't crash on)
     tables that fail validation if they're marked optional
  3. Parse declared date columns
  4. Join everything into one order-item-grain `unified` fact table
  5. Profile every table (row/col counts, dtypes, null rates, date ranges)
  6. Persist raw + unified tables as Parquet, and a DatasetManifest describing
     it all, for the next agent (Data Cleaning) to consume
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

from src.agents.base_agent import BaseAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest, TableProfile
from src.utils.profiling import profile_dataframe


class DataIngestionAgent(BaseAgent):
    agent_name = "data_ingestion"

    def __init__(self, config_path: str = "config/config.yaml", output_root: str = "data/processed"):
        super().__init__(output_root=output_root)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["ingestion"]
        self.raw_dir = Path(self.config["raw_data_dir"])

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_table(self, table_name: str, filename: str) -> Optional[pd.DataFrame]:
        path = self.raw_dir / filename
        if not path.exists():
            self.log.warning(f"Missing source file for '{table_name}': {path}")
            return None

        source_type = self.config.get("source_type", "csv")
        try:
            if source_type == "csv":
                df = pd.read_csv(path)
            elif source_type == "excel":
                df = pd.read_excel(path)
            else:
                raise ValueError(f"Unsupported source_type: {source_type}")
        except Exception as exc:  # noqa: BLE001
            self.log.error(f"Failed to read '{table_name}' from {path}: {exc}")
            return None

        n_before = len(df)
        df = df.drop_duplicates()
        if len(df) != n_before:
            self.log.info(f"'{table_name}': dropped {n_before - len(df)} exact duplicate rows on load")

        return df

    def _validate(self, table_name: str, df: pd.DataFrame) -> bool:
        required = self.config.get("required_columns", {}).get(table_name, [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            self.log.error(f"'{table_name}' missing required columns: {missing}")
            return False
        return True

    def _parse_dates(self, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.config.get("date_columns", {}).get(table_name, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    def _load_all_tables(self) -> Dict[str, pd.DataFrame]:
        tables: Dict[str, pd.DataFrame] = {}
        optional = set(self.config.get("optional_tables", []))

        for table_name, filename in self.config["sources"].items():
            df = self._load_table(table_name, filename)

            if df is None:
                if table_name in optional:
                    self.log.warning(f"Optional table '{table_name}' unavailable — continuing without it")
                    continue
                raise RuntimeError(f"Required table '{table_name}' could not be loaded from {filename}")

            if not self._validate(table_name, df):
                if table_name in optional:
                    self.log.warning(f"Optional table '{table_name}' failed validation — dropping it")
                    continue
                raise RuntimeError(f"Required table '{table_name}' failed schema validation")

            df = self._parse_dates(table_name, df)
            tables[table_name] = df
            self.log.info(f"Loaded '{table_name}': {len(df)} rows x {len(df.columns)} cols")

        return tables

    # ------------------------------------------------------------------
    # Joining
    # ------------------------------------------------------------------

    def _build_unified(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Join to an order-item grain fact table. Every join is left/inner as
        documented in config.yaml's `join` section; missing optional tables
        are simply skipped (unified table just won't gain those columns)."""

        if "order_items" not in tables or "orders" not in tables:
            raise RuntimeError(
                "Cannot build unified table: 'order_items' and 'orders' are both required."
            )

        unified = tables["order_items"].merge(
            tables["orders"], on="order_id", how="inner", suffixes=("", "_order")
        )
        self.log.info(f"unified = order_items JOIN orders -> {len(unified)} rows")

        if "products" in tables:
            unified = unified.merge(
                tables["products"], on="product_id", how="left", suffixes=("", "_product")
            )
            self.log.info(f"unified JOIN products -> {len(unified)} rows")

        if "category_translation" in tables:
            unified = unified.merge(
                tables["category_translation"], on="product_category_name", how="left"
            )
            self.log.info("unified JOIN category_translation")

        if "sellers" in tables:
            unified = unified.merge(
                tables["sellers"], on="seller_id", how="left", suffixes=("", "_seller")
            )
            self.log.info(f"unified JOIN sellers -> {len(unified)} rows")

        if "customers" in tables:
            unified = unified.merge(
                tables["customers"], on="customer_id", how="left", suffixes=("", "_customer")
            )
            self.log.info(f"unified JOIN customers -> {len(unified)} rows")

        if "payments" in tables:
            payments_agg = (
                tables["payments"]
                .groupby("order_id")
                .agg(
                    payment_value_total=("payment_value", "sum"),
                    payment_installments_max=("payment_installments", "max"),
                    payment_type_primary=("payment_type", lambda s: s.mode().iat[0] if not s.mode().empty else None),
                    n_payment_rows=("payment_value", "count"),
                )
                .reset_index()
            )
            unified = unified.merge(payments_agg, on="order_id", how="left")
            self.log.info(f"unified JOIN payments (order-grain agg) -> {len(unified)} rows")

        if "reviews" in tables:
            reviews_agg = (
                tables["reviews"]
                .groupby("order_id")
                .agg(review_score_avg=("review_score", "mean"))
                .reset_index()
            )
            unified = unified.merge(reviews_agg, on="order_id", how="left")
            self.log.info(f"unified JOIN reviews (order-grain agg) -> {len(unified)} rows")

        # A couple of derived columns useful to virtually every downstream
        # agent (EDA/ML/Forecasting) — cheap to compute here once.
        if {"order_delivered_customer_date", "order_purchase_timestamp"}.issubset(unified.columns):
            unified["delivery_days"] = (
                unified["order_delivered_customer_date"] - unified["order_purchase_timestamp"]
            ).dt.total_seconds() / 86400.0

        if {"price", "freight_value"}.issubset(unified.columns):
            unified["item_total_value"] = unified["price"] + unified["freight_value"]

        return unified

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self, upstream: Optional[DatasetManifest]) -> DatasetManifest:
        # Ingestion is the first agent — it has no upstream manifest to consume.
        tables = self._load_all_tables()
        unified = self._build_unified(tables)

        raw_dir = self.output_dir / "raw_tables"
        raw_dir.mkdir(parents=True, exist_ok=True)

        profiles: Dict[str, TableProfile] = {}
        for name, df in tables.items():
            rel_path = f"raw_tables/{name}.parquet"
            df.to_parquet(self.output_dir / rel_path, index=False)
            profiles["raw__" + name] = profile_dataframe(name, df, rel_path)

        unified_rel_path = "unified.parquet"
        unified.to_parquet(self.output_dir / unified_rel_path, index=False)
        profiles["unified"] = profile_dataframe("unified", unified, unified_rel_path)

        n_expected = len(self.config["sources"])
        n_loaded = len(tables)
        status = AgentStatus.SUCCESS if n_loaded == n_expected else AgentStatus.PARTIAL
        if status == AgentStatus.PARTIAL:
            self.log.warning(f"Loaded {n_loaded}/{n_expected} declared source tables (see logs above)")

        return DatasetManifest(
            agent_name=self.agent_name,
            status=status,
            output_dir=str(self.output_dir),
            primary_table="unified",
            tables=profiles,
            extra_metadata={
                "n_source_tables_declared": n_expected,
                "n_source_tables_loaded": n_loaded,
                "unified_grain": "order_item",
            },
        )


if __name__ == "__main__":
    agent = DataIngestionAgent()
    manifest = agent.run()
    print("\n" + manifest.summary())
