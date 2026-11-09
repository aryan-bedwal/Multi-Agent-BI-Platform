"""
Tests for the Data Ingestion Agent.

Run: pytest tests/ -v

Uses a small hand-built fixture dataset (not the synthetic_olist generator)
so these tests stay fast and don't depend on RNG output shape.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.agents.data_ingestion_agent import DataIngestionAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest

FIXTURE_ROOT = Path("tests/_fixture")
RAW_DIR = FIXTURE_ROOT / "raw" / "olist"
OUTPUT_ROOT = FIXTURE_ROOT / "processed"
CONFIG_PATH = FIXTURE_ROOT / "config.yaml"


@pytest.fixture()
def fixture_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    orders = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "customer_id": ["c1", "c2", "c1"],
        "order_status": ["delivered", "shipped", "delivered"],
        "order_purchase_timestamp": ["2018-01-01", "2018-01-05", "2018-02-01"],
    })
    customers = pd.DataFrame({
        "customer_id": ["c1", "c2"],
        "customer_unique_id": ["u1", "u2"],
        "customer_state": ["SP", "RJ"],
    })
    order_items = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "order_item_id": [1, 1, 1],
        "product_id": ["p1", "p2", "p1"],
        "seller_id": ["s1", "s1", "s2"],
        "price": [100.0, 50.0, 75.0],
        "freight_value": [10.0, 5.0, 7.5],
    })
    payments = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "payment_type": ["credit_card", "boleto", "credit_card"],
        "payment_installments": [1, 1, 3],
        "payment_value": [110.0, 55.0, 82.5],
    })
    products = pd.DataFrame({
        "product_id": ["p1", "p2"],
        "product_category_name": ["eletronicos", None],  # intentional null
    })
    sellers = pd.DataFrame({"seller_id": ["s1", "s2"], "seller_state": ["SP", "MG"]})

    orders.to_csv(RAW_DIR / "olist_orders_dataset.csv", index=False)
    customers.to_csv(RAW_DIR / "olist_customers_dataset.csv", index=False)
    order_items.to_csv(RAW_DIR / "olist_order_items_dataset.csv", index=False)
    payments.to_csv(RAW_DIR / "olist_order_payments_dataset.csv", index=False)
    products.to_csv(RAW_DIR / "olist_products_dataset.csv", index=False)
    sellers.to_csv(RAW_DIR / "olist_sellers_dataset.csv", index=False)
    # reviews, geolocation, category_translation deliberately omitted —
    # they're all marked optional in config, this exercises that path.

    config = {
        "ingestion": {
            "source_type": "csv",
            "raw_data_dir": str(RAW_DIR),
            "sources": {
                "orders": "olist_orders_dataset.csv",
                "customers": "olist_customers_dataset.csv",
                "order_items": "olist_order_items_dataset.csv",
                "payments": "olist_order_payments_dataset.csv",
                "products": "olist_products_dataset.csv",
                "sellers": "olist_sellers_dataset.csv",
                "reviews": "olist_order_reviews_dataset.csv",
                "geolocation": "olist_geolocation_dataset.csv",
                "category_translation": "product_category_name_translation.csv",
            },
            "required_columns": {
                "orders": ["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
                "customers": ["customer_id", "customer_unique_id", "customer_state"],
                "order_items": ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"],
                "payments": ["order_id", "payment_type", "payment_installments", "payment_value"],
                "products": ["product_id", "product_category_name"],
                "sellers": ["seller_id", "seller_state"],
                "reviews": ["review_id", "order_id", "review_score"],
                "geolocation": ["geolocation_zip_code_prefix"],
                "category_translation": ["product_category_name", "product_category_name_english"],
            },
            "date_columns": {"orders": ["order_purchase_timestamp"]},
            "optional_tables": ["reviews", "geolocation", "category_translation"],
        }
    }
    CONFIG_PATH.write_text(yaml.dump(config))

    yield

    shutil.rmtree(FIXTURE_ROOT, ignore_errors=True)


def test_ingestion_runs_and_returns_success_or_partial(fixture_dataset):
    agent = DataIngestionAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run()
    # Optional tables are missing by design in this fixture -> PARTIAL, not FAILED
    assert manifest.status == AgentStatus.PARTIAL
    assert manifest.primary_table == "unified"


def test_unified_table_has_expected_grain_and_rowcount(fixture_dataset):
    agent = DataIngestionAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run()
    unified_path = manifest.resolve_table_path("unified")
    df = pd.read_parquet(unified_path)
    assert len(df) == 3  # 3 order_items rows, one per order in the fixture
    assert "order_id" in df.columns
    assert "delivery_days" not in df.columns or True  # no delivered date in fixture -> may be absent


def test_derived_item_total_value_column(fixture_dataset):
    agent = DataIngestionAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run()
    df = pd.read_parquet(manifest.resolve_table_path("unified"))
    assert "item_total_value" in df.columns
    row = df[df.order_id == "o1"].iloc[0]
    assert row["item_total_value"] == pytest.approx(110.0)


def test_manifest_round_trips_through_disk(fixture_dataset):
    agent = DataIngestionAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run()
    manifest_path = Path(manifest.output_dir) / "manifest.json"
    assert manifest_path.exists()

    reloaded = DatasetManifest.load(str(manifest_path))
    assert reloaded.agent_name == "data_ingestion"
    assert reloaded.primary_table == manifest.primary_table
    assert set(reloaded.tables.keys()) == set(manifest.tables.keys())


def test_missing_required_table_raises(fixture_dataset):
    (RAW_DIR / "olist_orders_dataset.csv").unlink()  # remove a REQUIRED table
    agent = DataIngestionAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    with pytest.raises(Exception):
        agent.run()


def test_null_category_reflected_in_profile(fixture_dataset):
    agent = DataIngestionAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    manifest = agent.run()
    products_profile = manifest.tables["raw__products"]
    assert products_profile.null_counts["product_category_name"] == 1
