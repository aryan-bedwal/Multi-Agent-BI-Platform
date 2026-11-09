"""
Tests for the Data Cleaning Agent.

Builds a small upstream manifest by hand (rather than running the real
Ingestion Agent) so these tests are fast and target cleaning logic
specifically: dedup, dtype conversion, string standardization, missing-value
strategies, and outlier handling.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.agents.data_cleaning_agent import DataCleaningAgent
from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.profiling import profile_dataframe

FIXTURE_ROOT = Path("tests/_fixture_cleaning")
UPSTREAM_DIR = FIXTURE_ROOT / "upstream"
OUTPUT_ROOT = FIXTURE_ROOT / "processed"
CONFIG_PATH = FIXTURE_ROOT / "config.yaml"


@pytest.fixture()
def upstream_manifest():
    UPSTREAM_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "order_id": ["o1", "o1", "o2", "o3", "o4", "o5"],
        "order_item_id": ["1", "1", "1", "1", "1", "1"],  # o1 row duplicated (same key)
        "product_id": ["p1", "p1", "p2", "p3", "p4", "p5"],
        "order_status": [" Delivered", "delivered ", "SHIPPED", "delivered", "delivered", "delivered"],
        "customer_state": ["sp", "sp", " rj", "mg", "mg", "mg"],
        "product_category_name": ["eletronicos", "eletronicos", None, "moveis", "moveis", "moveis"],
        "price": [100.0, 100.0, 50.0, 75.0, 60.0, 100000.0],  # last is a clear price outlier
        "freight_value": [10.0, 10.0, 5.0, 7.5, 6.0, 8.0],
        "delivery_days": [5.0, 5.0, np.nan, 3.0, 4.0, 6.0],  # nan = legitimately not delivered
        "review_score_avg": [4.0, 4.0, np.nan, 5.0, 3.0, 2.0],
    })

    rel_path = "unified.parquet"
    df.to_parquet(UPSTREAM_DIR / rel_path, index=False)
    profile = profile_dataframe("unified", df, rel_path)

    manifest = DatasetManifest(
        agent_name="data_ingestion",
        status=AgentStatus.SUCCESS,
        output_dir=str(UPSTREAM_DIR),
        primary_table="unified",
        tables={"unified": profile},
    )
    manifest_path = manifest.save()

    config = {
        "cleaning": {
            "dedup_subset": ["order_id", "order_item_id", "product_id"],
            "dtype_overrides": {
                "order_id": "string",
                "order_status": "category",
                "customer_state": "category",
                "product_category_name": "category",
            },
            "string_standardize": {
                "order_status": "lower",
                "customer_state": "upper",
            },
            "missing_value_strategy": {
                "delivery_days": "skip",
                "review_score_avg": "flag_only",
                "product_category_name": "constant:unknown",
            },
            "outliers": {
                "price": {"action": "cap", "iqr_multiplier": 1.5},
                "delivery_days": {"action": "flag", "iqr_multiplier": 1.5},
            },
        }
    }
    CONFIG_PATH.write_text(yaml.dump(config))

    yield manifest_path

    shutil.rmtree(FIXTURE_ROOT, ignore_errors=True)


def _run_agent(upstream_manifest_path: str) -> DatasetManifest:
    agent = DataCleaningAgent(config_path=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
    return agent.run(upstream_manifest_path=upstream_manifest_path)


def test_cleaning_requires_upstream_manifest():
    agent = DataCleaningAgent(config_path="config/config.yaml", output_root=str(OUTPUT_ROOT))
    with pytest.raises(ValueError):
        agent.run(upstream_manifest_path=None)


def test_dedup_removes_true_key_duplicate(upstream_manifest):
    manifest = _run_agent(upstream_manifest)
    df = pd.read_parquet(manifest.resolve_table_path("unified_cleaned"))
    # o1/1/p1 appeared twice with identical values -> should collapse to 1 row
    assert len(df[(df.order_id == "o1") & (df.product_id == "p1")]) == 1
    assert manifest.extra_metadata["rows_removed_total"] == 1


def test_string_standardization_normalizes_case_and_whitespace(upstream_manifest):
    manifest = _run_agent(upstream_manifest)
    df = pd.read_parquet(manifest.resolve_table_path("unified_cleaned"))
    assert set(df["order_status"].unique()) == {"delivered", "shipped"}
    assert set(df["customer_state"].unique()) == {"SP", "RJ", "MG"}


def test_missing_value_strategies_respected(upstream_manifest):
    manifest = _run_agent(upstream_manifest)
    df = pd.read_parquet(manifest.resolve_table_path("unified_cleaned"))
    # 'skip' -> delivery_days NaN untouched
    assert df["delivery_days"].isna().sum() == 1
    # 'flag_only' -> review_score_avg NaN untouched, flag column added
    assert "review_score_avg_was_missing" in df.columns
    assert df["review_score_avg_was_missing"].sum() == 1
    # 'constant:unknown' -> product_category_name filled
    assert df["product_category_name"].isna().sum() == 0
    assert "unknown" in df["product_category_name"].astype(str).str.lower().values


def test_outlier_capping_reduces_max_price(upstream_manifest):
    manifest = _run_agent(upstream_manifest)
    df = pd.read_parquet(manifest.resolve_table_path("unified_cleaned"))
    assert df["price"].max() < 100000.0  # the 100000 outlier got capped


def test_outlier_flagging_does_not_alter_values(upstream_manifest):
    manifest = _run_agent(upstream_manifest)
    df = pd.read_parquet(manifest.resolve_table_path("unified_cleaned"))
    assert "delivery_days_is_outlier" in df.columns
    # values themselves unchanged (flag action, not cap). One row of the
    # original 6 was a true duplicate and got removed by dedup already.
    assert sorted(df["delivery_days"].dropna().tolist()) == pytest.approx([3.0, 4.0, 5.0, 6.0])


def test_raw_tables_pass_through_unchanged(upstream_manifest):
    # add a second raw table to upstream manifest to exercise pass-through
    src_manifest = DatasetManifest.load(upstream_manifest)
    extra_df = pd.DataFrame({"seller_id": ["s1", "s2"], "seller_state": ["sp", "rj"]})
    extra_rel = "sellers.parquet"
    extra_df.to_parquet(Path(src_manifest.output_dir) / extra_rel, index=False)
    src_manifest.tables["raw__sellers"] = profile_dataframe("sellers", extra_df, extra_rel)
    src_manifest.save(upstream_manifest)

    manifest = _run_agent(upstream_manifest)
    passthrough_path = manifest.resolve_table_path("raw__sellers")
    df = pd.read_parquet(passthrough_path)
    assert df["seller_state"].tolist() == ["sp", "rj"]  # untouched, not uppercased


def test_manifest_status_success_when_no_conversion_failures(upstream_manifest):
    manifest = _run_agent(upstream_manifest)
    assert manifest.status == AgentStatus.SUCCESS
    assert manifest.upstream_manifest == upstream_manifest
