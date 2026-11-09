"""
Shared data contracts for the multi-agent BI platform.

Every agent (Ingestion, Cleaning, EDA, ML, Forecasting, Anomaly Detection,
Explainability, Recommendation, Dashboard) reads a `DatasetManifest` in and
writes a `DatasetManifest` out. This is the ONE thing that has to stay stable
across the whole project so agents built months apart still plug together.

Keep this file dependency-light (pydantic + stdlib only) since every other
module imports it.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"  # ran, but with warnings (e.g. one source table failed)
    FAILED = "failed"


class TableProfile(BaseModel):
    """Profiling metadata for a single table/dataframe."""

    name: str
    n_rows: int
    n_columns: int
    columns: List[str]
    dtypes: Dict[str, str]
    null_counts: Dict[str, int]
    null_pct: Dict[str, float]
    memory_mb: float
    date_range: Optional[Dict[str, Optional[str]]] = None  # {"min": ..., "max": ...}
    file_path: str  # relative path to the parquet file holding this table


class LogEntry(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    level: str = "INFO"
    message: str


class DatasetManifest(BaseModel):
    """
    The contract handed from one agent to the next.

    - `primary_table` is the name of the table downstream agents should treat
      as "the dataset" by default (e.g. the joined fact table).
    - `tables` holds every table the agent produced, including raw/intermediate
      ones other agents might want (e.g. EDA wanting per-table distributions
      even though ML only wants the joined `primary_table`).
    - `output_dir` is where all parquet files for this agent's run live, so a
      later agent can resolve `file_path` entries into real paths.
    """

    agent_name: str
    run_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%dT%H%M%S"))
    status: AgentStatus
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    output_dir: str
    primary_table: str
    tables: Dict[str, TableProfile]
    upstream_manifest: Optional[str] = None  # path to the manifest this agent consumed
    logs: List[LogEntry] = Field(default_factory=list)
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)

    # -- convenience I/O -------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        path = path or str(Path(self.output_dir) / "manifest.json")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w",encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str) -> "DatasetManifest":
        with open(path, "r",encoding="utf-8") as f:
            return cls(**json.load(f))

    def resolve_table_path(self, table_name: str) -> Path:
        """Absolute path to a table's parquet file, resolved against output_dir."""
        profile = self.tables[table_name]
        p = Path(profile.file_path)
        if p.is_absolute():
            return p
        return Path(self.output_dir) / p

    def summary(self) -> str:
        lines = [
            f"[{self.agent_name}] status={self.status.value} run_id={self.run_id}",
            f"  primary_table = {self.primary_table}",
        ]
        for name, t in self.tables.items():
            lines.append(f"  - {name}: {t.n_rows} rows x {t.n_columns} cols")
        return "\n".join(lines)
