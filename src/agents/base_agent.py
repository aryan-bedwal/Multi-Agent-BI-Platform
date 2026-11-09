"""
BaseAgent — the contract every agent in the pipeline implements.

Design goals (so this doesn't need to be revisited as agents 2-9 get built):
  1. An agent doesn't need the previous agent's Python objects in memory — it
     can be handed just a manifest *path* and rehydrate everything from disk.
     That means agents can run in separate processes, separate sessions, or
     later get wrapped as LangGraph/CrewAI nodes without any rewrite.
  2. `run()` always returns a DatasetManifest, never a raw DataFrame, so the
     next agent's input shape never changes.
  3. Agents report partial success (AgentStatus.PARTIAL) instead of crashing
     the whole pipeline when e.g. one source table is missing/bad — the
     Business Recommendation Agent later should be able to say "insights are
     based on incomplete data" rather than the whole run dying silently.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.schemas.data_contracts import AgentStatus, DatasetManifest
from src.utils.logger import ManifestLogCollector, get_logger


class BaseAgent(ABC):
    agent_name: str = "base_agent"

    def __init__(self, output_root: str = "data/processed"):
        self.output_root = Path(output_root)
        self.output_dir = self.output_root / self.agent_name
        self.logger = get_logger(self.agent_name)
        self.log = ManifestLogCollector(self.logger)

    @abstractmethod
    def _run(self, upstream: Optional[DatasetManifest]) -> DatasetManifest:
        """Subclasses implement the actual work here and return a manifest
        with status/tables/primary_table filled in. Timing, output_dir, and
        upstream_manifest bookkeeping is handled by run() so subclasses can't
        forget it."""
        raise NotImplementedError

    def run(self, upstream_manifest_path: Optional[str] = None) -> DatasetManifest:
        upstream = (
            DatasetManifest.load(upstream_manifest_path)
            if upstream_manifest_path
            else None
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log.info(f"Starting {self.agent_name} run")
        start = time.time()

        try:
            manifest = self._run(upstream)
        except Exception as exc:  # noqa: BLE001 - top-level agent boundary
            self.log.error(f"{self.agent_name} failed: {exc}")
            manifest = DatasetManifest(
                agent_name=self.agent_name,
                status=AgentStatus.FAILED,
                output_dir=str(self.output_dir),
                primary_table="",
                tables={},
            )
            raise
        finally:
            elapsed = round(time.time() - start, 2)
            self.log.info(f"Finished {self.agent_name} run in {elapsed}s")

        manifest.output_dir = str(self.output_dir)
        manifest.upstream_manifest = upstream_manifest_path
        manifest.logs = self.log.entries
        manifest.extra_metadata.setdefault("elapsed_seconds", elapsed)
        manifest.save()
        self.logger.info(f"Manifest written to {self.output_dir / 'manifest.json'}")
        return manifest
