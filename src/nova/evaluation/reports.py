"""Structured run directories for reproducibility.

Every optimization run produces::

    runs/2026-09-27_001/
      config.json
      baseline.json
      failures.jsonl
      candidates.jsonl
      rewards.jsonl
      accepted_patch.json
      report.json

A reader can reconstruct what failed, what was proposed, whether it validated,
what it scored, and why it was accepted or rejected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nova.core.types import append_jsonl, to_jsonable, write_json


@dataclass(slots=True)
class RunDirectory:
    """Manage a single optimization run directory."""

    root: Path
    run_id: str
    files: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def create(cls, base: str | Path, run_id: str | None = None) -> RunDirectory:
        base_path = Path(base)
        run_id = run_id or time.strftime("%Y-%m-%d_%H%M%S", time.gmtime())
        root = base_path / run_id
        root.mkdir(parents=True, exist_ok=True)
        files = {
            "config": root / "config.json",
            "baseline": root / "baseline.json",
            "failures": root / "failures.jsonl",
            "candidates": root / "candidates.jsonl",
            "rewards": root / "rewards.jsonl",
            "accepted_patch": root / "accepted_patch.json",
            "report": root / "report.json",
        }
        return cls(root=root, run_id=run_id, files=files)

    @classmethod
    def open(cls, root: str | Path) -> RunDirectory:
        root_path = Path(root)
        files = {
            "config": root_path / "config.json",
            "baseline": root_path / "baseline.json",
            "failures": root_path / "failures.jsonl",
            "candidates": root_path / "candidates.jsonl",
            "rewards": root_path / "rewards.jsonl",
            "accepted_patch": root_path / "accepted_patch.json",
            "report": root_path / "report.json",
        }
        return cls(root=root_path, run_id=root_path.name, files=files)

    def path(self, key: str) -> Path:
        return self.files[key]

    def write_config(self, config: Any) -> Path:
        return write_json(self.files["config"], config)

    def write_baseline(self, baseline: Any) -> Path:
        return write_json(self.files["baseline"], baseline)

    def append_failure(self, packet: Any) -> Path:
        return append_jsonl(self.files["failures"], packet)

    def append_candidate(self, record: Any) -> Path:
        return append_jsonl(self.files["candidates"], record)

    def append_reward(self, record: Any) -> Path:
        return append_jsonl(self.files["rewards"], record)

    def write_accepted_patch(self, record: Any) -> Path:
        return write_json(self.files["accepted_patch"], record)

    def write_report(self, report: Any) -> Path:
        return write_json(self.files["report"], report)

    def read_report(self) -> Any:
        from nova.core.types import read_json

        return read_json(self.files["report"])

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable({"run_id": self.run_id, "root": str(self.root), "files": {k: str(v) for k, v in self.files.items()}})
