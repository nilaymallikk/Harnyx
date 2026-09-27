"""Harness versioning and rollback.

Every accepted harness modification is recorded as ``harness-v{n}`` with its
parent, the patch, the tasks evaluated, baseline/new scores, reward, timestamp,
model, and configuration. Versions enable inspection and rollback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harnyx.core.types import append_jsonl, read_json, read_jsonl, to_jsonable, write_json


@dataclass(slots=True)
class HarnessVersion:
    """A stored, accepted harness revision."""

    version: str
    parent: str | None = None
    patch: dict[str, Any] = field(default_factory=dict)
    tasks: list[str] = field(default_factory=list)
    baseline_score: float = 0.0
    new_score: float = 0.0
    reward: float = 0.0
    timestamp: str = ""
    model: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> HarnessVersion:
        return cls(
            version=str(raw.get("version", "")),
            parent=raw.get("parent"),
            patch=dict(raw.get("patch", {}) or {}),
            tasks=[str(t) for t in raw.get("tasks", []) or []],
            baseline_score=float(raw.get("baseline_score", 0.0)),
            new_score=float(raw.get("new_score", 0.0)),
            reward=float(raw.get("reward", 0.0)),
            timestamp=str(raw.get("timestamp", "")),
            model=str(raw.get("model", "")),
            config=dict(raw.get("config", {}) or {}),
            metadata=dict(raw.get("metadata", {}) or {}),
        )


class HarnessVersionStore:
    """File-backed store for harness versions (``harness-v0`` is the baseline)."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "versions.jsonl"
        self.current_path = self.directory / "current.json"

    def list_versions(self) -> list[str]:
        return [row.get("version", "") for row in read_jsonl(self.index_path)]

    def next_version(self) -> str:
        existing = self.list_versions()
        numbers = []
        for name in existing:
            try:
                numbers.append(int(name.split("-v", 1)[1]))
            except (IndexError, ValueError):
                continue
        # harness-v0 is the implicit base runtime; the first recorded patch is v1.
        return f"harness-v{max(numbers, default=0) + 1}"

    def current(self) -> str:
        if self.current_path.exists():
            return str(read_json(self.current_path).get("version", "harness-v0"))
        return "harness-v0"

    def record(
        self,
        *,
        patch: dict[str, Any] | None = None,
        tasks: list[str] | None = None,
        baseline_score: float = 0.0,
        new_score: float = 0.0,
        reward: float = 0.0,
        model: str = "",
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        parent: str | None = None,
    ) -> HarnessVersion:
        version = HarnessVersion(
            version=self.next_version(),
            parent=parent if parent is not None else self.current(),
            patch=dict(patch or {}),
            tasks=list(tasks or []),
            baseline_score=float(baseline_score),
            new_score=float(new_score),
            reward=float(reward),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            model=model,
            config=dict(config or {}),
            metadata=dict(metadata or {}),
        )
        append_jsonl(self.index_path, version.to_dict())
        write_json(self.directory / f"{version.version}.json", version.to_dict())
        write_json(self.current_path, {"version": version.version})
        return version

    def load(self, version: str) -> HarnessVersion:
        path = self.directory / f"{version}.json"
        if not path.exists():
            raise KeyError(f"unknown harness version: {version}")
        return HarnessVersion.from_dict(read_json(path))

    def latest(self) -> HarnessVersion | None:
        versions = self.list_versions()
        if not versions:
            return None
        return self.load(versions[-1])

    def rollback(self, version: str) -> HarnessVersion:
        """Point ``current`` at ``version`` and return it (no destructive delete)."""
        target = self.load(version)
        write_json(self.current_path, {"version": target.version})
        return target
