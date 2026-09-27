"""Structured, serializable trajectory recording.

NOVA records observable execution information only: observations, proposed and
executed actions, tool results, harness effects, errors, timestamps, and the
terminal result. It never records hidden chain-of-thought. Model-visible text
(e.g. an assistant message) may be stored as ``agent_output`` when it is needed
to reproduce the harness interaction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nova.core.task import Task
from nova.core.types import append_jsonl, to_jsonable


@dataclass(slots=True)
class TrajectoryStep:
    """One environment interaction step."""

    index: int
    observation: str
    action: dict[str, Any] | None = None
    tool_result: str | None = None
    agent_output: str | None = None
    harness_effects: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    timestamp: float = field(default_factory=time.time)
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrajectoryStep:
        return cls(
            index=int(raw.get("index", 0)),
            observation=str(raw.get("observation", "")),
            action=dict(raw["action"]) if raw.get("action") else None,
            tool_result=raw.get("tool_result"),
            agent_output=raw.get("agent_output"),
            harness_effects=list(raw.get("harness_effects", []) or []),
            error=raw.get("error"),
            timestamp=float(raw.get("timestamp", 0.0)),
            state=dict(raw.get("state", {}) or {}),
        )


@dataclass(slots=True)
class Trajectory:
    """A complete episode record for one task."""

    task_id: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_reward: float = 0.0
    success: bool = False
    status: str = "unknown"
    benchmark: str = ""
    agent_name: str = ""
    harness_version: str = "harness-v0"
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = to_jsonable(self)
        data["num_steps"] = len(self.steps)
        if self.ended_at is not None:
            data["duration_s"] = round(self.ended_at - self.started_at, 6)
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Trajectory:
        return cls(
            task_id=str(raw.get("task_id", "")),
            steps=[TrajectoryStep.from_dict(step) for step in raw.get("steps", []) or []],
            final_reward=float(raw.get("final_reward", 0.0)),
            success=bool(raw.get("success", False)),
            status=str(raw.get("status", "unknown")),
            benchmark=str(raw.get("benchmark", "")),
            agent_name=str(raw.get("agent_name", "")),
            harness_version=str(raw.get("harness_version", "harness-v0")),
            error=raw.get("error"),
            started_at=float(raw.get("started_at", 0.0)),
            ended_at=float(raw["ended_at"]) if raw.get("ended_at") is not None else None,
            metadata=dict(raw.get("metadata", {}) or {}),
        )

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def actions(self) -> list[dict[str, Any]]:
        return [step.action for step in self.steps if step.action is not None]

    def observations(self) -> list[str]:
        return [step.observation for step in self.steps]

    def harness_effects(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for step in self.steps:
            out.extend(step.harness_effects)
        return out


class TrajectoryRecorder:
    """Builds a :class:`Trajectory` incrementally and can stream to JSONL."""

    def __init__(
        self,
        task: Task,
        *,
        benchmark: str = "",
        agent_name: str = "",
        harness_version: str = "harness-v0",
        jsonl_path: str | Path | None = None,
    ) -> None:
        self.task = task
        self.benchmark = benchmark
        self.agent_name = agent_name
        self.harness_version = harness_version
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        self._steps: list[TrajectoryStep] = []
        self._started = time.time()

    def record_step(
        self,
        *,
        observation: str,
        action: dict[str, Any] | None = None,
        tool_result: str | None = None,
        agent_output: str | None = None,
        harness_effects: list[dict[str, Any]] | None = None,
        error: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> TrajectoryStep:
        step = TrajectoryStep(
            index=len(self._steps),
            observation=observation,
            action=action,
            tool_result=tool_result,
            agent_output=agent_output,
            harness_effects=list(harness_effects or []),
            error=error,
            state=dict(state or {}),
        )
        self._steps.append(step)
        return step

    def finish(
        self,
        *,
        reward: float,
        success: bool,
        status: str,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Trajectory:
        trajectory = Trajectory(
            task_id=self.task.id,
            steps=self._steps,
            final_reward=float(reward),
            success=bool(success),
            status=status,
            benchmark=self.benchmark,
            agent_name=self.agent_name,
            harness_version=self.harness_version,
            error=error,
            started_at=self._started,
            ended_at=time.time(),
            metadata=dict(metadata or {}),
        )
        if self.jsonl_path is not None:
            append_jsonl(self.jsonl_path, trajectory.to_dict())
        return trajectory
