"""Result containers for single-agent runs and batch evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from harnyx.core.trajectory import Trajectory
from harnyx.core.types import to_jsonable


@dataclass(slots=True)
class AgentResult:
    """Outcome of a single ``Agent.run`` call."""

    task_id: str
    success: bool
    reward: float
    trajectory: Trajectory
    error: str | None = None
    status: str = "completed"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(slots=True)
class EvaluationResult:
    """Aggregate result of evaluating one agent+harness on a task batch.

    ``rewards`` and ``successes`` are keyed by ``Task.id`` so baseline and
    patched runs can be compared per task, which is exactly what the Harness-R1
    same-batch reward requires.
    """

    rewards: dict[str, float] = field(default_factory=dict)
    successes: dict[str, bool] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    benchmark: str = ""
    agent_name: str = ""
    harness_version: str = "harness-v0"
    trajectories: dict[str, Trajectory] = field(default_factory=dict, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def harness_noop(self) -> bool:
        """True when the evaluated harness installed no executable intervention."""
        return bool(self.metadata.get("harness_noop", False))

    @property
    def n(self) -> int:
        return len(self.rewards)

    @property
    def mean_reward(self) -> float:
        return fmean(self.rewards.values()) if self.rewards else 0.0

    @property
    def success_rate(self) -> float:
        return (sum(1 for v in self.successes.values() if v) / len(self.successes)) if self.successes else 0.0

    @property
    def num_success(self) -> int:
        return sum(1 for value in self.successes.values() if value)

    @property
    def complete(self) -> bool:
        return bool(self.rewards) and not self.errors

    def to_dict(self) -> dict[str, Any]:
        data = to_jsonable(self)
        data["trajectories"] = {task_id: traj.to_dict() for task_id, traj in self.trajectories.items()}
        data.update(
            {
                "n": self.n,
                "mean_reward": self.mean_reward,
                "success_rate": self.success_rate,
                "num_success": self.num_success,
                "complete": self.complete,
            }
        )
        return data
