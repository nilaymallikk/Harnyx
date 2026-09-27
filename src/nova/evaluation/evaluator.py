"""Evaluators: run an agent over a task batch and aggregate outcomes.

The evaluator is the same-batch measurement device behind the Harness-R1 reward.
It reruns exactly the task identities it was given, before and after a patch, so
the reward is transductive but controlled for task composition.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from nova.core.agent import Agent
from nova.core.harness import Harness
from nova.core.result import EvaluationResult
from nova.core.task import Task
from nova.core.trajectory import TrajectoryRecorder

logger = logging.getLogger(__name__)


@runtime_checkable
class Evaluator(Protocol):
    """Structural interface for batch evaluation."""

    name: str

    def evaluate(
        self,
        agent: Agent,
        tasks: Sequence[Task],
        *,
        harness: Harness | None = None,
        harness_version: str = "harness-v0",
    ) -> EvaluationResult: ...


class LocalEvaluator:
    """Run an agent sequentially over tasks and aggregate rewards/successes."""

    name = "local-evaluator"

    def __init__(self, *, benchmark: str = "", trajectory_dir: str | Path | None = None) -> None:
        self.benchmark = benchmark
        self.trajectory_dir = Path(trajectory_dir) if trajectory_dir is not None else None

    def evaluate(
        self,
        agent: Agent,
        tasks: Sequence[Task],
        *,
        harness: Harness | None = None,
        harness_version: str = "harness-v0",
    ) -> EvaluationResult:
        result = EvaluationResult(
            benchmark=self.benchmark,
            agent_name=getattr(agent, "name", ""),
            harness_version=harness_version,
            metadata={"harness_noop": harness is None or bool(getattr(harness, "is_noop", False))},
        )
        jsonl_path = None
        if self.trajectory_dir is not None:
            self.trajectory_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = self.trajectory_dir / f"{harness_version}.jsonl"

        for task in tasks:
            recorder = TrajectoryRecorder(
                task,
                benchmark=self.benchmark,
                agent_name=getattr(agent, "name", ""),
                harness_version=harness_version,
                jsonl_path=jsonl_path,
            )
            try:
                agent_result = agent.run(task, harness=harness, recorder=recorder)
            except Exception as exc:  # one bad task must not abort the batch
                logger.warning("task %s failed: %s", task.id, exc)
                result.errors[task.id] = f"{type(exc).__name__}: {exc}"
                result.rewards[task.id] = 0.0
                result.successes[task.id] = False
                continue
            result.rewards[task.id] = float(agent_result.reward)
            result.successes[task.id] = bool(agent_result.success)
            result.trajectories[task.id] = agent_result.trajectory
            if agent_result.error:
                result.errors[task.id] = agent_result.error
        return result
