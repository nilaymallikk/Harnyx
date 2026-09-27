"""Adapters for Harness-R1 reference benchmarks (WebShop / ALFWorld / DBBench).

Harnyx does not bundle benchmark runtimes. This adapter lets an external
Harness-R1-style runner (the reference AgentBench runtimes, or any compatible
harness-aware runner) plug into Harnyx's :class:`Evaluator` interface. The runner
is responsible for invoking the Harnyx harness hooks at the four lifecycle points
and reporting the native task reward.

Reference reward mapping (paper Appendix B.4):
* WebShop   -> native continuous environment reward
* ALFWorld  -> binary task success
* DBBench   -> binary task success
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from harnyx.core.agent import Agent
from harnyx.core.harness import Harness
from harnyx.core.result import EvaluationResult
from harnyx.core.task import Task
from harnyx.core.trajectory import Trajectory

REFERENCE_BENCHMARKS = ("webshop", "alfworld", "dbbench")


@dataclass(slots=True)
class BenchmarkRun:
    """Normalized result of one external benchmark episode."""

    task_id: str
    reward: float
    success: bool
    trajectory: Trajectory | None = None
    error: str | None = None

    @classmethod
    def from_raw(cls, task_id: str, raw: Any) -> BenchmarkRun:
        def get(key: str, default: Any = None) -> Any:
            if isinstance(raw, dict):
                return raw.get(key, default)
            return getattr(raw, key, default)

        return cls(
            task_id=task_id,
            reward=float(get("reward", 0.0) or 0.0),
            success=bool(get("success", False)),
            trajectory=get("trajectory", None),
            error=get("error", None),
        )


@runtime_checkable
class BenchmarkRunner(Protocol):
    """A harness-aware external benchmark runner."""

    name: str

    def run(self, task: Task, harness: Harness | None) -> BenchmarkRun:
        """Execute one task with ``harness`` installed and return its outcome."""


class HarnessR1BenchmarkAdapter:
    """Expose an external benchmark runner as a Harnyx :class:`Evaluator`."""

    def __init__(self, runner: BenchmarkRunner | Callable[[Task, Harness | None], Any], *, benchmark: str) -> None:
        self.runner = runner
        self.benchmark = benchmark
        self.name = f"harness-r1-{benchmark}-adapter"

    def evaluate(
        self,
        agent: Agent,
        tasks: Sequence[Task],
        *,
        harness: Harness | None = None,
        harness_version: str = "harness-v0",
    ) -> EvaluationResult:
        del agent  # the external runner owns the target agent
        result = EvaluationResult(
            benchmark=self.benchmark,
            agent_name=getattr(self.runner, "name", "external"),
            harness_version=harness_version,
            metadata={"harness_noop": harness is None or bool(getattr(harness, "is_noop", False))},
        )
        for task in tasks:
            try:
                if hasattr(self.runner, "run"):
                    raw = self.runner.run(task, harness)
                else:
                    raw = self.runner(task, harness)  # type: ignore[operator]
                run = BenchmarkRun.from_raw(task.id, raw)
            except Exception as exc:
                result.rewards[task.id] = 0.0
                result.successes[task.id] = False
                result.errors[task.id] = f"{type(exc).__name__}: {exc}"
                continue
            result.rewards[task.id] = run.reward
            result.successes[task.id] = run.success
            if run.trajectory is not None:
                result.trajectories[task.id] = run.trajectory
            if run.error:
                result.errors[task.id] = run.error
        return result
