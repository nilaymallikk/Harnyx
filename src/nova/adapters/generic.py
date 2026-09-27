"""Generic adapters for plugging arbitrary agents and environments into NOVA."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

from nova.core.agent import Action, HarnessedAgent, Policy, StepResult
from nova.core.harness import Harness, HookContext
from nova.core.result import AgentResult
from nova.core.task import Task
from nova.core.trajectory import TrajectoryRecorder


class CallableAgent:
    """Wrap a plain function as an :class:`~nova.core.agent.Agent`."""

    def __init__(self, fn: Callable[[Task], AgentResult], *, name: str = "callable-agent") -> None:
        self._fn = fn
        self.name = name

    def run(
        self,
        task: Task,
        harness: Harness | None = None,
        recorder: TrajectoryRecorder | None = None,
    ) -> AgentResult:
        del harness, recorder
        return self._fn(task)


class GenericEnvironment:
    """Build an :class:`~nova.core.agent.Environment` from callables."""

    name = "generic-env"

    def __init__(
        self,
        *,
        reset_fn: Callable[[Task], str],
        step_fn: Callable[[Action], StepResult],
        state_fn: Callable[[], dict[str, Any]] | None = None,
        success_fn: Callable[[], bool] | None = None,
        reward_fn: Callable[[], float] | None = None,
        admissible_fn: Callable[[], list[str]] | None = None,
        predicates_fn: Callable[[], dict[str, Any]] | None = None,
        context_extra_fn: Callable[[], dict[str, Any]] | None = None,
        name: str | None = None,
    ) -> None:
        self._reset_fn = reset_fn
        self._step_fn = step_fn
        self._state_fn = state_fn or (lambda: {})
        self._success_fn = success_fn or (lambda: False)
        self._reward_fn = reward_fn or (lambda: 0.0)
        self._admissible_fn = admissible_fn or (lambda: [])
        self._predicates_fn = predicates_fn or (lambda: {})
        self._context_extra_fn = context_extra_fn or (lambda: {})
        if name:
            self.name = name

    def reset(self, task: Task) -> str:
        return self._reset_fn(task)

    def step(self, action: Action) -> StepResult:
        return self._step_fn(action)

    def state(self) -> dict[str, Any]:
        return self._state_fn()

    def success(self) -> bool:
        return self._success_fn()

    def episode_reward(self) -> float:
        return self._reward_fn()

    def admissible_actions(self) -> list[str]:
        return self._admissible_fn()

    def predicates(self) -> dict[str, Any]:
        return self._predicates_fn()

    def context_extra(self) -> dict[str, Any]:
        return self._context_extra_fn()


def build_agent(
    policy: Policy,
    environment: Any,
    *,
    benchmark: str = "",
    max_steps: int = 20,
    name: str | None = None,
    system_prompt: str | None = None,
) -> HarnessedAgent:
    """Convenience constructor for a :class:`HarnessedAgent`."""
    kwargs: dict[str, Any] = {"benchmark": benchmark, "max_steps": max_steps, "name": name}
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    return HarnessedAgent(policy, environment, **kwargs)


__all__ = ["CallableAgent", "GenericEnvironment", "build_agent", "Action", "HookContext", "MutableMapping", "Sequence", "Mapping"]
