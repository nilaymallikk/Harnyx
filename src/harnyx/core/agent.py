"""Agent interface and the generic Harness-R1 runtime loop.

The key separation Harnyx preserves from Harness-R1: the *frozen policy* proposes
actions, and the *editable harness* only observes the loop and returns
structured effects. The generic :class:`HarnessedAgent` runtime is the base
runtime the engineer edits; it drives the four lifecycle hooks and never lets a
hook execute an environment action directly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from harnyx.core.harness import BaseHarness, Harness, HookContext, HookEffect
from harnyx.core.result import AgentResult
from harnyx.core.task import Task
from harnyx.core.trajectory import TrajectoryRecorder
from harnyx.core.types import to_jsonable

DEFAULT_SYSTEM_PROMPT = "You are an agent that solves tasks by choosing actions."


@dataclass(frozen=True, slots=True)
class Action:
    """A proposed or executed action."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def text(self) -> str:
        return self.raw or self.name


@dataclass(slots=True)
class StepResult:
    """Environment response to a single action."""

    observation: str
    done: bool = False
    reward: float | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@runtime_checkable
class Environment(Protocol):
    """A deterministic-by-default task environment."""

    name: str

    def reset(self, task: Task) -> str:
        """Reset for ``task`` and return the initial observation."""

    def step(self, action: Action) -> StepResult:
        """Execute ``action`` and return the next observation."""

    def state(self) -> dict[str, Any]:
        """Return benchmark runtime state exposed to hooks."""

    def success(self) -> bool:
        """Return whether the episode succeeded."""

    def episode_reward(self) -> float:
        """Return the final environment reward for the episode."""

    def admissible_actions(self) -> list[str]:
        """Return actions currently accepted by the environment."""


@runtime_checkable
class Policy(Protocol):
    """A frozen decision policy (an LLM or any callable actor)."""

    name: str

    def act(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        step: int,
        admissible: Sequence[str],
    ) -> Action:
        """Choose an action given the message history."""


@runtime_checkable
class Agent(Protocol):
    """A runnable agent: frozen policy plus its base runtime."""

    name: str

    def run(
        self,
        task: Task,
        harness: Harness | None = None,
        recorder: TrajectoryRecorder | None = None,
    ) -> AgentResult: ...


ActionParser = Callable[[str], Action]


def default_action_parser(text: str) -> Action:
    """Parse a hook-supplied action string into an :class:`Action`."""
    cleaned = " ".join(str(text or "").split())
    return Action(name=cleaned, arguments={"value": cleaned}, raw=cleaned)


class HarnessedAgent:
    """The generic Harness-R1 runtime loop.

    On each step it calls ``make_pre_hint`` and ``on_before_action`` around the
    policy decision and ``on_post_step`` after the environment responds. A
    ``block_and_prompt`` effect suppresses the pending action and re-prompts the
    policy; ``rewrite_action``/``force_action`` replace it; ``inject_hint`` adds
    a recovery message.
    """

    def __init__(
        self,
        policy: Policy,
        environment: Environment,
        *,
        benchmark: str = "",
        max_steps: int = 20,
        name: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        action_parser: ActionParser | None = None,
        max_blocks_per_step: int = 3,
    ) -> None:
        self.policy = policy
        self.environment = environment
        self.benchmark = benchmark or getattr(environment, "name", "")
        self.max_steps = max_steps
        self.name = name or f"{getattr(policy, 'name', 'policy')}@{getattr(environment, 'name', 'env')}"
        self.system_prompt = system_prompt
        self.action_parser: ActionParser = action_parser or default_action_parser
        self.max_blocks_per_step = max_blocks_per_step

    # ------------------------------------------------------------------ #
    # Context construction
    # ------------------------------------------------------------------ #
    def _context(
        self,
        task: Task,
        observation: str,
        step: int,
        action: Action | None,
        state: Mapping[str, Any],
    ) -> HookContext:
        extra = getattr(self.environment, "context_extra", None)
        predicates = getattr(self.environment, "predicates", None)
        return HookContext(
            benchmark=self.benchmark,
            observation=observation,
            step=step,
            max_step=self.max_steps,
            remaining_steps=max(0, self.max_steps - step),
            task={"id": task.id, "instruction": task.instruction, **dict(task.payload)},
            state=dict(state),
            predicates=dict(predicates()) if callable(predicates) else {},
            action=action.to_dict() if action is not None else None,
            admissible=list(self.environment.admissible_actions()),
            extra=dict(extra()) if callable(extra) else {},
        )

    def _effect_dict(self, effect: HookEffect | None) -> list[dict[str, Any]]:
        return [effect.to_dict()] if effect is not None else []

    # ------------------------------------------------------------------ #
    # Episode
    # ------------------------------------------------------------------ #
    def run(
        self,
        task: Task,
        harness: Harness | None = None,
        recorder: TrajectoryRecorder | None = None,
    ) -> AgentResult:
        """Execute ``task`` under ``harness`` (default: no-op base harness)."""
        active_harness: Harness = harness if harness is not None else BaseHarness()
        harness_version = getattr(active_harness, "version", "harness-v0")
        recorder = recorder or TrajectoryRecorder(
            task,
            benchmark=self.benchmark,
            agent_name=self.name,
            harness_version=harness_version,
        )

        observation = self.environment.reset(task)
        notebook: MutableMapping[str, Any] = {}
        state = dict(self.environment.state())

        init_ctx = self._context(task, observation, 0, None, state)
        init_effect = active_harness.on_init(init_ctx, notebook)

        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        if init_effect is not None and init_effect.skills:
            tips = "\n".join(f"- {skill}" for skill in init_effect.skills)
            messages.append({"role": "system", "content": f"Some tips that may help:\n{tips}"})
        messages.append({"role": "user", "content": f"Task: {task.instruction}\nObservation:\n{observation}"})

        pending_action: Action | None = None
        pending_effects: list[dict[str, Any]] = self._effect_dict(init_effect)
        status = "completed"
        error: str | None = None
        success = False

        for step in range(self.max_steps):
            try:
                action, blocked = self._resolve_action(
                    active_harness, notebook, task, observation, step, messages, pending_action, pending_effects
                )
                if blocked:
                    state = dict(self.environment.state())
                    recorder.record_step(
                        observation=observation,
                        action=None,
                        harness_effects=list(pending_effects),
                        state=state,
                    )
                    pending_effects = []
                    pending_action = None
                    continue

                pending_action = None
                messages.append({"role": "assistant", "content": action.raw or action.name})
                result = self.environment.step(action)
                observation = result.observation
                messages.append({"role": "user", "content": f"Observation: {observation}"})
                state = dict(self.environment.state())

                post_ctx = self._context(task, observation, step, action, state)
                post_effect = active_harness.on_post_step(post_ctx, notebook)
                if post_effect is not None:
                    if post_effect.kind == "inject_hint" and post_effect.message:
                        messages.append({"role": "user", "content": post_effect.message})
                    elif post_effect.kind == "force_action" and post_effect.action:
                        pending_action = self.action_parser(post_effect.action)
                    pending_effects.extend(self._effect_dict(post_effect))

                recorder.record_step(
                    observation=observation,
                    action=action.to_dict(),
                    tool_result=result.observation,
                    harness_effects=list(pending_effects),
                    state=state,
                )
                pending_effects = []
                if result.done:
                    break
            except Exception as exc:  # episode-level failures must not crash the evaluator
                error = f"{type(exc).__name__}: {exc}"
                status = "error"
                break

        if status == "completed":
            try:
                reward = float(self.environment.episode_reward())
            except Exception:
                reward = 0.0
            success = bool(self.environment.success())
            if not success:
                status = "task_limit_reached" if self.environment.admissible_actions() else "failed"
        else:
            reward = 0.0

        trajectory = recorder.finish(
            reward=reward,
            success=success,
            status=status,
            error=error,
            metadata={"benchmark": self.benchmark},
        )
        return AgentResult(
            task_id=task.id,
            success=success,
            reward=reward,
            trajectory=trajectory,
            error=error,
            status=status,
            metadata={"benchmark": self.benchmark},
        )

    def _resolve_action(
        self,
        harness: Harness,
        notebook: MutableMapping[str, Any],
        task: Task,
        observation: str,
        step: int,
        messages: list[dict[str, Any]],
        pending_action: Action | None,
        pending_effects: list[dict[str, Any]],
    ) -> tuple[Action, bool]:
        """Return ``(action, blocked)``.

        When ``blocked`` is True the caller re-prompts without stepping the
        environment, mirroring the reference ``block_and_prompt`` semantics.
        """
        blocks = 0
        while True:
            state = dict(self.environment.state())
            if pending_action is None:
                pre_ctx = self._context(task, observation, step, None, state)
                pre_effect = harness.make_pre_hint(pre_ctx, notebook)
                if pre_effect is not None and pre_effect.message:
                    messages.append({"role": "user", "content": pre_effect.message})
                    pending_effects.extend(self._effect_dict(pre_effect))

                action = self.policy.act(
                    messages,
                    step=step,
                    admissible=list(self.environment.admissible_actions()),
                )
            else:
                action = pending_action

            act_ctx = self._context(task, observation, step, action, state)
            effect = harness.on_before_action(act_ctx, notebook)
            if effect is None:
                return action, False

            pending_effects.append(effect.to_dict())
            if effect.kind == "block_and_prompt":
                if effect.message:
                    messages.append({"role": "user", "content": effect.message})
                blocks += 1
                if blocks >= self.max_blocks_per_step:
                    return action, False
                pending_action = None
                continue
            if effect.kind in {"rewrite_action", "force_action"} and effect.action:
                return self.action_parser(effect.action), False
            return action, False
