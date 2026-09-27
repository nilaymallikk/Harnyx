"""Nyvero adapter.

Nyvero is treated as an external agent/runtime. NOVA core must not depend on it,
so this module translates between NOVA's ``Agent``/``Harness`` interfaces and a
small, documented duck-typed Nyvero contract. Nothing here imports Nyvero.

Nyvero adapter contract
-----------------------
A Nyvero *agent* is any object exposing::

    name: str
    def rollout(instruction: str, *, hooks: Mapping[str, Hook], **kwargs) -> NyveroRollout

where ``Hook`` is ``Callable[[Mapping[str, Any]], Mapping[str, Any] | None]`` and
keyed by the four NOVA lifecycle names. ``NyveroRollout`` exposes ``success``
(bool), ``reward`` (float), and ``steps`` (iterable); each step exposes
``observation``, ``action``, ``tool_result``, ``error``, and ``state`` as either
attributes or mapping keys.

A Nyvero *harness* is any object exposing callables ``on_init``, ``make_pre_hint``,
``on_before_action``, and ``on_post_step`` that accept a context mapping and
return a raw effect mapping. :class:`NyveroHarnessAdapter` exposes such an object
through NOVA's controlled :class:`~nova.core.harness.Harness` API, normalizing
effects with the same rules as the sandbox.

Because the exact Nyvero release is not vendored here, this contract is the
integration seam: point the adapter at a Nyvero runtime that implements it, and
NOVA can mine its failures and install sandboxed harness patches without forking
Nyvero.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

from nova.core.harness import BaseHarness, Harness, HookContext, HookEffect
from nova.core.result import AgentResult
from nova.core.task import Task
from nova.core.trajectory import TrajectoryRecorder
from nova.sandbox.limits import SandboxLimits
from nova.sandbox.runner import normalize_effect

Hook = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass(slots=True)
class NyveroStep:
    """Normalized view over a Nyvero rollout step."""

    observation: str = ""
    action: str = ""
    tool_result: str | None = None
    error: str | None = None
    state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> NyveroStep:
        return cls(
            observation=str(_get(raw, "observation", "") or ""),
            action=str(_get(raw, "action", "") or ""),
            tool_result=_get(raw, "tool_result", None),
            error=_get(raw, "error", None),
            state=dict(_get(raw, "state", {}) or {}),
        )


class NyveroHarnessAdapter(BaseHarness):
    """Expose a Nyvero harness object through NOVA's controlled harness API."""

    name = "nyvero-harness"

    def __init__(self, nyvero_harness: Any, *, limits: SandboxLimits | None = None) -> None:
        self.nyvero_harness = nyvero_harness
        self.limits = limits or SandboxLimits()

    def _call(self, hook_name: str, ctx: HookContext) -> HookEffect | None:
        fn = getattr(self.nyvero_harness, hook_name, None)
        if not callable(fn):
            return None
        try:
            raw = fn(ctx.to_dict())
        except Exception:
            return None
        return normalize_effect(hook_name, raw, ctx.to_dict(), self.limits)

    def on_init(self, ctx: HookContext, nb: MutableMapping[str, Any]) -> HookEffect | None:
        return self._call("on_init", ctx)

    def make_pre_hint(self, ctx: HookContext, nb: MutableMapping[str, Any]) -> HookEffect | None:
        return self._call("make_pre_hint", ctx)

    def on_before_action(self, ctx: HookContext, nb: MutableMapping[str, Any]) -> HookEffect | None:
        return self._call("on_before_action", ctx)

    def on_post_step(self, ctx: HookContext, nb: MutableMapping[str, Any]) -> HookEffect | None:
        return self._call("on_post_step", ctx)

    @property
    def is_noop(self) -> bool:
        return not any(callable(getattr(self.nyvero_harness, name, None)) for name in
                       ("on_init", "make_pre_hint", "on_before_action", "on_post_step"))


def build_hook_bridge(harness: Harness | None) -> dict[str, Hook]:
    """Translate a NOVA harness into Nyvero hook callables."""
    active = harness if harness is not None else BaseHarness()
    notebook: MutableMapping[str, Any] = {}

    def make(hook_name: str) -> Hook:
        def hook(context: Mapping[str, Any]) -> Mapping[str, Any] | None:
            ctx = _context_from_mapping(context)
            effect = getattr(active, hook_name)(ctx, notebook)
            return effect.to_dict() if effect is not None else None

        return hook

    return {name: make(name) for name in ("on_init", "make_pre_hint", "on_before_action", "on_post_step")}


def _context_from_mapping(context: Mapping[str, Any]) -> HookContext:
    known = {
        "benchmark": context.get("benchmark", ""),
        "observation": context.get("observation", ""),
        "step": int(context.get("step", 0) or 0),
        "max_step": int(context.get("max_step", 0) or 0),
        "remaining_steps": int(context.get("remaining_steps", 0) or 0),
        "task": dict(context.get("task", {}) or {}),
        "state": dict(context.get("state", {}) or {}),
        "predicates": dict(context.get("predicates", {}) or {}),
        "action": dict(context["action"]) if context.get("action") else None,
        "admissible": list(context.get("admissible", []) or []),
    }
    reserved = set(known)
    extra = {key: value for key, value in context.items() if key not in reserved}
    return HookContext(**known, extra=extra)


class NyveroAgentAdapter:
    """Translate a Nyvero rollout agent into a NOVA :class:`Agent`."""

    def __init__(
        self,
        nyvero_agent: Any,
        *,
        benchmark: str = "nyvero",
        name: str | None = None,
        rollout_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.nyvero_agent = nyvero_agent
        self.benchmark = benchmark
        self.name = name or getattr(nyvero_agent, "name", "nyvero-agent")
        self.rollout_kwargs = dict(rollout_kwargs or {})

    def run(
        self,
        task: Task,
        harness: Harness | None = None,
        recorder: TrajectoryRecorder | None = None,
    ) -> AgentResult:
        recorder = recorder or TrajectoryRecorder(task, benchmark=self.benchmark, agent_name=self.name)
        hooks = build_hook_bridge(harness)
        rollout = self.nyvero_agent.rollout(task.instruction, hooks=hooks, **self.rollout_kwargs)

        raw_steps = list(_get(rollout, "steps", []) or [])
        for raw_step in raw_steps:
            step = NyveroStep.from_raw(raw_step)
            recorder.record_step(
                observation=step.observation,
                action={"name": step.action, "arguments": {"value": step.action}, "raw": step.action} if step.action else None,
                tool_result=step.tool_result,
                harness_effects=list(_get(raw_step, "harness_effects", []) or []),
                error=step.error,
                state=step.state,
            )

        success = bool(_get(rollout, "success", False))
        reward = float(_get(rollout, "reward", 0.0) or 0.0)
        status = str(_get(rollout, "status", "completed") or "completed")
        trajectory = recorder.finish(reward=reward, success=success, status=status)
        return AgentResult(
            task_id=task.id,
            success=success,
            reward=reward,
            trajectory=trajectory,
            status=status,
            metadata={"adapter": "nyvero", "benchmark": self.benchmark},
        )
