"""Nyvero adapter.

`Nyvero <https://github.com/NilLab-agi/Nyvero>`_ is a minimal coding-agent
harness: one loop that calls an OpenAI-compatible model, runs each tool call
through one shared permission-checked executor, appends the results, and repeats.

This adapter drives that loop from Harnyx so the four harness hooks wrap it
without editing Nyvero:

======================  =================================================================
Nyvero loop point       Harnyx hook
======================  =================================================================
before the first call   ``on_init``          (reusable skills / tool hints)
before each model call  ``make_pre_hint``    (state-conditioned hint)
before ``execute_tool`` ``on_before_action`` (block / rewrite / force a tool call)
after ``execute_tool``  ``on_post_step``     (recovery hint / force)
======================  =================================================================

Harnyx never imports Nyvero: :class:`NyveroBackend` carries the callables, and
:meth:`NyveroBackend.from_installation` wires the real ones lazily.

Reward is *not* built in. A coding agent has no native success signal, so you
must supply an objective :data:`OutcomeFn` (for example, run the repository's
tests in the workspace and return ``(exit_code == 0, 1.0 or 0.0)``). Harnyx's
reward is the realized rerun delta, never a judge.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harnyx.core.agent import Action
from harnyx.core.harness import BaseHarness, Harness, HookContext, HookEffect
from harnyx.core.result import AgentResult
from harnyx.core.task import Task
from harnyx.core.trajectory import TrajectoryRecorder
from harnyx.sandbox.limits import SandboxLimits

# A backend model call returns {"content": str, "tool_calls": [{"id","name","arguments"}]}
ModelCall = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], dict[str, Any]]
ToolExecutor = Callable[[str, dict[str, Any]], str]
# Given a task and the workspace after a run, return (success, reward).
OutcomeFn = Callable[[Task, Path], tuple[bool, float]]
# Convert a hook action string into a concrete replacement for the pending call.
EffectApplier = Callable[[str, Action], Action]

HOOK_NAMES = ("on_init", "make_pre_hint", "on_before_action", "on_post_step")

_ERROR_MARKERS = ("tool execution failed", "denied", "unknown tool", "traceback", "error:")


@dataclass
class NyveroBackend:
    """The Nyvero callables Harnyx needs, without importing Nyvero."""

    call_model: ModelCall
    execute_tool: ToolExecutor
    tools: list[dict[str, Any]]
    system_prompt: str

    @classmethod
    def from_installation(
        cls,
        *,
        repo: str | Path | None = None,
        auto_approve: bool = True,
    ) -> NyveroBackend:
        """Wire a real Nyvero checkout.

        Args:
            repo: path to the Nyvero repository (added to ``sys.path`` if given).
            auto_approve: replace Nyvero's interactive confirmation with an
                automatic yes so the loop can run headless. ``DENY`` rules
                (destructive commands) still apply; only the *confirm* step is
                bypassed. An OS sandbox (bubblewrap/seatbelt) still wraps bash.
        """
        if repo is not None:
            path = str(Path(repo).resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
        from nyvero import llm, prompt, ui  # type: ignore[import-not-found]
        from nyvero import tools as nyvero_tools  # type: ignore[import-not-found]

        if auto_approve:
            ui.confirm = lambda *args, **kwargs: True

        schemas = [
            nyvero_tools.BASH_TOOL,
            nyvero_tools.READ_FILE_TOOL,
            nyvero_tools.WRITE_FILE_TOOL,
            nyvero_tools.LIST_FILES_TOOL,
            nyvero_tools.FILE_EXISTS_TOOL,
            nyvero_tools.DELETE_FILE_TOOL,
            nyvero_tools.EDIT_FILE_TOOL,
            nyvero_tools.READ_SKILL_TOOL,
            nyvero_tools.ADD_TODO_TOOL,
            nyvero_tools.LIST_TODOS_TOOL,
            nyvero_tools.UPDATE_TODO_TOOL,
            nyvero_tools.TASK_TOOL,
        ]

        def call_model(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
            message = llm.call_llm(messages, tools=tools)
            calls = []
            for call in getattr(message, "tool_calls", None) or []:
                calls.append(
                    {
                        "id": getattr(call, "id", ""),
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                )
            return {"content": message.content or "", "tool_calls": calls}

        return cls(call_model=call_model, execute_tool=nyvero_tools.execute_tool, tools=schemas, system_prompt=prompt.SYSTEM_PROMPT)


def _run_check(command: str, workspace: Path, timeout: float, reward: float) -> tuple[bool, float]:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False, 0.0
    ok = proc.returncode == 0
    return ok, reward if ok else 0.0


def workspace_check(
    command: str,
    *,
    timeout: float = 600.0,
    reward: float = 1.0,
) -> OutcomeFn:
    """Outcome function that runs one trusted check command in the workspace.

    ``command`` is *your* check (e.g. ``"uv run pytest -q"``), never model
    output. Success is ``exit code 0``.
    """

    def outcome(task: Task, workspace: Path) -> tuple[bool, float]:
        return _run_check(command, workspace, timeout, reward)

    return outcome


def per_task_check(
    commands: Mapping[str, str],
    *,
    timeout: float = 600.0,
    reward: float = 1.0,
) -> OutcomeFn:
    """Outcome function that runs a different check command per ``task.id``."""

    def outcome(task: Task, workspace: Path) -> tuple[bool, float]:
        command = commands.get(task.id)
        if command is None:
            return False, 0.0
        return _run_check(command, workspace, timeout, reward)

    return outcome


def _primary_argument_key(arguments: Mapping[str, Any]) -> str | None:
    for key, value in arguments.items():
        if isinstance(value, str):
            return key
    return None


def default_effect_applier(text: str, current: Action) -> Action:
    """Map a hook action string onto a pending Nyvero tool call.

    The default replaces the current call's first string argument (``command``
    for ``bash``, ``path`` for the file tools). Override for structured
    rewrites.
    """
    key = _primary_argument_key(current.arguments)
    arguments = dict(current.arguments)
    if key is None:
        arguments["value"] = text
    else:
        arguments[key] = text
    return Action(name=current.name, arguments=arguments, raw=text)


class NyveroAgentAdapter:
    """Run a Nyvero loop as a Harnyx :class:`~harnyx.core.agent.Agent`."""

    def __init__(
        self,
        backend: NyveroBackend,
        *,
        benchmark: str = "nyvero",
        max_steps: int = 30,
        workspace: str | Path = ".",
        outcome: OutcomeFn | None = None,
        apply_effect: EffectApplier | None = None,
        name: str | None = None,
    ) -> None:
        self.backend = backend
        self.benchmark = benchmark
        self.max_steps = max_steps
        self.workspace = Path(workspace)
        self.outcome = outcome
        self.apply_effect = apply_effect or default_effect_applier
        self.name = name or "nyvero-agent"

    # ------------------------------------------------------------------ #
    def _context(
        self,
        task: Task,
        messages: Sequence[Mapping[str, Any]],
        step: int,
        action: Action | None,
        last_result: str | None,
    ) -> HookContext:
        tool_results = [str(m.get("content", "")) for m in messages if m.get("role") == "tool"]
        error_streak = 0
        for result in reversed(tool_results):
            if any(marker in result.lower() for marker in _ERROR_MARKERS):
                error_streak += 1
            else:
                break
        state = {
            "step": step,
            "remaining_steps": max(0, self.max_steps - step),
            "last_result": last_result or (tool_results[-1] if tool_results else ""),
            "recent_tool_results": tool_results[-5:],
            "num_tool_results": len(tool_results),
            "error_streak": error_streak,
            "workspace": str(self.workspace),
        }
        if action is not None:
            state["tool_name"] = action.name
            state["arguments"] = dict(action.arguments)
        return HookContext(
            benchmark=self.benchmark,
            observation=last_result or "",
            step=step,
            max_step=self.max_steps,
            remaining_steps=max(0, self.max_steps - step),
            task={"id": task.id, "instruction": task.instruction},
            state=state,
            predicates={
                "error_streak": error_streak > 0,
                "repeated_tool_result": bool(tool_results and tool_results[-1] == last_result),
                "near_step_limit": step >= self.max_steps - 2,
            },
            action=action.to_dict() if action is not None else None,
            admissible=[schema["function"]["name"] for schema in self.backend.tools],
        )

    def _apply(self, effect: HookEffect, current: Action) -> Action:
        return self.apply_effect(effect.action, current)

    def _parse_arguments(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, Mapping):
            return dict(raw)
        try:
            parsed = json.loads(raw or "{}")
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    # ------------------------------------------------------------------ #
    def run(
        self,
        task: Task,
        harness: Harness | None = None,
        recorder: TrajectoryRecorder | None = None,
    ) -> AgentResult:
        active: Harness = harness if harness is not None else BaseHarness()
        recorder = recorder or TrajectoryRecorder(task, benchmark=self.benchmark, agent_name=self.name)
        notebook: MutableMapping[str, Any] = {}
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.backend.system_prompt},
            {"role": "user", "content": task.instruction},
        ]

        init = active.on_init(self._context(task, messages, 0, None, None), notebook)
        if init is not None and init.skills:
            tips = "\n".join(f"- {skill}" for skill in init.skills)
            messages.append({"role": "system", "content": f"Reusable guidance:\n{tips}"})

        status = "completed"
        last_result: str | None = None
        for step in range(self.max_steps):
            hint = active.make_pre_hint(self._context(task, messages, step, None, last_result), notebook)
            if hint is not None and hint.message:
                messages.append({"role": "user", "content": f"[harness hint] {hint.message}"})

            reply = self.backend.call_model(messages, self.backend.tools)
            calls = list(reply.get("tool_calls") or [])
            if not calls:
                messages.append({"role": "assistant", "content": str(reply.get("content", ""))})
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": str(reply.get("content", "")),
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["name"], "arguments": call["arguments"]},
                        }
                        for call in calls
                    ],
                }
            )

            for call in calls:
                action = Action(
                    name=str(call["name"]),
                    arguments=self._parse_arguments(call.get("arguments")),
                    raw=str(call.get("arguments", "")),
                )
                effects: list[dict[str, Any]] = []

                before = active.on_before_action(self._context(task, messages, step, action, last_result), notebook)
                if before is not None:
                    effects.append(before.to_dict())
                    if before.kind == "block_and_prompt":
                        last_result = f"[harness blocked] {before.message or 'action blocked'}"
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": last_result})
                        recorder.record_step(
                            observation=last_result, action=action.to_dict(), tool_result=last_result,
                            harness_effects=effects, state={"step": step},
                        )
                        continue
                    if before.kind in {"rewrite_action", "force_action"} and before.action:
                        action = self._apply(before, action)

                last_result = self.backend.execute_tool(action.name, action.arguments)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": last_result})

                after = active.on_post_step(self._context(task, messages, step, action, last_result), notebook)
                if after is not None:
                    effects.append(after.to_dict())
                    if after.kind == "inject_hint" and after.message:
                        messages.append({"role": "user", "content": f"[harness recovery] {after.message}"})

                recorder.record_step(
                    observation=last_result, action=action.to_dict(), tool_result=last_result,
                    harness_effects=effects, state={"step": step},
                )
        else:
            status = "task_limit_reached"

        success, reward = self.outcome(task, self.workspace) if self.outcome is not None else (False, 0.0)
        trajectory = recorder.finish(reward=reward, success=success, status=status)
        return AgentResult(
            task_id=task.id,
            success=success,
            reward=reward,
            trajectory=trajectory,
            status=status,
            metadata={"adapter": "nyvero", "benchmark": self.benchmark},
        )


def build_hook_bridge(harness: Harness | None, *, limits: SandboxLimits | None = None) -> dict[str, Callable[[Mapping[str, Any]], dict[str, Any] | None]]:
    """Expose a Harnyx harness as four callables for Nyvero's *own* loop.

    Use this to install an accepted patch into a live Nyvero `agent.py` loop:
    call ``bridge["on_before_action"](context)`` before ``execute_tool`` and
    ``bridge["on_post_step"](context)`` after it.
    """
    active: Harness = harness if harness is not None else BaseHarness()
    limits = limits or SandboxLimits()
    notebook: MutableMapping[str, Any] = {}

    def make(hook_name: str) -> Callable[[Mapping[str, Any]], dict[str, Any] | None]:
        def call(context: Mapping[str, Any]) -> dict[str, Any] | None:
            effect = getattr(active, hook_name)(_context_from_mapping(context), notebook)
            return effect.to_dict() if effect is not None else None

        return call

    bridge = {name: make(name) for name in HOOK_NAMES}
    del limits
    return bridge


def _context_from_mapping(context: Mapping[str, Any]) -> HookContext:
    known = {
        "benchmark": str(context.get("benchmark", "nyvero")),
        "observation": str(context.get("observation", "")),
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


__all__ = [
    "EffectApplier",
    "NyveroAgentAdapter",
    "NyveroBackend",
    "OutcomeFn",
    "build_hook_bridge",
    "default_effect_applier",
    "per_task_check",
    "workspace_check",
]
