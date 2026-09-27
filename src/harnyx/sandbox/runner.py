"""In-process, AST-hardened hook compiler and runner.

This reproduces the reference execution model: hooks are compiled with a
restricted builtin set and executed in-process under a signal timeout and a line
budget. Runtime failures normalize to ``None`` (no intervention) so a bad patch
cannot crash an episode. Stronger process isolation is available via
:class:`~harnyx.sandbox.isolation.SubprocessSandbox`.
"""

from __future__ import annotations

import builtins
import copy
import math
import re
import signal
import sys
import threading
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from types import FrameType
from typing import Any

from harnyx.core.harness import ALLOWED_EFFECT_KINDS, HookEffect
from harnyx.errors import PatchCompileError as HookCompileError
from harnyx.sandbox.limits import SandboxLimits
from harnyx.sandbox.policy import (
    ALFWORLD_INSTANCE_ACTION_RE,
    SAFE_BUILTIN_NAMES,
    HookPolicy,
    validate_hook_source,
)


class HookRuntimeTimeout(BaseException):
    """Internal signal used to abort a hook that exceeds its time budget."""


class _AttrDict(dict):
    """Dict that also supports read-style attribute access inside hooks."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _to_attrdict(value: Any) -> Any:
    if isinstance(value, dict):
        return _AttrDict({key: _to_attrdict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_attrdict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_attrdict(item) for item in value)
    return value


@dataclass(slots=True)
class CompiledHook:
    """A validated, compiled hook callable with its provenance."""

    source: str
    benchmark: str | None
    fn: Callable[[dict[str, Any], dict[str, Any]], Any] = field(repr=False)


class HookCompiler:
    """Compiles validated hook source into an executable :class:`CompiledHook`."""

    def __init__(self, policy: HookPolicy | None = None) -> None:
        self.policy = policy or HookPolicy()

    def compile(self, source: str, *, benchmark: str | None = None) -> CompiledHook:
        tree = validate_hook_source(source, benchmark=benchmark, policy=self.policy)
        namespace = self._namespace()
        try:
            code = compile(tree, "<harness_code_hook>", "exec")
            exec(code, namespace)  # noqa: S102 - restricted namespace, AST-validated
        except Exception as exc:  # pragma: no cover - defensive
            raise HookCompileError(f"hook failed to compile: {exc}") from exc
        fn = namespace.get("hook")
        if not callable(fn):
            raise HookCompileError("hook object is not callable")
        return CompiledHook(source=source, benchmark=benchmark, fn=fn)

    @staticmethod
    def _namespace() -> dict[str, Any]:
        safe_builtins = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES if hasattr(builtins, name)}
        return {
            "__builtins__": safe_builtins,
            "math": math,
            "re": re,
            "SequenceMatcher": SequenceMatcher,
        }


class LocalSandbox:
    """Default hook sandbox: in-process, restricted builtins, bounded execution."""

    def __init__(self, limits: SandboxLimits | None = None, policy: HookPolicy | None = None) -> None:
        self.limits = limits or SandboxLimits()
        self.policy = policy or HookPolicy(limits=self.limits)
        self.compiler = HookCompiler(self.policy)

    def compile(self, source: str, *, benchmark: str | None = None) -> CompiledHook:
        return self.compiler.compile(source, benchmark=benchmark)

    def run(
        self,
        compiled: CompiledHook,
        ctx: Mapping[str, Any],
        nb: MutableMapping[str, Any],
        *,
        hook_name: str,
    ) -> HookEffect | None:
        if hook_name not in ALLOWED_EFFECT_KINDS:
            return None
        ctx_copy = _to_attrdict(copy.deepcopy(dict(ctx)))
        previous_handler: Any = None
        use_signal = threading.current_thread() is threading.main_thread() and hasattr(signal, "setitimer")

        line_count = 0

        def timeout_handler(signum: int, frame: FrameType | None) -> None:
            raise HookRuntimeTimeout("hook timed out")

        def trace(frame: FrameType, event: str, arg: Any) -> Any:
            nonlocal line_count
            if event == "line":
                line_count += 1
                if line_count > self.limits.line_budget:
                    raise HookRuntimeTimeout("hook line budget exceeded")
            return trace

        try:
            if use_signal:
                previous_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, max(0.001, float(self.limits.time_budget_s)))
            old_trace = sys.gettrace()
            sys.settrace(trace)
            try:
                result = compiled.fn(ctx_copy, nb)
            finally:
                sys.settrace(old_trace)
            return normalize_effect(hook_name, result, ctx_copy, self.limits)
        except HookRuntimeTimeout:
            return None
        except Exception:
            return None
        finally:
            if use_signal:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                if previous_handler is not None:
                    signal.signal(signal.SIGALRM, previous_handler)


def _clean_text(value: Any, *, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.strip().split())
    if not text:
        return ""
    return text[:max_chars]


def _clean_action(value: Any, ctx: Mapping[str, Any], *, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    action = " ".join(value.strip().lower().split())
    if not action:
        return ""
    admissible = {" ".join(str(item).strip().lower().split()) for item in (ctx.get("admissible") or [])}
    if ALFWORLD_INSTANCE_ACTION_RE.search(action) and action not in admissible:
        return ""
    return action[:max_chars]


def normalize_effect(
    hook_name: str,
    result: Any,
    ctx: Mapping[str, Any],
    limits: SandboxLimits | None = None,
) -> HookEffect | None:
    """Normalize a raw hook return into a :class:`HookEffect` or ``None``.

    Following the reference contract, malformed or unsupported returns degrade
    to ``None`` (no intervention) rather than raising.
    """
    limits = limits or SandboxLimits()
    if result is None or not isinstance(result, dict):
        return None

    if hook_name == "on_init":
        skills: list[str] = []
        raw_skills = result.get("skills") or []
        if isinstance(raw_skills, list):
            for item in raw_skills[:5]:
                text = ""
                if isinstance(item, dict):
                    text = _clean_text(item.get("text", ""), max_chars=limits.max_return_text_chars)
                elif isinstance(item, str):
                    text = _clean_text(item, max_chars=limits.max_return_text_chars)
                if text:
                    skills.append(text)
        tool_hint = _clean_text(result.get("tool_hint", ""), max_chars=limits.max_return_text_chars)
        if not skills and not tool_hint:
            return None
        return HookEffect(kind="init", skills=tuple(skills), tool_hint=tool_hint)

    if hook_name == "make_pre_hint":
        message = _clean_text(result.get("message", ""), max_chars=limits.max_return_text_chars)
        return HookEffect(kind="hint", message=message) if message else None

    kind = str(result.get("kind") or "").strip()
    if kind not in ALLOWED_EFFECT_KINDS.get(hook_name, frozenset()):
        return None
    message = _clean_text(result.get("message", ""), max_chars=limits.max_return_text_chars)
    action = ""
    if kind in {"force_action", "rewrite_action"}:
        action = _clean_action(result.get("action", ""), ctx, max_chars=limits.max_return_text_chars)
        if not action:
            if hook_name == "on_post_step" and message:
                return HookEffect(kind="inject_hint", message=message)
            return None
    if kind in {"block_and_prompt", "inject_hint"} and not message:
        return None
    return HookEffect(kind=kind, message=message, action=action)
