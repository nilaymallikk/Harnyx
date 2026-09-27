"""The executable harness abstraction.

A Harnyx harness is the editable runtime that surrounds a frozen agent policy. It
exposes four lifecycle hooks, reproducing the Harness-R1 patch interface
(paper Appendix C, reference ``code_runner.py``):

===========================  ==================================================
Hook                         Invocation and permitted effect
===========================  ==================================================
``on_init``                  Before the first target decision. Adds reusable
                             task guidance (``skills``) or a ``tool_hint``.
``make_pre_hint``            Before a target decision. Injects a
                             state-conditioned ``message``.
``on_before_action``         After the target proposes an action, before the
                             environment executes it. May ``block_and_prompt``,
                             ``rewrite_action``, or ``force_action``.
``on_post_step``             After environment feedback. Injects recovery
                             guidance (``inject_hint``) or schedules a
                             ``force_action``.
===========================  ==================================================

The hook callable signature is exactly ``hook(ctx, nb)`` where ``ctx`` is a
read-style mapping of runtime context and ``nb`` is a mutable per-episode
notebook. The host runtime — never the hook — interprets the returned effect and
executes any environment action.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from harnyx.core.types import to_jsonable

# Lifecycle positions, in execution order.
HookNames = frozenset({"on_init", "make_pre_hint", "on_before_action", "on_post_step"})
HOOK_NAMES = HookNames

ACTION_EFFECT_KINDS = frozenset({"block_and_prompt", "force_action", "rewrite_action"})
FEEDBACK_EFFECT_KINDS = frozenset({"inject_hint", "force_action"})

# Default per-hook return-kind whitelist (mirrors reference code_runner.py).
ALLOWED_EFFECT_KINDS: dict[str, frozenset[str]] = {
    "on_init": frozenset(),
    "make_pre_hint": frozenset(),
    "on_before_action": ACTION_EFFECT_KINDS,
    "on_post_step": FEEDBACK_EFFECT_KINDS,
}


@dataclass(slots=True)
class HookContext:
    """Runtime evidence passed to a hook.

    ``extra`` carries benchmark-specific namespaces (e.g. ``webshop``,
    ``alfworld``, ``dbbench``) which are merged at the top level of the mapping
    the hook receives, matching the reference runtime context shape.
    """

    benchmark: str = ""
    observation: str = ""
    step: int = 0
    max_step: int = 0
    remaining_steps: int = 0
    task: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    predicates: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] | None = None
    admissible: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "benchmark": self.benchmark,
            "observation": self.observation,
            "step": self.step,
            "max_step": self.max_step,
            "remaining_steps": self.remaining_steps,
            "task": self.task,
            "state": self.state,
            "predicates": self.predicates,
            "action": self.action,
            "admissible": self.admissible,
        }
        for key, value in self.extra.items():
            data.setdefault(key, value)
        return to_jsonable(data)


@dataclass(slots=True)
class HookEffect:
    """A normalized hook return value.

    ``kind`` is one of ``init``, ``hint``, ``block_and_prompt``,
    ``force_action``, ``rewrite_action``, or ``inject_hint``. Invalid hook
    returns normalize to ``None`` rather than raising, so a bad patch degrades to
    no intervention instead of crashing an episode.
    """

    kind: str
    message: str = ""
    action: str = ""
    skills: tuple[str, ...] = ()
    tool_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @property
    def mutates_action(self) -> bool:
        return self.kind in {"force_action", "rewrite_action"} and bool(self.action)


@runtime_checkable
class HookSandbox(Protocol):
    """Structural interface for the sandbox that compiles and runs hook code."""

    def compile(self, source: str, *, benchmark: str | None = None) -> Any:
        """Compile untrusted hook source into an opaque executable handle."""

    def run(
        self,
        compiled: Any,
        ctx: Mapping[str, Any],
        nb: MutableMapping[str, Any],
        *,
        hook_name: str,
    ) -> HookEffect | None:
        """Execute a compiled hook, returning a normalized effect or ``None``."""


class Harness(Protocol):
    """Agent-agnostic executable harness interface."""

    name: str

    def on_init(self, ctx: HookContext, nb: MutableMapping[str, Any]) -> HookEffect | None: ...

    def make_pre_hint(self, ctx: HookContext, nb: MutableMapping[str, Any]) -> HookEffect | None: ...

    def on_before_action(self, ctx: HookContext, nb: MutableMapping[str, Any]) -> HookEffect | None: ...

    def on_post_step(self, ctx: HookContext, nb: MutableMapping[str, Any]) -> HookEffect | None: ...


class BaseHarness:
    """A no-op harness: the unmodified base runtime.

    This is the ``harness-v0`` baseline. It is also the base class for
    :class:`ExecutableHarness`.
    """

    name = "base-harness"

    def on_init(self, ctx: HookContext, nb: MutableMapping[str, Any]) -> HookEffect | None:
        return None

    def make_pre_hint(self, ctx: HookContext, nb: MutableMapping[str, Any]) -> HookEffect | None:
        return None

    def on_before_action(self, ctx: HookContext, nb: MutableMapping[str, Any]) -> HookEffect | None:
        return None

    def on_post_step(self, ctx: HookContext, nb: MutableMapping[str, Any]) -> HookEffect | None:
        return None

    @property
    def is_noop(self) -> bool:
        """Whether this harness can alter behavior. The base harness cannot."""
        return True


class ExecutableHarness(BaseHarness):
    """A harness whose lifecycle hooks are installed from an executable patch.

    The harness holds compiled hooks emitted by the sandbox. At most one hook is
    installed per lifecycle position, matching the Harness-R1 prompt contract
    ("use each hook at most once").
    """

    name = "executable-harness"

    def __init__(
        self,
        hooks: Mapping[str, Any] | None = None,
        *,
        sandbox: HookSandbox,
        benchmark: str = "",
        description: str = "",
        source_patch: Mapping[str, Any] | None = None,
    ) -> None:
        self._hooks: dict[str, Any] = dict(hooks or {})
        self._sandbox = sandbox
        self.benchmark = benchmark
        self.description = description
        self.source_patch = dict(source_patch or {})
        unknown = set(self._hooks) - set(HookNames)
        if unknown:
            raise ValueError(f"unknown lifecycle hooks: {sorted(unknown)}")

    @property
    def hook_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._hooks))

    @property
    def is_noop(self) -> bool:
        return not self._hooks

    def _invoke(
        self,
        hook_name: str,
        ctx: HookContext,
        nb: MutableMapping[str, Any],
    ) -> HookEffect | None:
        compiled = self._hooks.get(hook_name)
        if compiled is None:
            return None
        return self._sandbox.run(compiled, ctx.to_dict(), nb, hook_name=hook_name)

    def on_init(self, ctx: HookContext, nb: MutableMapping[str, Any]) -> HookEffect | None:
        return self._invoke("on_init", ctx, nb)

    def make_pre_hint(self, ctx: HookContext, nb: MutableMapping[str, Any]) -> HookEffect | None:
        return self._invoke("make_pre_hint", ctx, nb)

    def on_before_action(self, ctx: HookContext, nb: MutableMapping[str, Any]) -> HookEffect | None:
        return self._invoke("on_before_action", ctx, nb)

    def on_post_step(self, ctx: HookContext, nb: MutableMapping[str, Any]) -> HookEffect | None:
        return self._invoke("on_post_step", ctx, nb)

    @classmethod
    def from_patch(cls, patch: Any, *, sandbox: HookSandbox) -> ExecutableHarness:
        """Compile a :class:`~harnyx.engineering.patch.HarnessPatch` into a harness."""
        hooks: dict[str, Any] = {}
        for hook in patch.hooks:  # type: ignore[attr-defined]
            hooks[hook.hook] = sandbox.compile(hook.code, benchmark=patch.benchmark)
        return cls(
            hooks,
            sandbox=sandbox,
            benchmark=getattr(patch, "benchmark", ""),
            description=getattr(patch, "description", ""),
            source_patch=patch.to_dict() if hasattr(patch, "to_dict") else None,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "benchmark": self.benchmark,
            "description": self.description,
            "hooks": list(self.hook_names),
            "is_noop": self.is_noop,
        }


def merge_init_effects(effects: Sequence[HookEffect]) -> HookEffect | None:
    """Merge ``on_init`` effects (skills union, first tool hint wins)."""
    skills: list[str] = []
    tool_hint = ""
    for effect in effects:
        if effect is None:
            continue
        skills.extend(effect.skills)
        if not tool_hint and effect.tool_hint:
            tool_hint = effect.tool_hint
    if not skills and not tool_hint:
        return None
    return HookEffect(kind="init", skills=tuple(dict.fromkeys(skills)), tool_hint=tool_hint)
