from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nova.adapters.nyvero import (
    NyveroAgentAdapter,
    NyveroHarnessAdapter,
    build_hook_bridge,
)
from nova.core.harness import ExecutableHarness, HookContext
from nova.core.task import Task
from nova.demo.toy import TOY_BENCHMARK, build_toy_patch_text
from nova.engineering.patch import extract_patch
from nova.sandbox.runner import LocalSandbox


@dataclass
class FakeNyveroStep:
    observation: str
    action: str = ""
    tool_result: str | None = None
    error: str | None = None
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeNyveroRollout:
    success: bool
    reward: float
    steps: list[FakeNyveroStep]
    status: str = "completed"


class FakeNyveroAgent:
    name = "fake-nyvero"

    def rollout(self, instruction: str, *, hooks=None, **kwargs: Any) -> FakeNyveroRollout:
        assert hooks is not None
        assert set(hooks) == {"on_init", "make_pre_hint", "on_before_action", "on_post_step"}
        # The bridge must expose NOVA harness effects as raw mappings.
        _ = hooks["on_before_action"](
            {
                "benchmark": "toy",
                "state": {"verified": False},
                "action": {"name": "submit"},
                "admissible": ["submit", "check"],
            }
        )
        return FakeNyveroRollout(
            success=True,
            reward=1.0,
            steps=[FakeNyveroStep(observation="verified", action="check")],
        )


class FakeNyveroHarness:
    def on_before_action(self, context: dict[str, Any]) -> dict[str, Any] | None:
        if context["state"].get("verified"):
            return None
        return {"kind": "block_and_prompt", "message": "Verify first."}


def test_hook_bridge_returns_all_four_callables() -> None:
    bridge = build_hook_bridge(None)
    assert set(bridge) == {"on_init", "make_pre_hint", "on_before_action", "on_post_step"}
    assert bridge["on_before_action"]({"state": {}, "action": {}}) is None


def test_nyvero_harness_adapter_normalizes_effects() -> None:
    adapter = NyveroHarnessAdapter(FakeNyveroHarness())
    assert adapter.is_noop is False
    ctx = HookContext(benchmark="toy", state={"verified": False}, action={"name": "submit"})
    effect = adapter.on_before_action(ctx, {})
    assert effect is not None
    assert effect.kind == "block_and_prompt"
    assert effect.message == "Verify first."
    ctx.state["verified"] = True
    assert adapter.on_before_action(ctx, {}) is None


def test_nyvero_agent_adapter_translates_rollout() -> None:
    patch = extract_patch(build_toy_patch_text(), benchmark=TOY_BENCHMARK, require_think=True, prefill_think=True)
    harness = ExecutableHarness.from_patch(patch, sandbox=LocalSandbox())
    adapter = NyveroAgentAdapter(FakeNyveroAgent())
    result = adapter.run(Task(id="t1", instruction="do it"), harness=harness)
    assert result.success is True
    assert result.reward == 1.0
    assert result.trajectory.num_steps == 1
    assert result.trajectory.steps[0].action == {"name": "check", "arguments": {"value": "check"}, "raw": "check"}


def test_nyvero_agent_adapter_without_harness() -> None:
    adapter = NyveroAgentAdapter(FakeNyveroAgent())
    result = adapter.run(Task(id="t1", instruction="do it"))
    assert result.success is True


def test_empty_nyvero_harness_is_noop() -> None:
    adapter = NyveroHarnessAdapter(object())
    assert adapter.is_noop is True
