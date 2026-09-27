from __future__ import annotations

from nova.core.harness import (
    ACTION_EFFECT_KINDS,
    BaseHarness,
    ExecutableHarness,
    HookContext,
    HookEffect,
    merge_init_effects,
)
from nova.engineering.patch import CodeHook, HarnessPatch
from nova.sandbox.runner import LocalSandbox

GOOD_HOOK = """def hook(ctx, nb):
    action = ctx.get('action') or {}
    state = ctx.get('state') or {}
    nb['seen'] = nb.get('seen', 0) + 1
    if str(action.get('name')) == 'submit' and not state.get('verified'):
        return {'kind': 'block_and_prompt', 'message': 'Verify before submitting.'}
    return None
"""


def make_patch(hook: str = "on_before_action", code: str = GOOD_HOOK) -> HarnessPatch:
    return HarnessPatch(benchmark="toy", hooks=(CodeHook(hook=hook, code=code),))


def test_base_harness_is_noop() -> None:
    harness = BaseHarness()
    assert harness.is_noop is True
    ctx = HookContext(benchmark="toy", observation="x", state={"verified": False})
    for name in ("on_init", "make_pre_hint", "on_before_action", "on_post_step"):
        assert getattr(harness, name)(ctx, {}) is None


def test_executable_harness_installs_hooks_and_runs() -> None:
    sandbox = LocalSandbox()
    harness = ExecutableHarness.from_patch(make_patch(), sandbox=sandbox)
    assert harness.is_noop is False
    assert harness.hook_names == ("on_before_action",)

    ctx = HookContext(
        benchmark="toy",
        observation="pending",
        state={"verified": False},
        action={"name": "submit", "arguments": {"value": "submit"}},
    )
    nb: dict = {}
    effect = harness.on_before_action(ctx, nb)
    assert isinstance(effect, HookEffect)
    assert effect.kind == "block_and_prompt"
    assert effect.message == "Verify before submitting."
    assert nb["seen"] == 1

    # Verified state passes the action through untouched.
    ctx.state["verified"] = True
    assert harness.on_before_action(ctx, nb) is None


def test_executable_harness_rejects_unknown_hook_name() -> None:
    import pytest

    with pytest.raises(ValueError):
        ExecutableHarness({"nope": object()}, sandbox=LocalSandbox())


def test_hook_context_merges_benchmark_extra() -> None:
    ctx = HookContext(benchmark="webshop", extra={"webshop": {"search_queries": ["mug"]}})
    data = ctx.to_dict()
    assert data["webshop"]["search_queries"] == ["mug"]
    assert data["benchmark"] == "webshop"


def test_merge_init_effects_deduplicates_skills() -> None:
    merged = merge_init_effects(
        [
            HookEffect(kind="init", skills=("a", "b"), tool_hint="hint"),
            HookEffect(kind="init", skills=("b", "c")),
        ]
    )
    assert merged is not None
    assert merged.skills == ("a", "b", "c")
    assert merged.tool_hint == "hint"
    assert merge_init_effects([]) is None


def test_action_effect_kinds_constant() -> None:
    assert {"block_and_prompt", "force_action", "rewrite_action"} == ACTION_EFFECT_KINDS
