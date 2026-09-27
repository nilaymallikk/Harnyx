from __future__ import annotations

import pytest

from harnyx.errors import PatchCompileError, SandboxTimeout
from harnyx.sandbox.isolation import SubprocessSandbox
from harnyx.sandbox.limits import SandboxLimits
from harnyx.sandbox.runner import LocalSandbox, normalize_effect


def test_compile_and_run_safe_hook_updates_notebook() -> None:
    sandbox = LocalSandbox()
    compiled = sandbox.compile(
        "def hook(ctx, nb):\n"
        "    nb['seen'] = nb.get('seen', 0) + 1\n"
        "    if 'error' in str(ctx.get('observation', '')).lower():\n"
        "        return {'message': 'Inspect the latest error before retrying.'}\n"
        "    return None\n"
    )
    nb: dict = {}
    effect = sandbox.run(compiled, {"observation": "ERROR: bad"}, nb, hook_name="make_pre_hint")
    assert nb["seen"] == 1
    assert effect is not None
    assert effect.kind == "hint"
    assert effect.message == "Inspect the latest error before retrying."


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef hook(ctx, nb):\n    return None\n",
        "from os import path\ndef hook(ctx, nb):\n    return None\n",
        "def hook(ctx, nb):\n    return open('/etc/passwd')\n",
        "def hook(ctx, nb):\n    return __import__('subprocess')\n",
        "def hook(ctx, nb):\n    return eval('1')\n",
        "def hook(ctx, nb):\n    exec('x=1')\n",
        "def hook(ctx, nb):\n    return compile('1', '<s>', 'eval')\n",
        "def hook(ctx, nb):\n    return __builtins__\n",
        "def hook(ctx, nb):\n    return ().__class__.__mro__\n",
        "def hook(ctx, nb):\n    while True:\n        pass\n",
        "class Evil:\n    pass\ndef hook(ctx, nb):\n    return None\n",
        "def hook(ctx, nb):\n    return lambda: 1\n",
    ],
)
def test_forbidden_constructs_are_rejected(source: str) -> None:
    with pytest.raises(PatchCompileError):
        LocalSandbox().compile(source)


def test_missing_hook_or_wrong_signature_rejected() -> None:
    for source in (
        "def other(ctx, nb):\n    return None\n",
        "def hook(ctx):\n    return None\n",
        "def hook(ctx, nb, extra):\n    return None\n",
        "def hook():\n    return None\n",
    ):
        with pytest.raises(PatchCompileError):
            LocalSandbox().compile(source)


def test_runtime_exception_degrades_to_none() -> None:
    sandbox = LocalSandbox()
    compiled = sandbox.compile("def hook(ctx, nb):\n    return 1 / 0\n")
    assert sandbox.run(compiled, {}, {}, hook_name="make_pre_hint") is None


def test_line_budget_timeout_degrades_to_none() -> None:
    limits = SandboxLimits(line_budget=100, time_budget_s=5.0)
    sandbox = LocalSandbox(limits)
    compiled = sandbox.compile("def hook(ctx, nb):\n    for i in range(10**9):\n        x = i\n    return None\n")
    assert sandbox.run(compiled, {}, {}, hook_name="make_pre_hint") is None


def test_alfworld_numbered_action_leakage_rejected() -> None:
    LocalSandbox().compile("def hook(ctx, nb):\n    return {'message': 'go to page 2'}\n", benchmark="webshop")
    with pytest.raises(PatchCompileError):
        LocalSandbox().compile("def hook(ctx, nb):\n    return {'message': 'go to page 2'}\n", benchmark="alfworld")


def test_normalize_effect_contracts() -> None:
    assert normalize_effect("on_init", {"skills": [{"text": "do x"}], "tool_hint": "h"}, {}).skills == ("do x",)
    assert normalize_effect("on_init", {}, {}) is None
    assert normalize_effect("make_pre_hint", {"message": "  a   b "}, {}).message == "a b"
    assert normalize_effect("make_pre_hint", {"nope": 1}, {}) is None
    assert normalize_effect("on_before_action", {"kind": "block_and_prompt"}, {}) is None
    effect = normalize_effect("on_before_action", {"kind": "block_and_prompt", "message": "stop"}, {})
    assert effect is not None and effect.kind == "block_and_prompt"
    assert normalize_effect("on_before_action", {"kind": "rewrite_action", "action": "check"}, {}).action == "check"
    # Unknown kinds degrade to no intervention.
    assert normalize_effect("on_post_step", {"kind": "explode", "message": "x"}, {}) is None


def test_subprocess_sandbox_runs_safe_hook() -> None:
    sandbox = SubprocessSandbox()
    compiled = sandbox.compile("def hook(ctx, nb):\n    return {'message': 'hello'}\n")
    effect = sandbox.run(compiled, {}, {}, hook_name="make_pre_hint")
    assert effect is not None
    assert effect.message == "hello"


def test_subprocess_sandbox_rejects_forbidden_source() -> None:
    with pytest.raises(PatchCompileError):
        SubprocessSandbox().compile("import socket\ndef hook(ctx, nb):\n    return None\n")


def test_subprocess_sandbox_enforces_wall_timeout() -> None:
    limits = SandboxLimits(time_budget_s=10.0, line_budget=10**9, cpu_seconds=None, wall_timeout_s=0.4)
    sandbox = SubprocessSandbox(limits)
    compiled = sandbox.compile("def hook(ctx, nb):\n    for i in range(10**9):\n        x = i\n    return None\n")
    with pytest.raises(SandboxTimeout):
        sandbox.run(compiled, {}, {}, hook_name="make_pre_hint")


def test_subprocess_sandbox_reflects_notebook_mutation() -> None:
    sandbox = SubprocessSandbox()
    compiled = sandbox.compile("def hook(ctx, nb):\n    nb['count'] = nb.get('count', 0) + 1\n    return None\n")
    nb: dict = {"count": 5}
    sandbox.run(compiled, {}, nb, hook_name="on_init")
    assert nb["count"] == 6
