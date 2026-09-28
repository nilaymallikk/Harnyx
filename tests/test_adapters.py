from __future__ import annotations

from pathlib import Path
from typing import Any

from harnyx.adapters.nyvero import (
    NyveroAgentAdapter,
    NyveroBackend,
    build_hook_bridge,
    default_effect_applier,
    per_task_check,
    workspace_check,
)
from harnyx.core.agent import Action
from harnyx.core.harness import ExecutableHarness
from harnyx.core.task import Task
from harnyx.engineering.patch import CodeHook, HarnessPatch
from harnyx.sandbox.runner import LocalSandbox


def make_backend(script: list[dict[str, Any]], executed: list[tuple[str, dict]]) -> NyveroBackend:
    def call_model(messages, tools):  # noqa: ANN001
        return script.pop(0) if script else {"content": "done", "tool_calls": []}

    def execute_tool(name: str, arguments: dict) -> str:
        executed.append((name, arguments))
        return f"{name} ok"

    return NyveroBackend(
        call_model=call_model,
        execute_tool=execute_tool,
        tools=[{"type": "function", "function": {"name": "bash"}}],
        system_prompt="sys",
    )


def _adapter(script, executed, **kwargs):
    return NyveroAgentAdapter(make_backend(script, executed), benchmark="nyvero", **kwargs)


def test_plain_loop_runs_tool_and_reports_outcome() -> None:
    script = [
        {"content": "", "tool_calls": [{"id": "1", "name": "bash", "arguments": '{"command": "ls"}'}]},
        {"content": "done", "tool_calls": []},
    ]
    executed: list[tuple[str, dict]] = []
    adapter = _adapter(script, executed, outcome=lambda task, ws: (True, 1.0))
    result = adapter.run(Task(id="t1", instruction="list files"))
    assert executed == [("bash", {"command": "ls"})]
    assert result.success is True
    assert result.reward == 1.0
    assert result.trajectory.num_steps == 1


def test_guard_blocks_a_dangerous_tool_call() -> None:
    script = [
        {"content": "", "tool_calls": [{"id": "1", "name": "bash", "arguments": '{"command": "rm -rf /"}'}]},
        {"content": "done", "tool_calls": []},
    ]
    executed: list[tuple[str, dict]] = []
    patch = HarnessPatch(
        benchmark="nyvero",
        hooks=(
            CodeHook(
                hook="on_before_action",
                code=(
                    "def hook(ctx, nb):\n"
                    "    args = (ctx.get('action') or {}).get('arguments') or {}\n"
                    "    if 'rm ' in str(args.get('command', '')):\n"
                    "        return {'kind': 'block_and_prompt', 'message': 'No destructive commands.'}\n"
                    "    return None\n"
                ),
            ),
        ),
    )
    harness = ExecutableHarness.from_patch(patch, sandbox=LocalSandbox())
    adapter = _adapter(script, executed, outcome=lambda task, ws: (True, 1.0))
    result = adapter.run(Task(id="t1", instruction="clean up"), harness=harness)
    assert executed == []  # the dangerous call never reached the executor
    assert result.trajectory.steps[0].harness_effects[0]["kind"] == "block_and_prompt"


def test_guard_can_rewrite_an_action() -> None:
    script = [
        {"content": "", "tool_calls": [{"id": "1", "name": "bash", "arguments": '{"command": "ls"}'}]},
        {"content": "done", "tool_calls": []},
    ]
    executed: list[tuple[str, dict]] = []
    patch = HarnessPatch(
        benchmark="nyvero",
        hooks=(
            CodeHook(
                hook="on_before_action",
                code=(
                    "def hook(ctx, nb):\n"
                    "    args = (ctx.get('action') or {}).get('arguments') or {}\n"
                    "    if args.get('command') == 'ls':\n"
                    "        return {'kind': 'rewrite_action', 'action': 'ls -la'}\n"
                    "    return None\n"
                ),
            ),
        ),
    )
    harness = ExecutableHarness.from_patch(patch, sandbox=LocalSandbox())
    adapter = _adapter(script, executed, outcome=lambda task, ws: (True, 1.0))
    adapter.run(Task(id="t1", instruction="list"), harness=harness)
    assert executed == [("bash", {"command": "ls -la"})]


def test_workspace_check_runs_a_command(tmp_path: Path) -> None:
    task = Task(id="t", instruction="x")
    assert workspace_check("exit 0")(task, tmp_path) == (True, 1.0)
    assert workspace_check("exit 3")(task, tmp_path) == (False, 0.0)


def test_per_task_check_selects_by_task_id(tmp_path: Path) -> None:
    check = per_task_check({"a": "exit 0", "b": "exit 1"})
    assert check(Task(id="a", instruction="x"), tmp_path) == (True, 1.0)
    assert check(Task(id="b", instruction="x"), tmp_path) == (False, 0.0)
    assert check(Task(id="missing", instruction="x"), tmp_path) == (False, 0.0)


def test_default_effect_applier_replaces_first_string_argument() -> None:
    current = Action(name="bash", arguments={"command": "ls", "timeout": 5}, raw="ls")
    patched = default_effect_applier("pwd", current)
    assert patched.name == "bash"
    assert patched.arguments["command"] == "pwd"


def test_hook_bridge_returns_four_callables() -> None:
    bridge = build_hook_bridge(None)
    assert set(bridge) == {"on_init", "make_pre_hint", "on_before_action", "on_post_step"}
    assert bridge["on_before_action"]({"state": {}, "action": {}}) is None
