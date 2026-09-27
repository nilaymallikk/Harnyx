"""Example Harnyx plugin used by the CLI.

The scenario: a frozen policy reports a value before inspecting the data. A
pre-action guard blocks the report until the agent has counted. This is the same
shape as the toy demo but with a different environment, demonstrating how to plug
a custom agent/tasks/engineer into ``harnyx optimize``.

Usage::

    harnyx optimize --plugin examples.plugins.verify_agent:build
    harnyx evaluate --plugin examples.plugins.verify_agent:build
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from harnyx import (
    Action,
    HarnessedAgent,
    ScriptedHarnessEngineer,
    StepResult,
    Task,
)
from harnyx.engineering.patch import CodeHook, HarnessPatch

BENCHMARK = "verify"

GUARD_CODE = (
    "def hook(ctx, nb):\n"
    "    action = ctx.get('action') or {}\n"
    "    state = ctx.get('state') or {}\n"
    "    if str(action.get('name')) == 'report' and not state.get('counted'):\n"
    "        return {'kind': 'block_and_prompt', 'message': 'Inspect the data before reporting.'}\n"
    "    return None\n"
)


class CountingEnvironment:
    name = "counting"

    def reset(self, task: Task) -> str:
        self.counted = False
        self.reported = False
        self._reward = 0.0
        self.expected = int(task.payload.get("expected", 0))
        return "Data is hidden. Actions: inspect, report."

    def step(self, action: Action) -> StepResult:
        name = (action.name or "").strip().lower()
        if name == "inspect":
            self.counted = True
            return StepResult(observation=f"Inspection complete. count={self.expected}", done=False)
        if name == "report":
            self.reported = True
            if self.counted:
                self._reward = 1.0
                return StepResult(observation="Report accepted.", done=True, reward=1.0)
            self._reward = 0.0
            return StepResult(observation="Report rejected: data not inspected.", done=True, reward=0.0)
        return StepResult(observation=f"Unknown action: {name}", done=False)

    def state(self) -> dict[str, Any]:
        return {"counted": self.counted, "reported": self.reported}

    def predicates(self) -> dict[str, Any]:
        return {"counted": self.counted}

    def success(self) -> bool:
        return self.reported and self.counted

    def episode_reward(self) -> float:
        return self._reward

    def admissible_actions(self) -> list[str]:
        return ["inspect", "report"]


class ReportFirstPolicy:
    name = "report-first-policy"

    def act(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        step: int,
        admissible: Sequence[str],
    ) -> Action:
        history = "\n".join(str(m.get("content", "")).lower() for m in messages)
        if "inspect the data" in history and "inspection complete" not in history:
            return Action(name="inspect", arguments={"value": "inspect"}, raw="inspect")
        return Action(name="report", arguments={"value": "report"}, raw="report")


def good_patch() -> HarnessPatch:
    return HarnessPatch(
        benchmark=BENCHMARK,
        description="Block report until data has been inspected.",
        hooks=(CodeHook(hook="on_before_action", code=GUARD_CODE),),
        source="example",
    )


def build() -> dict[str, Any]:
    agent = HarnessedAgent(ReportFirstPolicy(), CountingEnvironment(), benchmark=BENCHMARK, max_steps=6)
    tasks = [
        Task(id="verify-1", instruction="Report the item count.", payload={"expected": 3}),
        Task(id="verify-2", instruction="Report the item count.", payload={"expected": 7}),
    ]
    return {
        "agent": agent,
        "tasks": tasks,
        "benchmark": BENCHMARK,
        "engineer": ScriptedHarnessEngineer([good_patch()]),
    }


__all__ = ["BENCHMARK", "CountingEnvironment", "ReportFirstPolicy", "build", "good_patch"]
