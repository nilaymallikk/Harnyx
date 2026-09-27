"""A deterministic toy environment for the Harness-R1 loop.

The scenario mirrors the paper's WebShop "premature purchase" case at miniature
scale: a frozen policy submits before verifying, and a narrow pre-action guard
delays submission until the environment is verified.

    baseline: policy submits immediately -> reward 0
    patched:  guard blocks the premature submit, policy verifies, then submits
              -> reward 1
    reward:   +1

No external model is involved: a scripted engineer returns the guard patch, and
the real parser, validator, sandbox, evaluator, reward, selector, and versioning
are all exercised end to end.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nova.core.agent import Action, HarnessedAgent, StepResult
from nova.core.task import Task
from nova.engineering.harness_engineer import ScriptedHarnessEngineer
from nova.engineering.patch import extract_patch
from nova.evaluation.evaluator import LocalEvaluator
from nova.optimization.optimizer import HarnessOptimizer, OptimizationConfig

TOY_BENCHMARK = "toy"


class GuardedCommitEnvironment:
    """A task that fails unless the agent verifies before submitting."""

    name = "guarded-commit"

    def __init__(self) -> None:
        self.verified = False
        self.submitted = False
        self._reward = 0.0

    def reset(self, task: Task) -> str:
        self.verified = False
        self.submitted = False
        self._reward = 0.0
        return "A commit is pending. Actions: submit, check."

    def step(self, action: Action) -> StepResult:
        name = (action.name or "").strip().lower()
        if name == "check":
            self.verified = True
            return StepResult(observation="Verification complete.", done=False, reward=0.0)
        if name == "submit":
            self.submitted = True
            if self.verified:
                self._reward = 1.0
                return StepResult(observation="Commit accepted.", done=True, reward=1.0)
            self._reward = 0.0
            return StepResult(observation="Commit rejected: not verified.", done=True, reward=0.0)
        return StepResult(observation=f"Unknown action: {name}", done=False, reward=0.0)

    def state(self) -> dict[str, Any]:
        return {"verified": self.verified, "submitted": self.submitted}

    def predicates(self) -> dict[str, Any]:
        return {"verified": self.verified, "submitted": self.submitted}

    def success(self) -> bool:
        return self.submitted and self.verified

    def episode_reward(self) -> float:
        return self._reward

    def admissible_actions(self) -> list[str]:
        return ["submit", "check"]


class SubmitFirstPolicy:
    """A frozen deterministic policy that submits unless asked to verify."""

    name = "submit-first-policy"

    def act(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        step: int,
        admissible: Sequence[str],
    ) -> Action:
        history = "\n".join(str(message.get("content", "")).lower() for message in messages)
        asked_to_verify = "verify before" in history
        already_verified = "verification complete" in history
        if asked_to_verify and not already_verified:
            return Action(name="check", arguments={"value": "check"}, raw="check")
        return Action(name="submit", arguments={"value": "submit"}, raw="submit")


def build_toy_patch_text() -> str:
    """The engineer's full ``prefilled-think`` response for the toy failure."""
    return (
        "Recurring failure: the policy submits before the environment is verified. "
        "The narrow, observable fix is a pre-action guard that blocks submit while "
        "state.verified is false and asks the policy to verify first.\n"
        "</think>\n"
        "<patch>\n"
        "{\n"
        '  "benchmark": "toy",\n'
        '  "description": "Block submit until the environment is verified.",\n'
        '  "actions": [\n'
        "    {\n"
        '      "type": "add_code_hook",\n'
        '      "hook": "on_before_action",\n'
        '      "code": "def hook(ctx, nb):\\n'
        "    action = ctx.get('action') or {}\\n"
        "    state = ctx.get('state') or {}\\n"
        "    if str(action.get('name')) == 'submit' and not state.get('verified'):\\n"
        "        return {'kind': 'block_and_prompt', 'message': 'Verify before submitting.'}\\n"
        "    return None\\n"
        '"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "</patch>"
    )


def build_toy_tasks() -> list[Task]:
    return [Task(id="toy-1", instruction="Submit the answer without committing prematurely.")]


def build_toy_agent(max_steps: int = 5) -> HarnessedAgent:
    return HarnessedAgent(
        SubmitFirstPolicy(),
        GuardedCommitEnvironment(),
        benchmark=TOY_BENCHMARK,
        max_steps=max_steps,
        name="toy-agent",
    )


def run_demo(run_root: str | Path = "runs", *, candidates: int = 3, iterations: int = 1) -> dict[str, Any]:
    """Run the deterministic end-to-end demo and return a summary dict."""
    patch = extract_patch(
        build_toy_patch_text(),
        benchmark=TOY_BENCHMARK,
        require_think=True,
        prefill_think=True,
        source="scripted-engineer",
    )
    engineer = ScriptedHarnessEngineer([patch], loop=True)
    optimizer = HarnessOptimizer(
        build_toy_agent(),
        engineer,
        evaluator=LocalEvaluator(benchmark=TOY_BENCHMARK),
        benchmark=TOY_BENCHMARK,
        model="scripted",
        config=OptimizationConfig(candidates=candidates, iterations=iterations, reward_metric="delta_average_reward"),
        run_dir=run_root,
    )
    result = optimizer.optimize(build_toy_tasks())
    return {
        "baseline_success": result.baseline.num_success,
        "baseline_reward": result.baseline.mean_reward,
        "patched_success": result.final.num_success,
        "patched_reward": result.final.mean_reward,
        "reward": result.final.mean_reward - result.baseline.mean_reward,
        "accepted": [version.version for version in result.accepted_versions],
        "run_dir": result.run_dir,
    }
