#!/usr/bin/env python3
"""Optimize a Nyvero coding-agent harness with Harnyx.

This wires the real Nyvero API (https://github.com/NilLab-agi/Nyvero) to Harnyx:

    baseline:  run Nyvero on the task batch, run each task's check
    optimize:  mine failures -> engineer writes a harness patch -> sandbox
               -> rerun the same tasks -> keep the patch only if it improves

The harness patch never touches Nyvero's source and never touches the model. It
is an executable overlay applied by the adapter at four loop points.

You must supply:
  * a Nyvero checkout (``--nyvero-repo``),
  * a workspace the agent edits (``--workspace``),
  * one trusted check command per task (``CHECKS`` below), and
  * an OpenAI-compatible endpoint for the engineer (``--engineer-base-url``).

The engineer endpoint is just any chat-completions server; it can be the same
DeepSeek endpoint Nyvero uses.

Run:  python examples/nyvero/optimize_nyvero.py \
          --workspace ~/myproject \
          --engineer-base-url https://api.deepseek.com --engineer-model deepseek-chat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from harnyx import (  # noqa: E402
    HarnessOptimizer,
    LLMHarnessEngineer,
    LocalEvaluator,
    Task,
)
from harnyx.adapters.nyvero import NyveroAgentAdapter, NyveroBackend, per_task_check  # noqa: E402
from harnyx.llm.openai import OpenAICompatibleProvider  # noqa: E402
from harnyx.optimization.optimizer import OptimizationConfig  # noqa: E402

# One task = one instruction + the command that objectively proves it is done.
# `{workspace}` is replaced with --workspace at build time. Keep these fail-to-pass:
# they must FAIL on the untouched workspace and PASS after a correct fix.
TASKS: list[tuple[str, str, str]] = [
    (
        "fix-parser-edge-case",
        "The parser drops empty lines at end-of-file. Find and fix it; run the tests.",
        "uv run pytest tests/test_parser.py -q",
    ),
    (
        "add-missing-validation",
        "`load_config` accepts a negative timeout. Add validation and a test.",
        "uv run pytest tests/test_config.py -q",
    ),
]


def build_tasks() -> tuple[list[Task], dict[str, str]]:
    tasks: list[Task] = []
    checks: dict[str, str] = {}
    for task_id, instruction, command in TASKS:
        tasks.append(Task(id=task_id, instruction=instruction))
        checks[task_id] = command
    return tasks, checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nyvero-repo", default=str(Path.home() / "Nyvero"))
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--engineer-base-url", required=True)
    parser.add_argument("--engineer-model", required=True)
    parser.add_argument("--engineer-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--run-dir", default="runs/nyvero")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        print(f"workspace not found: {workspace}", file=sys.stderr)
        return 2

    tasks, checks = build_tasks()

    # 1. The Nyvero loop, headless and non-interactive.
    backend = NyveroBackend.from_installation(repo=args.nyvero_repo, auto_approve=True)

    # 2. Wrap it as a Harnyx agent. The outcome function is the objective reward.
    agent = NyveroAgentAdapter(
        backend,
        benchmark="nyvero",
        max_steps=args.max_steps,
        workspace=workspace,
        outcome=per_task_check(checks, timeout=900.0),
    )

    # 3. The engineer: a model that reads failure traces and writes harness patches.
    engineer = LLMHarnessEngineer(
        OpenAICompatibleProvider(
            base_url=args.engineer_base_url,
            model=args.engineer_model,
            env_key=args.engineer_api_key_env,
        ),
        benchmark="nyvero",
    )

    # 4. Mine failures -> propose patches -> rerun the same tasks -> keep what improves.
    result = HarnessOptimizer(
        agent,
        engineer,
        evaluator=LocalEvaluator(benchmark="nyvero"),
        benchmark="nyvero",
        model=args.engineer_model,
        config=OptimizationConfig(candidates=args.candidates, iterations=args.iterations),
        run_dir=args.run_dir,
    ).optimize(tasks)

    print()
    print(f"baseline success : {result.baseline.num_success}/{result.baseline.n}")
    print(f"final success    : {result.final.num_success}/{result.final.n}")
    print(f"accepted         : {', '.join(v.version for v in result.accepted_versions) or 'none'}")
    print(f"run directory    : {result.run_dir}")
    print()
    print("To install an accepted patch into a live Nyvero session, load")
    print(f"  {result.run_dir}/accepted_patch.json")
    print("and call the four hooks around Nyvero's loop (see docs/nyvero.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
