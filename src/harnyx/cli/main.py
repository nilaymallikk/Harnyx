"""Harnyx command-line interface.

Commands
--------
``harnyx run``                 Deterministic end-to-end toy demo.
``harnyx evaluate``            Evaluate an agent (toy or plugin) on a task batch.
``harnyx optimize``            Run failure -> patch -> rerun -> reward -> accept.
``harnyx generate-failures``   Build a failure packet from trajectory JSONL.
``harnyx generate-patches``    Generate candidate patches for a packet.
``harnyx inspect-trajectory``  Pretty-print a trajectory JSONL file.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

from harnyx.config import HarnyxConfig, SandboxConfig, load_config
from harnyx.core.result import EvaluationResult
from harnyx.core.task import Task
from harnyx.core.trajectory import Trajectory
from harnyx.core.types import read_json, read_jsonl, write_json, write_jsonl
from harnyx.engineering.harness_engineer import LLMHarnessEngineer
from harnyx.engineering.validation import PatchValidator
from harnyx.errors import HarnyxError
from harnyx.evaluation.evaluator import LocalEvaluator
from harnyx.llm.openai import OpenAICompatibleProvider
from harnyx.optimization.failure_analysis import FailurePacket, TraceFailureAnalyzer
from harnyx.optimization.optimizer import HarnessOptimizer, OptimizationConfig
from harnyx.sandbox.isolation import SubprocessSandbox
from harnyx.sandbox.limits import SandboxLimits
from harnyx.sandbox.runner import LocalSandbox


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    handler = args.handler
    try:
        return int(handler(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130
    except HarnyxError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harnyx", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="deterministic end-to-end toy demo")
    run.add_argument("--run-dir", default="runs", help="directory for run artifacts")
    run.add_argument("--candidates", type=int, default=3)
    run.add_argument("--iterations", type=int, default=1)
    run.set_defaults(handler=_cmd_run)

    evaluate = sub.add_parser("evaluate", help="evaluate an agent (toy or plugin)")
    evaluate.add_argument("--plugin", help="module:factory returning {'agent','tasks',...}")
    evaluate.add_argument("--config", help="Harnyx config JSON/YAML file")
    evaluate.add_argument("--benchmark", default=None)
    evaluate.add_argument("--output", help="write the EvaluationResult JSON here")
    evaluate.set_defaults(handler=_cmd_evaluate)

    optimize = sub.add_parser("optimize", help="run the harness optimization loop")
    optimize.add_argument("--plugin", help="module:factory returning {'agent','tasks',...}")
    optimize.add_argument("--config", help="Harnyx config JSON/YAML file")
    optimize.add_argument("--base-url", help="OpenAI-compatible engineer endpoint")
    optimize.add_argument("--model", help="engineer model name")
    optimize.add_argument("--api-key-env", default=None)
    optimize.add_argument("--candidates", type=int, default=None)
    optimize.add_argument("--iterations", type=int, default=None)
    optimize.add_argument("--run-dir", default=None)
    optimize.add_argument("--allow-regressions", action="store_true")
    optimize.set_defaults(handler=_cmd_optimize)

    gf = sub.add_parser("generate-failures", help="build a failure packet from trajectory JSONL")
    gf.add_argument("--trajectories", required=True, help="trajectory JSONL path")
    gf.add_argument("--benchmark", required=True)
    gf.add_argument("--batch-id", default="batch")
    gf.add_argument("--reward-threshold", type=float, default=1.0)
    gf.add_argument("--max-traces", type=int, default=20)
    gf.add_argument("--strategy", choices=["round_robin", "lowest_reward", "first"], default="round_robin")
    gf.add_argument("--output", required=True, help="output packet JSON path")
    gf.set_defaults(handler=_cmd_generate_failures)

    gp = sub.add_parser("generate-patches", help="generate candidate patches for a packet")
    gp.add_argument("--packet", required=True, help="failure packet JSON path")
    gp.add_argument("--base-url", required=True)
    gp.add_argument("--model", required=True)
    gp.add_argument("--api-key-env", default="HARNYX_ENGINEER_API_KEY")
    gp.add_argument("--num-candidates", type=int, default=8)
    gp.add_argument("--temperature", type=float, default=0.7)
    gp.add_argument("--output", required=True, help="candidate JSONL path")
    gp.set_defaults(handler=_cmd_generate_patches)

    it = sub.add_parser("inspect-trajectory", help="pretty-print a trajectory JSONL")
    it.add_argument("path", help="trajectory JSONL path")
    it.add_argument("--task-id", help="only show this task id")
    it.set_defaults(handler=_cmd_inspect_trajectory)

    return parser


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def _cmd_run(args: argparse.Namespace) -> int:
    from harnyx.demo.toy import run_demo

    summary = run_demo(args.run_dir, candidates=args.candidates, iterations=args.iterations)
    print("Harnyx toy demo")
    print(f"  research run directory : {summary['run_dir']}")
    print(f"  baseline success       : {summary['baseline_success']}/1  (mean reward {summary['baseline_reward']:.3f})")
    print(f"  patched success        : {summary['patched_success']}/1  (mean reward {summary['patched_reward']:.3f})")
    print(f"  engineer reward        : {summary['reward']:+.3f}")
    print(f"  accepted versions      : {', '.join(summary['accepted']) or 'none'}")
    return 0 if summary["patched_reward"] > summary["baseline_reward"] else 1


def _cmd_evaluate(args: argparse.Namespace) -> int:
    config = _load_config_arg(args)
    default_benchmark = args.benchmark or config.evaluation.benchmark or "toy"
    agent, tasks, benchmark = _resolve_agent_and_tasks(args.plugin, default_benchmark=default_benchmark)
    evaluator = LocalEvaluator(benchmark=benchmark, trajectory_dir=config.evaluation.trajectory_dir)
    result = evaluator.evaluate(agent, tasks)
    _print_evaluation(result)
    if args.output:
        write_json(args.output, result.to_dict())
        print(f"wrote {args.output}")
    return 0


def _cmd_optimize(args: argparse.Namespace) -> int:
    config = _load_config_arg(args)
    default_benchmark = config.evaluation.benchmark or "toy"
    agent, tasks, benchmark = _resolve_agent_and_tasks(args.plugin, default_benchmark=default_benchmark)
    engineer = _build_engineer(args, benchmark, plugin_spec=args.plugin, config=config)
    sandbox = _build_sandbox(config.sandbox)
    validator = None
    if config.optimization.smoke_test:
        validator = PatchValidator(smoke_sandbox=SubprocessSandbox(sandbox.limits))
    opt = config.optimization
    opt_config = OptimizationConfig(
        candidates=args.candidates if args.candidates is not None else opt.candidates,
        iterations=args.iterations if args.iterations is not None else opt.iterations,
        reward_metric=opt.reward_metric,
        valid_bonus=opt.valid_bonus,
        accept_threshold=opt.accept_threshold,
        reject_regressions=opt.reject_regressions,
        allow_regressions=args.allow_regressions or opt.allow_regressions,
        max_traces=opt.max_traces,
        selection_strategy=opt.selection_strategy,
        reward_threshold=opt.reward_threshold,
        patch_cache=opt.patch_cache,
        accept_first_valid=opt.accept_first_valid,
    )
    optimizer = HarnessOptimizer(
        agent,
        engineer,
        evaluator=LocalEvaluator(benchmark=benchmark, trajectory_dir=config.evaluation.trajectory_dir),
        sandbox=sandbox,
        validator=validator,
        benchmark=benchmark,
        model=config.engineer.model or (args.model or ""),
        config=opt_config,
        run_dir=args.run_dir or config.run_dir,
    )
    result = optimizer.optimize(tasks)
    print("Harnyx optimize")
    print(f"  run directory   : {result.run_dir}")
    print(f"  baseline reward : {result.baseline.mean_reward:.4f}  success {result.baseline.num_success}/{result.baseline.n}")
    print(f"  final reward    : {result.final.mean_reward:.4f}  success {result.final.num_success}/{result.final.n}")
    print(f"  accepted        : {', '.join(v.version for v in result.accepted_versions) or 'none'}")
    for report in result.iterations:
        print(
            f"  iter {report.iteration}: failures={report.num_failures} candidates={report.num_candidates} "
            f"valid={report.num_valid} accepted={report.accepted} ({report.reason})"
        )
    return 0 if result.improved or not result.accepted_versions else 1


def _cmd_generate_failures(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.trajectories)
    if not rows:
        print(f"no trajectories found in {args.trajectories}", file=sys.stderr)
        return 2
    trajectories = [Trajectory.from_dict(row) for row in rows]
    analyzer = TraceFailureAnalyzer(
        benchmark=args.benchmark,
        reward_threshold=args.reward_threshold,
        max_traces=args.max_traces,
        strategy=args.strategy,
    )
    packet = analyzer.analyze(trajectories, batch_id=args.batch_id)
    write_json(args.output, packet.to_dict())
    print(f"wrote {args.output}: {len(packet.cases)} failure case(s) from {len(trajectories)} trajectory(ies)")
    return 0


def _cmd_generate_patches(args: argparse.Namespace) -> int:
    packet = FailurePacket.from_dict(read_json(args.packet))
    provider = OpenAICompatibleProvider(
        base_url=args.base_url,
        model=args.model,
        env_key=args.api_key_env,
        default_temperature=args.temperature,
    )
    engineer = LLMHarnessEngineer(provider, benchmark=packet.benchmark, temperature=args.temperature)
    candidates = engineer.generate_candidates(packet, args.num_candidates)
    write_jsonl(
        args.output,
        [{"parse_ok": True, "patch": candidate.to_dict(), "hooks": list(candidate.hook_names)} for candidate in candidates],
    )
    print(f"wrote {args.output}: {len(candidates)} candidate patch(es)")
    return 0


def _cmd_inspect_trajectory(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.path)
    if not rows:
        print(f"no trajectories found in {args.path}", file=sys.stderr)
        return 2
    for row in rows:
        trajectory = Trajectory.from_dict(row)
        if args.task_id and trajectory.task_id != args.task_id:
            continue
        print(f"task={trajectory.task_id} status={trajectory.status} success={trajectory.success} reward={trajectory.final_reward:.4f}")
        for step in trajectory.steps:
            action = (step.action or {}).get("raw") or (step.action or {}).get("name", "")
            print(f"  [{step.index}] action={action!r}")
            if step.tool_result:
                print(f"        observation={step.tool_result[:200]!r}")
            for effect in step.harness_effects:
                print(f"        harness_effect={effect}")
        print()
    return 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve_agent_and_tasks(plugin: str | None, *, default_benchmark: str) -> tuple[Any, list[Task], str]:
    if plugin:
        payload = _load_factory(plugin)
        if isinstance(payload, dict):
            return payload["agent"], list(payload["tasks"]), str(payload.get("benchmark", default_benchmark))
        agent, tasks = payload
        return agent, list(tasks), default_benchmark
    from harnyx.demo.toy import build_toy_agent, build_toy_tasks

    return build_toy_agent(), build_toy_tasks(), "toy"


def _load_config_arg(args: argparse.Namespace) -> HarnyxConfig:
    path = getattr(args, "config", None)
    return load_config(path) if path else HarnyxConfig()


def _build_sandbox(config: SandboxConfig) -> LocalSandbox | SubprocessSandbox:
    limits = SandboxLimits(
        time_budget_s=config.hook_time_budget_ms / 1000.0,
        line_budget=config.line_budget,
        wall_timeout_s=config.wall_timeout_seconds,
        memory_mb=config.memory_mb,
        cpu_seconds=config.cpu_seconds,
    )
    if config.backend == "subprocess":
        return SubprocessSandbox(limits)
    return LocalSandbox(limits)


def _build_engineer(
    args: argparse.Namespace,
    benchmark: str,
    *,
    plugin_spec: str | None,
    config: HarnyxConfig,
) -> Any:
    if plugin_spec:
        payload = _load_factory(plugin_spec)
        if isinstance(payload, dict) and payload.get("engineer") is not None:
            return payload["engineer"]
    engineer_cfg = config.engineer
    base_url = getattr(args, "base_url", None) or engineer_cfg.base_url
    model = getattr(args, "model", None) or engineer_cfg.model
    if base_url and model:
        provider = OpenAICompatibleProvider(
            base_url=base_url,
            model=model,
            env_key=getattr(args, "api_key_env", None) or engineer_cfg.api_key_env,
            default_temperature=engineer_cfg.temperature,
            default_max_tokens=engineer_cfg.max_tokens,
        )
        return LLMHarnessEngineer(
            provider,
            benchmark=benchmark,
            temperature=engineer_cfg.temperature,
            max_tokens=engineer_cfg.max_tokens,
            prefill_think=engineer_cfg.prefill_think,
            require_think=engineer_cfg.require_think,
            include_response_template=engineer_cfg.include_response_template,
        )
    # Default to the deterministic scripted engineer for the toy benchmark.
    from harnyx.demo.toy import TOY_BENCHMARK, build_toy_patch_text
    from harnyx.engineering.harness_engineer import ScriptedHarnessEngineer
    from harnyx.engineering.patch import extract_patch

    if benchmark == TOY_BENCHMARK:
        patch = extract_patch(build_toy_patch_text(), benchmark=TOY_BENCHMARK, require_think=True, prefill_think=True)
        return ScriptedHarnessEngineer([patch])
    raise SystemExit("provide --base-url/--model or a plugin supplying an engineer")


def _load_factory(spec: str) -> Any:
    module_name, _, attr = spec.partition(":")
    if not module_name:
        raise SystemExit(f"invalid factory spec: {spec!r}")
    sys.path.insert(0, os.getcwd())
    module = importlib.import_module(module_name)
    factory = getattr(module, attr or "build")
    return factory()


def _print_evaluation(result: EvaluationResult) -> None:
    print(f"agent={result.agent_name} benchmark={result.benchmark} harness={result.harness_version}")
    print(f"  n={result.n} mean_reward={result.mean_reward:.4f} success_rate={result.success_rate:.4f} "
          f"({result.num_success}/{result.n})")
    if result.errors:
        print(f"  errors: {json.dumps(result.errors, sort_keys=True)}")


if __name__ == "__main__":
    raise SystemExit(main())
