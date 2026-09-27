"""Ablation studies for the Harness-R1 recipe.

Variants
--------
A  baseline agent (no harness edit)
B  random harness patches (safe templates) + outcome selection
C  proposed patches accepted without outcome feedback (first valid)
D  outcome-based selection (the Harness-R1 acceptance rule)
E  full Harness-R1 (trained engineer + outcome selection); pass the trained
   engineer via ``engineer=`` or ``llm_engineer=``
F  Harnyx extension (D + patch cache + failure clustering + regression protection)

These run on any ``agent``/``tasks`` the caller supplies. On the deterministic
toy benchmark they complete without a model; with a real engine/agent they measure
the paper's ablation questions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harnyx.core.agent import Agent
from harnyx.core.task import Task
from harnyx.core.types import to_jsonable, write_json
from harnyx.engineering.harness_engineer import HarnessEngineer
from harnyx.engineering.random_engineer import RandomHarnessEngineer
from harnyx.evaluation.evaluator import Evaluator, LocalEvaluator
from harnyx.optimization.optimizer import HarnessOptimizer, OptimizationConfig


@dataclass(frozen=True, slots=True)
class AblationSpec:
    """Static description of an ablation arm."""

    key: str
    name: str
    description: str


ABLATIONS: dict[str, AblationSpec] = {
    "A": AblationSpec("A", "baseline", "No harness edit; unmodified target."),
    "B": AblationSpec("B", "random_patches", "Random safe-hook patches, outcome-selected."),
    "C": AblationSpec("C", "no_outcome_feedback", "First valid proposal accepted regardless of reward."),
    "D": AblationSpec("D", "outcome_selection", "Best patch by realized same-batch reward."),
    "E": AblationSpec("E", "full_harness_r1", "Trained engineer + outcome selection."),
    "F": AblationSpec("F", "harnyx_extension", "D + patch cache + failure clustering + regression guard."),
}


@dataclass(slots=True)
class AblationResult:
    """Measured result of one ablation arm."""

    key: str
    name: str
    baseline_reward: float
    baseline_success: float
    final_reward: float
    final_success: float
    reward: float
    accepted_versions: list[str] = field(default_factory=list)
    num_candidates: int = 0
    num_evaluated: int = 0
    run_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def _config_for(key: str) -> OptimizationConfig:
    base = OptimizationConfig(candidates=4, iterations=1)
    if key == "B":
        return base
    if key == "C":
        return OptimizationConfig(
            candidates=4, iterations=1, accept_threshold=float("-inf"), reject_regressions=False, accept_first_valid=True
        )
    if key == "D":
        return base
    if key == "E":
        return OptimizationConfig(candidates=8, iterations=1)
    if key == "F":
        return OptimizationConfig(candidates=8, iterations=1, patch_cache=True, cluster_failures=True)
    raise ValueError(f"unknown ablation key: {key!r}")


def run_ablation(
    key: str,
    agent: Agent,
    tasks: Sequence[Task],
    *,
    benchmark: str,
    run_root: str | Path,
    engineer: HarnessEngineer | None = None,
    evaluator: Evaluator | None = None,
) -> AblationResult:
    """Run one ablation arm and return its measured summary."""
    if key not in ABLATIONS:
        raise ValueError(f"unknown ablation key: {key!r}")
    spec = ABLATIONS[key]
    evaluator = evaluator or LocalEvaluator(benchmark=benchmark)
    run_dir = Path(run_root) / f"ablation-{key.lower()}"

    if key == "A":
        baseline = evaluator.evaluate(agent, list(tasks))
        return AblationResult(
            key=key,
            name=spec.name,
            baseline_reward=baseline.mean_reward,
            baseline_success=baseline.success_rate,
            final_reward=baseline.mean_reward,
            final_success=baseline.success_rate,
            reward=0.0,
            run_dir=None,
        )

    active_engineer: HarnessEngineer
    if key == "B":
        active_engineer = RandomHarnessEngineer(benchmark=benchmark, seed=0)
    elif engineer is not None:
        active_engineer = engineer
    else:
        active_engineer = _default_engineer(benchmark)

    optimizer = HarnessOptimizer(
        agent,
        active_engineer,
        evaluator=evaluator,
        benchmark=benchmark,
        config=_config_for(key),
        run_dir=run_dir,
    )
    result = optimizer.optimize(list(tasks))
    num_candidates = sum(report.num_candidates for report in result.iterations)
    return AblationResult(
        key=key,
        name=spec.name,
        baseline_reward=result.baseline.mean_reward,
        baseline_success=result.baseline.success_rate,
        final_reward=result.final.mean_reward,
        final_success=result.final.success_rate,
        reward=result.final.mean_reward - result.baseline.mean_reward,
        accepted_versions=[version.version for version in result.accepted_versions],
        num_candidates=num_candidates,
        num_evaluated=sum(report.num_evaluated for report in result.iterations),
        run_dir=result.run_dir,
    )


def run_all_ablations(
    agent: Agent,
    tasks: Sequence[Task],
    *,
    benchmark: str,
    run_root: str | Path,
    engineer: HarnessEngineer | None = None,
    evaluator: Evaluator | None = None,
    keys: Sequence[str] = ("A", "B", "C", "D", "E", "F"),
) -> list[AblationResult]:
    """Run several ablation arms and write a combined report."""
    results = [
        run_ablation(key, agent, tasks, benchmark=benchmark, run_root=run_root, engineer=engineer, evaluator=evaluator)
        for key in keys
    ]
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "ablations.json", {"benchmark": benchmark, "results": [r.to_dict() for r in results]})
    return results


def _default_engineer(benchmark: str) -> HarnessEngineer:
    """Fall back to the toy scripted engineer when no engineer is supplied."""
    from harnyx.demo.toy import TOY_BENCHMARK, build_toy_patch_text
    from harnyx.engineering.harness_engineer import ScriptedHarnessEngineer
    from harnyx.engineering.patch import extract_patch

    if benchmark == TOY_BENCHMARK:
        patch = extract_patch(build_toy_patch_text(), benchmark=TOY_BENCHMARK, require_think=True, prefill_think=True)
        return ScriptedHarnessEngineer([patch])
    raise ValueError("provide engineer= for non-toy benchmarks")
