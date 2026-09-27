"""The outcome-grounded harness optimization loop.

This is the inference/evaluation half of Harness-R1: given a frozen agent and an
evaluator, it mines a failure packet, samples candidate patches, validates and
sandboxes them, reruns the same task batch, computes the same-batch outcome
reward, and accepts a patch only if it improves without regressions. It is
usable with a pretrained engineer and does not require training.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from harnyx.core.agent import Agent
from harnyx.core.harness import ExecutableHarness, Harness
from harnyx.core.result import EvaluationResult
from harnyx.core.task import Task
from harnyx.core.types import to_jsonable
from harnyx.engineering.harness_engineer import HarnessEngineer
from harnyx.engineering.validation import PatchValidator
from harnyx.evaluation.evaluator import Evaluator, LocalEvaluator
from harnyx.evaluation.reports import RunDirectory
from harnyx.optimization.failure_analysis import FailureAnalyzer, TraceFailureAnalyzer
from harnyx.optimization.patch_generation import CandidatePatch, PatchGenerator
from harnyx.optimization.reward import OutcomeReward, RewardFunction, RewardResult
from harnyx.optimization.selection import CandidateSelector, ScoredCandidate, SelectionResult
from harnyx.sandbox.runner import LocalSandbox
from harnyx.versioning import HarnessVersion, HarnessVersionStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OptimizationConfig:
    """Configuration for one optimization run.

    Defaults mirror the released Harness-R1 reference where a value exists:
    ``candidates`` is the GRPO group size K = 8, and the reward metric is the
    full-batch mean reward change.
    """

    candidates: int = 8
    iterations: int = 1
    reward_metric: str = "delta_average_reward"
    valid_bonus: float = 0.0
    accept_threshold: float = 0.0
    reject_regressions: bool = True
    allow_regressions: bool = False
    max_traces: int = 20
    selection_strategy: str = "round_robin"
    reward_threshold: float = 1.0
    smoke_test: bool = False
    # Harnyx extension: reuse candidate evaluations when the same patch is
    # proposed for the same failure packet (deterministic evaluators).
    patch_cache: bool = True
    # Ablation C: accept the first valid proposal instead of the best outcome.
    accept_first_valid: bool = False
    # Harnyx extension: cluster failures by runtime signature before selection.
    cluster_failures: bool = False

    def __post_init__(self) -> None:
        if self.candidates < 1:
            raise ValueError("candidates must be >= 1")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(slots=True)
class IterationReport:
    """Per-iteration provenance record."""

    iteration: int
    packet_fingerprint: str
    num_failures: int
    num_candidates: int
    num_valid: int
    num_evaluated: int
    accepted: bool
    reason: str
    best_reward: float = 0.0
    accepted_version: str | None = None
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(slots=True)
class OptimizationResult:
    """Full result of an optimization run."""

    baseline: EvaluationResult
    final: EvaluationResult
    iterations: list[IterationReport] = field(default_factory=list)
    accepted_versions: list[HarnessVersion] = field(default_factory=list)
    run_dir: str | None = None
    config: OptimizationConfig | None = None

    @property
    def improved(self) -> bool:
        return self.final.mean_reward > self.baseline.mean_reward

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "config": self.config.to_dict() if self.config else None,
            "baseline": self.baseline.to_dict(),
            "final": self.final.to_dict(),
            "improved": self.improved,
            "iterations": [report.to_dict() for report in self.iterations],
            "accepted_versions": [version.to_dict() for version in self.accepted_versions],
        }


class HarnessOptimizer:
    """Outcome-grounded optimizer for executable harness patches."""

    def __init__(
        self,
        agent: Agent,
        engineer: HarnessEngineer,
        *,
        evaluator: Evaluator | None = None,
        sandbox: LocalSandbox | None = None,
        validator: PatchValidator | None = None,
        reward: RewardFunction | None = None,
        analyzer: FailureAnalyzer | None = None,
        config: OptimizationConfig | None = None,
        run_dir: str | RunDirectory | None = None,
        version_store: HarnessVersionStore | None = None,
        benchmark: str = "",
        model: str = "",
    ) -> None:
        self.agent = agent
        self.engineer = engineer
        self.config = config or OptimizationConfig()
        self.benchmark = benchmark
        self.model = model
        self.sandbox = sandbox or LocalSandbox()
        self.evaluator = evaluator or LocalEvaluator(benchmark=benchmark)
        self.validator = validator or PatchValidator(
            compiler=self.sandbox.compiler if isinstance(self.sandbox, LocalSandbox) else None
        )
        self.reward = reward or OutcomeReward(metric=self.config.reward_metric, valid_bonus=self.config.valid_bonus)
        self.analyzer = analyzer or TraceFailureAnalyzer(
            benchmark=benchmark,
            reward_threshold=self.config.reward_threshold,
            max_traces=self.config.max_traces,
            strategy=self.config.selection_strategy,
        )
        self.generator = PatchGenerator(self.engineer, self.validator)
        self.run: RunDirectory | None = (
            run_dir if isinstance(run_dir, RunDirectory) else (RunDirectory.create(run_dir) if run_dir else None)
        )
        self.version_store = version_store or (HarnessVersionStore(self.run.root / "harness") if self.run else None)
        self._evaluation_cache: dict[str, tuple[RewardResult, EvaluationResult]] = {}

    # ------------------------------------------------------------------ #
    def optimize(self, tasks: Sequence[Task], *, harness: Harness | None = None) -> OptimizationResult:
        """Run the failure -> patch -> rerun -> reward -> accept loop."""
        if not tasks:
            raise ValueError("optimize() requires at least one task")
        task_list = list(tasks)
        task_ids = [task.id for task in task_list]

        if self.run is not None:
            self.run.write_config(self.config.to_dict())

        baseline = self.evaluator.evaluate(self.agent, task_list, harness=harness, harness_version="harness-v0")
        initial_baseline = baseline
        if self.run is not None:
            self.run.write_baseline(baseline.to_dict())

        regression_suite: set[str] = {task_id for task_id, ok in baseline.successes.items() if ok}
        current_harness = harness
        accepted_versions: list[HarnessVersion] = []
        reports: list[IterationReport] = []

        for iteration in range(self.config.iterations):
            packet = self.analyzer.analyze(
                list(baseline.trajectories.values()),
                baseline=baseline,
                batch_id=f"iter{iteration}",
            )
            if self.run is not None:
                self.run.append_failure(packet.to_dict())

            if not packet.cases:
                reports.append(
                    IterationReport(
                        iteration=iteration,
                        packet_fingerprint=packet.fingerprint(),
                        num_failures=0,
                        num_candidates=0,
                        num_valid=0,
                        num_evaluated=0,
                        accepted=False,
                        reason="no_failures",
                    )
                )
                break

            candidates = self.generator.generate(packet, self.config.candidates)
            scored: list[ScoredCandidate] = []
            baseline_version = getattr(current_harness, "version", "harness-v0")

            for candidate in candidates:
                if self.run is not None:
                    self.run.append_candidate(candidate.to_dict())
                if not candidate.is_valid or candidate.patch is None:
                    scored.append(
                        ScoredCandidate(
                            candidate=candidate,
                            reward=_zero_reward(candidate, "invalid_patch"),
                        )
                    )
                    continue
                patch_harness = ExecutableHarness.from_patch(candidate.patch, sandbox=self.sandbox)
                cache_key = f"{packet.fingerprint()}:{candidate.fingerprint()}"
                cached = self._evaluation_cache.get(cache_key) if self.config.patch_cache else None
                if cached is not None:
                    reward_result, patched = cached
                    scored.append(
                        ScoredCandidate(
                            candidate=candidate, reward=reward_result, patched=patched, harness=patch_harness
                        )
                    )
                    continue
                version = f"{baseline_version}+cand{candidate.index}"
                patched = self.evaluator.evaluate(
                    self.agent,
                    task_list,
                    harness=patch_harness,
                    harness_version=version,
                )
                reward_result = self.reward.score(baseline, patched, task_ids=task_ids, valid=not patch_harness.is_noop)
                if self.config.patch_cache:
                    self._evaluation_cache[cache_key] = (reward_result, patched)
                if self.run is not None:
                    self.run.append_reward(
                        {"candidate_index": candidate.index, "reward": reward_result.to_dict()}
                    )
                scored.append(
                    ScoredCandidate(candidate=candidate, reward=reward_result, patched=patched, harness=patch_harness)
                )

            selector = CandidateSelector(
                min_reward=self.config.accept_threshold,
                reject_regressions=self.config.reject_regressions,
                allow_regressions=self.config.allow_regressions,
                regression_suite=regression_suite,
                accept_first_valid=self.config.accept_first_valid,
            )
            selection = selector.select(scored)
            report = self._report(iteration, packet.fingerprint(), len(packet.cases), scored, selection)
            reports.append(report)

            if selection.accepted and selection.best is not None:
                best = selection.best
                version_record = self._record_acceptance(best, packet, baseline, task_ids)
                report.accepted_version = version_record.version if version_record else None
                if version_record is not None:
                    accepted_versions.append(version_record)
                current_harness = best.harness
                baseline = best.patched if best.patched is not None else baseline
                regression_suite.update(best.reward.improvements)
            else:
                logger.info("iteration %d did not accept a patch: %s", iteration, selection.reason)

        result = OptimizationResult(
            baseline=initial_baseline,
            final=baseline,
            iterations=reports,
            accepted_versions=accepted_versions,
            run_dir=str(self.run.root) if self.run else None,
            config=self.config,
        )
        if self.run is not None:
            self.run.write_report(result.to_dict())
        return result

    # ------------------------------------------------------------------ #
    def _report(
        self,
        iteration: int,
        fingerprint: str,
        num_failures: int,
        scored: list[ScoredCandidate],
        selection: SelectionResult,
    ) -> IterationReport:
        best_reward = selection.best.reward.reward if selection.best is not None else 0.0
        return IterationReport(
            iteration=iteration,
            packet_fingerprint=fingerprint,
            num_failures=num_failures,
            num_candidates=len(scored),
            num_valid=sum(1 for item in scored if item.candidate.is_valid),
            num_evaluated=sum(1 for item in scored if item.patched is not None),
            accepted=selection.accepted,
            reason=selection.reason,
            best_reward=best_reward,
            rejected=list(selection.rejected),
        )

    def _record_acceptance(
        self,
        best: ScoredCandidate,
        packet: Any,
        baseline: EvaluationResult,
        task_ids: list[str],
    ) -> HarnessVersion | None:
        patch_dict = best.candidate.patch.to_dict() if best.candidate.patch is not None else {}
        if self.run is not None:
            self.run.write_accepted_patch(
                {
                    "patch": patch_dict,
                    "reward": best.reward.to_dict(),
                    "packet_fingerprint": packet.fingerprint(),
                    "tasks": task_ids,
                }
            )
        if self.version_store is None:
            return None
        return self.version_store.record(
            patch=patch_dict,
            tasks=task_ids,
            baseline_score=baseline.mean_reward,
            new_score=best.patched.mean_reward if best.patched is not None else baseline.mean_reward,
            reward=best.reward.reward,
            model=self.model,
            config=self.config.to_dict(),
            metadata={"packet_fingerprint": packet.fingerprint(), "benchmark": self.benchmark},
        )


def _zero_reward(candidate: CandidatePatch, reason: str) -> RewardResult:
    return RewardResult(reward=0.0, valid=False, reason=reason, detail={"errors": candidate.validation.errors})
