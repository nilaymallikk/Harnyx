"""Outcome-based reward for harness patches.

The Harness-R1 reward is *not* a learned judge. It is the realized change in
full-batch task performance after rerunning the frozen target on exactly the
same task identities (paper Eq. 1):

    Δ_B(P) = mean_i( R_i^P - R_i^0 )
    r(B, P) = Δ_B(P)  if the patch is valid and evaluation is complete
              0        otherwise

Because the reward is outcome-grounded, an invalid patch, a behaviorally inert
patch, or an incomplete evaluation scores zero — never a fabricated positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from nova.core.result import EvaluationResult
from nova.core.types import to_jsonable


@dataclass(slots=True)
class RewardResult:
    """Reward for one candidate patch on one task batch."""

    reward: float
    valid: bool
    baseline_mean: float = 0.0
    patched_mean: float = 0.0
    delta: float = 0.0
    task_deltas: dict[str, float] = field(default_factory=dict)
    improvements: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    reason: str = "ok"
    metric: str = "delta_average_reward"
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@runtime_checkable
class RewardFunction(Protocol):
    """Structural interface for harness-patch reward functions."""

    def score(
        self,
        baseline: EvaluationResult,
        patched: EvaluationResult,
        *,
        task_ids: list[str] | None = None,
        valid: bool = True,
    ) -> RewardResult: ...


class OutcomeReward:
    """Deterministic same-batch outcome reward (Harness-R1 Eq. 1)."""

    def __init__(
        self,
        *,
        metric: str = "delta_average_reward",
        valid_bonus: float = 0.0,
        require_complete: bool = True,
        require_same_tasks: bool = True,
    ) -> None:
        if metric not in {"delta_average_reward", "delta_success_rate"}:
            raise ValueError("metric must be 'delta_average_reward' or 'delta_success_rate'")
        self.metric = metric
        self.valid_bonus = valid_bonus
        self.require_complete = require_complete
        self.require_same_tasks = require_same_tasks

    def score(
        self,
        baseline: EvaluationResult,
        patched: EvaluationResult,
        *,
        task_ids: list[str] | None = None,
        valid: bool = True,
    ) -> RewardResult:
        ids = list(task_ids) if task_ids is not None else sorted(baseline.rewards)
        if not ids:
            return RewardResult(reward=0.0, valid=True, reason="empty_batch", metric=self.metric)

        if not valid:
            return self._zero(baseline, ids, reason="invalid_patch")

        if getattr(patched, "harness_noop", False):
            return self._zero(baseline, ids, reason="runtime_noop")

        missing = [task_id for task_id in ids if task_id not in patched.rewards]
        if self.require_complete and (missing or patched.errors):
            return self._zero(
                baseline,
                ids,
                reason="incomplete_eval",
                detail={"missing": missing[:10], "errors": sorted(patched.errors)[:10]},
            )
        if self.require_same_tasks and missing:
            return self._zero(baseline, ids, reason="task_mismatch", detail={"missing": missing[:10]})

        comparable = [task_id for task_id in ids if task_id in baseline.rewards and task_id in patched.rewards]
        if not comparable:
            return self._zero(baseline, ids, reason="no_comparable_tasks")

        task_deltas = {
            task_id: float(patched.rewards[task_id]) - float(baseline.rewards[task_id]) for task_id in comparable
        }
        baseline_mean = sum(float(baseline.rewards[t]) for t in comparable) / len(comparable)
        patched_mean = sum(float(patched.rewards[t]) for t in comparable) / len(comparable)

        if self.metric == "delta_success_rate":
            base_rate = sum(1 for t in comparable if baseline.successes.get(t)) / len(comparable)
            patched_rate = sum(1 for t in comparable if patched.successes.get(t)) / len(comparable)
            reward = patched_rate - base_rate
        else:
            reward = patched_mean - baseline_mean

        improvements = [t for t in comparable if task_deltas[t] > 0]
        regressions = [
            t
            for t in comparable
            if task_deltas[t] < 0 and bool(baseline.successes.get(t, baseline.rewards[t] >= 1.0))
        ]
        unchanged = [t for t in comparable if task_deltas[t] == 0]

        return RewardResult(
            reward=reward + (self.valid_bonus if valid else 0.0),
            valid=True,
            baseline_mean=baseline_mean,
            patched_mean=patched_mean,
            delta=reward,
            task_deltas=task_deltas,
            improvements=improvements,
            regressions=regressions,
            unchanged=unchanged,
            reason="ok",
            metric=self.metric,
        )

    @staticmethod
    def _zero(
        baseline: EvaluationResult,
        task_ids: list[str],
        *,
        reason: str,
        detail: dict[str, Any] | None = None,
    ) -> RewardResult:
        present = [t for t in task_ids if t in baseline.rewards]
        baseline_mean = sum(float(baseline.rewards[t]) for t in present) / max(1, len(present))
        return RewardResult(
            reward=0.0,
            valid=False,
            baseline_mean=baseline_mean,
            patched_mean=baseline_mean,
            delta=0.0,
            reason=reason,
            detail=dict(detail or {}),
        )
