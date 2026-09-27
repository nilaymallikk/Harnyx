"""Outcome-based candidate selection with regression protection.

The engineer's patch is accepted only when a *rerun* confirms a task gain. A
patch that improves the target failure but breaks a previously solved task is
rejected unless regressions are explicitly allowed — the "do no harm" rule that
the paper's lifecycle ablations motivate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nova.core.result import EvaluationResult
from nova.core.types import to_jsonable
from nova.optimization.patch_generation import CandidatePatch
from nova.optimization.reward import RewardResult


@dataclass(slots=True)
class ScoredCandidate:
    """A candidate patch after same-batch rerun evaluation."""

    candidate: CandidatePatch
    reward: RewardResult
    patched: EvaluationResult | None = None
    harness: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "reward": self.reward.to_dict(),
            "patched": self.patched.to_dict() if self.patched is not None else None,
        }


@dataclass(slots=True)
class SelectionResult:
    """Outcome of selecting among scored candidates."""

    accepted: bool
    best: ScoredCandidate | None = None
    reason: str = ""
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "accepted": self.accepted,
                "reason": self.reason,
                "best_index": self.best.candidate.index if self.best is not None else None,
                "best_reward": self.best.reward.reward if self.best is not None else None,
                "rejected": self.rejected,
            }
        )


class CandidateSelector:
    """Select the best non-regressive, improving candidate patch."""

    def __init__(
        self,
        *,
        min_reward: float = 0.0,
        reject_regressions: bool = True,
        allow_regressions: bool = False,
        regression_suite: frozenset[str] | set[str] | None = None,
        accept_first_valid: bool = False,
    ) -> None:
        self.min_reward = min_reward
        self.reject_regressions = reject_regressions
        self.allow_regressions = allow_regressions
        self.regression_suite = frozenset(regression_suite or ())
        # Ablation C: accept the first valid proposal without looking at its
        # realized outcome. This is *not* the Harness-R1 selection rule.
        self.accept_first_valid = accept_first_valid

    def select(self, scored: list[ScoredCandidate]) -> SelectionResult:
        """Return the accepted candidate and a reason for every rejection."""
        rejected: list[dict[str, Any]] = []
        eligible: list[ScoredCandidate] = []

        for item in scored:
            index = item.candidate.index
            if not item.candidate.is_valid:
                rejected.append({"index": index, "reason": "invalid_patch", "detail": item.candidate.validation.errors})
                continue
            if item.candidate.duplicate_of is not None:
                rejected.append({"index": index, "reason": "duplicate", "duplicate_of": item.candidate.duplicate_of})
                continue
            if not item.reward.valid:
                rejected.append({"index": index, "reason": item.reward.reason})
                continue
            if item.reward.reward <= self.min_reward:
                rejected.append({"index": index, "reason": "no_improvement", "reward": item.reward.reward})
                continue
            regression_reason = self._regression_reason(item)
            if regression_reason is not None:
                rejected.append({"index": index, "reason": regression_reason, "regressions": item.reward.regressions})
                continue
            eligible.append(item)

        if not eligible:
            return SelectionResult(accepted=False, best=None, reason="no_acceptable_candidate", rejected=rejected)

        if self.accept_first_valid:
            best = min(eligible, key=lambda item: item.candidate.index)
            return SelectionResult(accepted=True, best=best, reason="accepted_first_valid", rejected=rejected)
        best = max(eligible, key=lambda item: item.reward.reward)
        return SelectionResult(accepted=True, best=best, reason="accepted", rejected=rejected)

    def _regression_reason(self, item: ScoredCandidate) -> str | None:
        if not self.reject_regressions or self.allow_regressions:
            return None
        if item.reward.regressions:
            return "regression_detected"
        if item.patched is None or not self.regression_suite:
            return None
        suite = [task_id for task_id in self.regression_suite if task_id in item.patched.successes]
        failed = [task_id for task_id in suite if not item.patched.successes.get(task_id, False)]
        return "regression_suite_failed" if failed else None
