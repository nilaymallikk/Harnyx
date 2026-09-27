from __future__ import annotations

from nova.core.result import EvaluationResult
from nova.optimization.reward import OutcomeReward


def baseline() -> EvaluationResult:
    return EvaluationResult(
        rewards={"a": 0.0, "b": 1.0, "c": 0.5},
        successes={"a": False, "b": True, "c": False},
    )


def test_delta_average_reward() -> None:
    patched = EvaluationResult(
        rewards={"a": 1.0, "b": 1.0, "c": 0.5},
        successes={"a": True, "b": True, "c": False},
    )
    result = OutcomeReward().score(baseline(), patched, task_ids=["a", "b", "c"])
    assert result.valid is True
    # baseline mean = (0 + 1 + 0.5)/3 = 0.5; patched mean = (1 + 1 + 0.5)/3 = 0.8333
    assert result.reward == pytest_approx(1 / 3)
    assert result.task_deltas["a"] == 1.0
    assert result.improvements == ["a"]


def test_delta_success_rate_metric() -> None:
    patched = EvaluationResult(
        rewards={"a": 1.0, "b": 1.0, "c": 0.5},
        successes={"a": True, "b": True, "c": False},
    )
    result = OutcomeReward(metric="delta_success_rate").score(baseline(), patched, task_ids=["a", "b", "c"])
    assert result.reward == pytest_approx(1 / 3)


def test_invalid_patch_scores_zero() -> None:
    result = OutcomeReward().score(baseline(), baseline(), valid=False)
    assert result.reward == 0.0
    assert result.valid is False
    assert result.reason == "invalid_patch"


def test_incomplete_evaluation_scores_zero() -> None:
    patched = EvaluationResult(rewards={"a": 1.0}, successes={"a": True}, errors={"b": "boom"})
    result = OutcomeReward().score(baseline(), patched, task_ids=["a", "b", "c"])
    assert result.reward == 0.0
    assert result.reason == "incomplete_eval"


def test_missing_task_scores_zero() -> None:
    patched = EvaluationResult(rewards={"a": 1.0}, successes={"a": True})
    result = OutcomeReward().score(baseline(), patched, task_ids=["a", "b", "c"])
    assert result.reward == 0.0
    assert result.reason == "incomplete_eval"


def test_regressions_are_reported_for_previously_successful_tasks() -> None:
    patched = EvaluationResult(
        rewards={"a": 1.0, "b": 0.0, "c": 0.5},
        successes={"a": True, "b": False, "c": False},
    )
    result = OutcomeReward().score(baseline(), patched, task_ids=["a", "b", "c"])
    assert "b" in result.regressions
    assert "a" in result.improvements


def test_noop_harness_scores_zero() -> None:
    patched = EvaluationResult(rewards={"a": 1.0}, successes={"a": True}, metadata={"harness_noop": True})
    result = OutcomeReward().score(baseline(), patched, task_ids=["a"])
    assert result.reward == 0.0
    assert result.reason == "runtime_noop"


def test_valid_bonus_added() -> None:
    patched = EvaluationResult(rewards={"a": 1.0, "b": 1.0, "c": 0.5}, successes={"a": True, "b": True, "c": False})
    result = OutcomeReward(valid_bonus=0.1).score(baseline(), patched, task_ids=["a", "b", "c"])
    assert result.reward == pytest_approx(1 / 3 + 0.1)
    assert result.delta == pytest_approx(1 / 3)


from pytest import approx as pytest_approx  # noqa: E402
