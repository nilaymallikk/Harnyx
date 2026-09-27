from __future__ import annotations

from nova.core.result import EvaluationResult
from nova.demo.toy import TOY_BENCHMARK, build_toy_agent, build_toy_patch_text, build_toy_tasks, run_demo
from nova.engineering.harness_engineer import ScriptedHarnessEngineer
from nova.engineering.patch import CodeHook, HarnessPatch, extract_patch
from nova.engineering.validation import ValidationResult
from nova.optimization.optimizer import HarnessOptimizer, OptimizationConfig
from nova.optimization.patch_generation import CandidatePatch
from nova.optimization.reward import RewardResult
from nova.optimization.selection import CandidateSelector, ScoredCandidate


def _toy_patch() -> HarnessPatch:
    return extract_patch(build_toy_patch_text(), benchmark=TOY_BENCHMARK, require_think=True, prefill_think=True)


def test_toy_demo_end_to_end(tmp_path) -> None:
    summary = run_demo(tmp_path, candidates=3, iterations=1)
    assert summary["baseline_reward"] == 0.0
    assert summary["patched_reward"] == 1.0
    assert summary["reward"] == 1.0
    assert summary["accepted"] == ["harness-v1"]


def test_optimizer_writes_run_artifacts(tmp_path) -> None:
    summary = run_demo(tmp_path, candidates=2, iterations=1)
    run_dir = summary["run_dir"]
    from pathlib import Path

    root = Path(run_dir)
    for name in (
        "config.json",
        "baseline.json",
        "failures.jsonl",
        "candidates.jsonl",
        "rewards.jsonl",
        "accepted_patch.json",
        "report.json",
    ):
        assert (root / name).exists(), name
    versions = list((root / "harness").glob("harness-v*.json"))
    assert versions


def test_optimizer_accepts_no_patch_when_engineer_cannot_improve(tmp_path) -> None:
    noop = HarnessPatch(
        benchmark=TOY_BENCHMARK,
        hooks=(CodeHook(hook="on_before_action", code="def hook(ctx, nb):\n    return None\n"),),
    )
    optimizer = HarnessOptimizer(
        build_toy_agent(),
        ScriptedHarnessEngineer([noop]),
        benchmark=TOY_BENCHMARK,
        config=OptimizationConfig(candidates=1, iterations=1),
        run_dir=tmp_path,
    )
    result = optimizer.optimize(build_toy_tasks())
    assert result.accepted_versions == []
    assert result.iterations[0].accepted is False


def _scored(index: int, reward: float, regressions: list[str], successes: dict[str, bool]) -> ScoredCandidate:
    patch = HarnessPatch(
        benchmark="toy",
        hooks=(CodeHook(hook="on_before_action", code="def hook(ctx, nb):\n    return None\n"),),
    )
    candidate = CandidatePatch(index=index, patch=patch, validation=ValidationResult(ok=True, patch=patch))
    reward_result = RewardResult(reward=reward, valid=True, regressions=regressions)
    patched = EvaluationResult(rewards={k: float(v) for k, v in successes.items()}, successes=successes)
    return ScoredCandidate(candidate=candidate, reward=reward_result, patched=patched)


def test_selector_rejects_regression() -> None:
    selector = CandidateSelector(reject_regressions=True)
    good = _scored(0, 0.5, [], {"a": True, "b": True})
    regressive = _scored(1, 0.8, ["b"], {"a": True, "b": False})
    result = selector.select([good, regressive])
    assert result.accepted is True
    assert result.best is not None
    assert result.best.candidate.index == 0
    assert any(item["reason"] == "regression_detected" for item in result.rejected)


def test_selector_regression_suite_protects_solved_tasks() -> None:
    selector = CandidateSelector(reject_regressions=True, regression_suite={"b"})
    candidate = _scored(0, 0.5, [], {"a": True, "b": False})
    result = selector.select([candidate])
    assert result.accepted is False
    assert result.rejected[0]["reason"] == "regression_suite_failed"


def test_selector_rejects_no_improvement() -> None:
    selector = CandidateSelector(min_reward=0.0)
    result = selector.select([_scored(0, 0.0, [], {"a": True})])
    assert result.accepted is False
    assert result.rejected[0]["reason"] == "no_improvement"


def test_reward_is_deterministic_across_runs(tmp_path) -> None:
    first = run_demo(tmp_path / "a", candidates=2, iterations=1)
    second = run_demo(tmp_path / "b", candidates=2, iterations=1)
    assert first["reward"] == second["reward"] == 1.0
    assert first["patched_reward"] == second["patched_reward"] == 1.0


def test_version_store_rollback(tmp_path) -> None:
    from nova.versioning import HarnessVersionStore

    store = HarnessVersionStore(tmp_path / "versions")
    v1 = store.record(patch={"x": 1}, reward=1.0)
    v2 = store.record(patch={"x": 2}, reward=2.0)
    assert v1.version == "harness-v1"
    assert v2.version == "harness-v2"
    assert v2.parent == "harness-v1"
    assert store.current() == "harness-v2"
    rolled = store.rollback("harness-v1")
    assert rolled.version == "harness-v1"
    assert store.current() == "harness-v1"
    assert store.latest().version == "harness-v2"
