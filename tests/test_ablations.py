from __future__ import annotations

from nova.demo.toy import TOY_BENCHMARK, build_toy_agent, build_toy_tasks
from nova.engineering.random_engineer import RandomHarnessEngineer
from nova.engineering.validation import PatchValidator
from nova.experiments.ablations import ABLATIONS, run_ablation, run_all_ablations


def test_random_engineer_produces_valid_patches() -> None:
    engineer = RandomHarnessEngineer(benchmark=TOY_BENCHMARK, seed=7)
    patches = engineer.generate_candidates(None, 10)
    assert len(patches) == 10
    validator = PatchValidator()
    for patch in patches:
        result = validator.validate(patch)
        assert result.ok, result.errors
        assert len(patch.hook_names) == len(set(patch.hook_names))


def test_ablation_keys_defined() -> None:
    assert set(ABLATIONS) == {"A", "B", "C", "D", "E", "F"}


def test_ablation_baseline_and_outcome(tmp_path) -> None:
    baseline = run_ablation("A", build_toy_agent(), build_toy_tasks(), benchmark=TOY_BENCHMARK, run_root=tmp_path)
    assert baseline.reward == 0.0
    assert baseline.final_reward == 0.0

    outcome = run_ablation("D", build_toy_agent(), build_toy_tasks(), benchmark=TOY_BENCHMARK, run_root=tmp_path)
    assert outcome.reward == 1.0
    assert outcome.accepted_versions


def test_run_all_ablations_writes_report(tmp_path) -> None:
    results = run_all_ablations(
        build_toy_agent(), build_toy_tasks(), benchmark=TOY_BENCHMARK, run_root=tmp_path
    )
    assert {r.key for r in results} == {"A", "B", "C", "D", "E", "F"}
    assert (tmp_path / "ablations.json").exists()
    baseline = next(r for r in results if r.key == "A")
    assert baseline.reward == 0.0
    full = next(r for r in results if r.key == "D")
    assert full.reward == 1.0
