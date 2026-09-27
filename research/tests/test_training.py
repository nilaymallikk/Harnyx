from __future__ import annotations

import json

from research.training.dataset import (
    SFTDatasetBuilder,
    assistant_response_from_patch,
    filter_training_records,
    write_sft_dataset,
)
from research.training.grpo import GRPOConfig, HarnessPatchReward
from research.training.sft import SFTConfig

from harnyx.core.task import Task
from harnyx.core.trajectory import TrajectoryRecorder
from harnyx.demo.toy import TOY_BENCHMARK, build_toy_patch_text
from harnyx.engineering.patch import extract_patch
from harnyx.optimization.failure_analysis import FailurePacket, TraceFailureAnalyzer


def _packet() -> FailurePacket:
    recorder = TrajectoryRecorder(Task(id="f1", instruction="verify then submit"))
    recorder.record_step(observation="pending", action={"name": "submit"}, tool_result="rejected")
    trajectory = recorder.finish(reward=0.0, success=False, status="failed")
    return TraceFailureAnalyzer(benchmark=TOY_BENCHMARK).analyze([trajectory])


def _patch():
    return extract_patch(build_toy_patch_text(), benchmark=TOY_BENCHMARK, require_think=True, prefill_think=True)


def test_sft_example_format(tmp_path) -> None:
    examples = SFTDatasetBuilder().build([(_packet(), _patch())])
    assert len(examples) == 1
    example = examples[0]
    messages = example.to_messages()
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert "harness engineer" in messages[0]["content"].lower()
    assert "<patch>" in messages[2]["content"]
    assert "<think>" in messages[2]["content"]
    assert "add_code_hook" in messages[2]["content"]

    out = tmp_path / "sft.jsonl"
    write_sft_dataset(examples, out)
    row = json.loads(out.read_text().splitlines()[0])
    assert row["messages"][0]["role"] == "system"


def test_assistant_response_reconstructs_patch() -> None:
    text = assistant_response_from_patch(_patch())
    assert text.startswith("<think>")
    assert "<patch>" in text
    assert '"on_before_action"' in text


def test_filter_training_records() -> None:
    packet, patch = _packet(), _patch()
    valid = (packet, patch, {"valid": True, "reward": 0.5})
    invalid = (packet, patch, {"valid": False, "reward": 0.0})
    negative = (packet, patch, {"valid": True, "reward": -1.0})
    kept = filter_training_records([valid, invalid, negative], min_reward=0.0)
    assert len(kept) == 1
    assert kept[0][1] is patch


def test_sft_config_matches_reference() -> None:
    config = SFTConfig()
    assert config.epochs == 2
    assert config.learning_rate == 1e-5
    assert config.context_length == 32768
    assert config.global_batch_size == 24
    assert config.seed == 42


def test_grpo_config_matches_reference() -> None:
    config = GRPOConfig()
    assert config.num_generations == 8
    assert config.rollout_batch_size == 4
    assert config.global_batch_size == 32
    assert config.learning_rate == 1e-6
    assert (config.clip_lower, config.clip_upper) == (0.20, 0.28)
    assert config.max_prompt_length == 28672
    assert config.max_response_length == 12288


def test_harness_patch_reward_bridges_outcomes() -> None:
    calls: list[object] = []

    def evaluate_patch(patch, metadata):
        calls.append(patch)
        return 1.0 if patch is not None else 0.0

    reward = HarnessPatchReward(evaluate_patch)
    rewards = reward([build_toy_patch_text(), "not a patch at all"], benchmark=[TOY_BENCHMARK, TOY_BENCHMARK])
    assert rewards[0] == 1.0
    assert rewards[1] == 0.0
    assert calls[1] is None
