from __future__ import annotations

from harnyx.core.result import EvaluationResult
from harnyx.core.task import Task
from harnyx.core.trajectory import Trajectory, TrajectoryRecorder
from harnyx.optimization.failure_analysis import (
    FailurePacket,
    TraceFailureAnalyzer,
    compute_signals,
    sanitize,
    select_cases,
)


def _trajectory(task_id: str, reward: float, success: bool, actions: list[str], observation: str) -> Trajectory:
    recorder = TrajectoryRecorder(Task(id=task_id, instruction=f"instruction for {task_id}"))
    for action in actions:
        recorder.record_step(observation=observation, action={"name": action, "raw": action}, tool_result=observation)
    return recorder.finish(reward=reward, success=success, status="completed" if success else "failed")


def test_analyzer_extracts_only_failures() -> None:
    trajectories = [
        _trajectory("pass-1", 1.0, True, ["check", "submit"], "ok"),
        _trajectory("fail-1", 0.0, False, ["submit", "submit"], "Nothing happens"),
        _trajectory("fail-2", 0.0, False, ["submit"], "Nothing happens"),
    ]
    baseline = EvaluationResult(
        rewards={t.task_id: t.final_reward for t in trajectories},
        successes={t.task_id: t.success for t in trajectories},
    )
    packet = TraceFailureAnalyzer(benchmark="toy").analyze(trajectories, baseline=baseline, batch_id="b1")
    assert packet.benchmark == "toy"
    assert packet.batch_id == "b1"
    assert sorted(packet.failed_task_ids) == ["fail-1", "fail-2"]
    assert packet.baseline_pass == 1
    assert packet.batch_size == 3
    assert packet.cases[0].signals["repeated_action_values"] >= 1


def test_packet_prompt_is_deterministic_and_bounded() -> None:
    trajectories = [_trajectory("fail-1", 0.0, False, ["submit"], "Nothing happens")]
    packet = TraceFailureAnalyzer(benchmark="toy").analyze(trajectories)
    text1 = packet.to_prompt_text()
    text2 = packet.to_prompt_text()
    assert text1 == text2
    assert "Harness-R1 Direct Failure Trace Packet" in text1
    assert "fail-1" in text1
    assert packet.fingerprint() == TraceFailureAnalyzer(benchmark="toy").analyze(trajectories).fingerprint()


def test_packet_round_trip() -> None:
    trajectories = [_trajectory("fail-1", 0.0, False, ["submit"], "err")]
    packet = TraceFailureAnalyzer(benchmark="toy").analyze(trajectories)
    restored = FailurePacket.from_dict(packet.to_dict())
    assert restored.benchmark == packet.benchmark
    assert restored.failed_task_ids == packet.failed_task_ids
    assert restored.baseline_rewards == packet.baseline_rewards


def test_sanitize_redacts_ids_and_paths() -> None:
    text = "b012345678 /mnt/data/x /home/user/secret"
    cleaned = sanitize(text)
    assert "b012345678" not in cleaned
    assert "/mnt/data" not in cleaned
    assert "/home/user" not in cleaned


def test_compute_signals_counts_errors_and_noops() -> None:
    recorder = TrajectoryRecorder(Task(id="x", instruction="i"))
    recorder.record_step(observation="Nothing happens", action={"name": "go"}, tool_result="Nothing happens")
    recorder.record_step(observation="err", action={"name": "go"}, error="ValueError: bad")
    trajectory = recorder.finish(reward=0.0, success=False, status="failed")
    signals = compute_signals(trajectory)
    assert signals["no_op_observations"] == 1
    assert signals["errors"] == 1
    assert signals["tool_actions"] == 2
    assert signals["repeated_action_values"] == 1


def test_select_cases_strategies() -> None:
    from harnyx.optimization.failure_analysis import FailureCase

    made = [FailureCase(task_id=f"t{i}", reward=float(i) / 10) for i in range(5)]
    assert len(select_cases(made, 3, "lowest_reward")) == 3
    assert select_cases(made, 3, "first")[0].task_id == "t0"
    assert len(select_cases(made, 3, "round_robin")) == 3
