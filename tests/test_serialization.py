from __future__ import annotations

import json

from harnyx.core.result import EvaluationResult
from harnyx.core.task import Task
from harnyx.core.trajectory import TrajectoryRecorder
from harnyx.core.types import (
    append_jsonl,
    canonical_json,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from harnyx.engineering.patch import CodeHook, HarnessPatch
from harnyx.optimization.failure_analysis import FailurePacket, TraceFailureAnalyzer


def test_canonical_json_is_deterministic() -> None:
    a = {"b": 1, "a": [3, 2, {"d": 4, "c": 5}]}
    b = {"a": [3, 2, {"c": 5, "d": 4}], "b": 1}
    assert canonical_json(a) == canonical_json(b)
    assert canonical_json({"x": [1, 2]}) == '{"x":[1,2]}'


def test_write_and_read_json(tmp_path) -> None:
    path = tmp_path / "nested" / "doc.json"
    write_json(path, {"z": 1, "a": 2})
    assert read_json(path) == {"a": 2, "z": 1}
    # Writes are atomic and deterministic.
    assert path.read_text().index('"a"') < path.read_text().index('"z"')


def test_jsonl_round_trip(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"i": 2}, {"i": 1}])
    append_jsonl(path, {"i": 3})
    rows = read_jsonl(path)
    assert rows == [{"i": 2}, {"i": 1}, {"i": 3}]
    assert read_jsonl(tmp_path / "missing.jsonl") == []


def test_evaluation_result_serializes_trajectories() -> None:
    recorder = TrajectoryRecorder(Task(id="t", instruction="i"))
    recorder.record_step(observation="o", action={"name": "a"})
    trajectory = recorder.finish(reward=1.0, success=True, status="completed")
    result = EvaluationResult(
        rewards={"t": 1.0},
        successes={"t": True},
        trajectories={"t": trajectory},
    )
    payload = result.to_dict()
    assert payload["mean_reward"] == 1.0
    assert payload["trajectories"]["t"]["task_id"] == "t"
    json.dumps(payload)


def test_patch_dict_is_json_round_trippable() -> None:
    patch = HarnessPatch(
        benchmark="toy",
        description="d",
        hooks=(CodeHook(hook="on_init", code="def hook(ctx, nb):\n    return None\n"),),
    )
    restored = HarnessPatch.from_dict(json.loads(patch.to_json()))
    assert restored.to_dict() == patch.to_dict()


def test_failure_packet_serialization_is_stable() -> None:
    recorder = TrajectoryRecorder(Task(id="f", instruction="i"))
    recorder.record_step(observation="o", action={"name": "a"}, tool_result="nothing happens")
    trajectory = recorder.finish(reward=0.0, success=False, status="failed")
    packet = TraceFailureAnalyzer(benchmark="toy").analyze([trajectory])
    restored = FailurePacket.from_dict(json.loads(canonical_json(packet.to_dict())))
    assert restored.to_dict() == packet.to_dict()
