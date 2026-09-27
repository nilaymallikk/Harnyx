from __future__ import annotations

import json

from harnyx.core.task import Task
from harnyx.core.trajectory import Trajectory, TrajectoryRecorder


def test_recorder_builds_serializable_trajectory(tmp_path) -> None:
    task = Task(id="t1", instruction="do the thing")
    jsonl = tmp_path / "trajectories.jsonl"
    recorder = TrajectoryRecorder(task, benchmark="toy", agent_name="agent", jsonl_path=jsonl)
    recorder.record_step(observation="start", action={"name": "look"})
    recorder.record_step(
        observation="done",
        action={"name": "act"},
        tool_result="ok",
        harness_effects=[{"kind": "inject_hint", "message": "careful"}],
    )
    trajectory = recorder.finish(reward=1.0, success=True, status="completed")

    assert trajectory.num_steps == 2
    assert trajectory.actions() == [{"name": "look"}, {"name": "act"}]
    assert trajectory.observations() == ["start", "done"]
    assert len(trajectory.harness_effects()) == 1

    payload = trajectory.to_dict()
    assert payload["task_id"] == "t1"
    assert payload["num_steps"] == 2
    assert payload["success"] is True
    json.dumps(payload)  # must be JSON-serializable

    lines = [json.loads(line) for line in jsonl.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["task_id"] == "t1"


def test_trajectory_round_trip() -> None:
    task = Task(id="t2", instruction="x")
    recorder = TrajectoryRecorder(task)
    recorder.record_step(observation="obs", action={"name": "a", "arguments": {"value": "a"}}, tool_result="res")
    original = recorder.finish(reward=0.5, success=False, status="failed", error="boom")
    restored = Trajectory.from_dict(original.to_dict())
    assert restored.task_id == original.task_id
    assert restored.final_reward == original.final_reward
    assert restored.status == original.status
    assert restored.error == original.error
    assert len(restored.steps) == 1
    assert restored.steps[0].tool_result == "res"


def test_trajectory_step_round_trip() -> None:
    from harnyx.core.trajectory import TrajectoryStep

    step = TrajectoryStep(index=3, observation="o", action={"name": "n"}, error="e", state={"k": 1})
    restored = TrajectoryStep.from_dict(step.to_dict())
    assert restored.index == 3
    assert restored.action == {"name": "n"}
    assert restored.state == {"k": 1}
    assert restored.error == "e"
