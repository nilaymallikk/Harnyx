from __future__ import annotations

import json

import pytest

from nova.cli.main import main
from nova.core.task import Task
from nova.core.trajectory import TrajectoryRecorder
from nova.core.types import write_jsonl


def _trajectory_rows(path) -> None:
    recorder = TrajectoryRecorder(Task(id="f1", instruction="do it"))
    recorder.record_step(observation="Nothing happens", action={"name": "go"}, tool_result="Nothing happens")
    trajectory = recorder.finish(reward=0.0, success=False, status="failed")
    write_jsonl(path, [trajectory.to_dict()])


def test_cli_run(tmp_path, capsys) -> None:
    assert main(["run", "--run-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "NOVA toy demo" in out
    assert "engineer reward" in out


def test_cli_evaluate_prints_metrics(capsys) -> None:
    assert main(["evaluate"]) == 0
    out = capsys.readouterr().out
    assert "mean_reward" in out
    assert "success_rate" in out


def test_cli_generate_failures(tmp_path) -> None:
    traj = tmp_path / "trajs.jsonl"
    out = tmp_path / "packet.json"
    _trajectory_rows(traj)
    assert main(["generate-failures", "--trajectories", str(traj), "--benchmark", "toy", "--output", str(out)]) == 0
    packet = json.loads(out.read_text())
    assert packet["benchmark"] == "toy"
    assert len(packet["cases"]) == 1


def test_cli_inspect_trajectory(tmp_path, capsys) -> None:
    traj = tmp_path / "trajs.jsonl"
    _trajectory_rows(traj)
    assert main(["inspect-trajectory", str(traj)]) == 0
    out = capsys.readouterr().out
    assert "task=f1" in out


def test_cli_no_command_returns_usage() -> None:
    assert main([]) == 2


def test_cli_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_optimize_toy(tmp_path) -> None:
    assert main(["optimize", "--run-dir", str(tmp_path), "--candidates", "2"]) == 0
