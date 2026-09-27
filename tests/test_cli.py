from __future__ import annotations

import json

import pytest

from harnyx.cli.main import main
from harnyx.core.task import Task
from harnyx.core.trajectory import TrajectoryRecorder
from harnyx.core.types import write_jsonl


def _trajectory_rows(path) -> None:
    recorder = TrajectoryRecorder(Task(id="f1", instruction="do it"))
    recorder.record_step(observation="Nothing happens", action={"name": "go"}, tool_result="Nothing happens")
    trajectory = recorder.finish(reward=0.0, success=False, status="failed")
    write_jsonl(path, [trajectory.to_dict()])


def test_cli_run(tmp_path, capsys) -> None:
    assert main(["run", "--run-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Harnyx toy demo" in out
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


def test_cli_optimize_respects_config(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "optimization": {"candidates": 2, "iterations": 2},
                "run_dir": str(tmp_path / "runs"),
                "sandbox": {"backend": "local"},
            }
        ),
        encoding="utf-8",
    )
    # The plugin supplies its own engineer, so no network is needed.
    assert main(["optimize", "--config", str(config), "--plugin", "examples.plugins.verify_agent:build"]) == 0
    run_dirs = list((tmp_path / "runs").glob("*"))
    assert run_dirs
    report = json.loads((run_dirs[0] / "report.json").read_text())
    assert report["config"]["candidates"] == 2
    assert len(report["iterations"]) == 2


def test_cli_evaluate_respects_config(tmp_path, capsys) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"evaluation": {"benchmark": "verify"}}), encoding="utf-8")
    assert main(["evaluate", "--config", str(config), "--plugin", "examples.plugins.verify_agent:build"]) == 0
    assert "benchmark=verify" in capsys.readouterr().out
