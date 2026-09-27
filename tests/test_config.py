from __future__ import annotations

import pytest

from nova.config import NovaConfig, load_config, save_config
from nova.errors import ConfigError


def test_config_defaults_and_round_trip() -> None:
    config = NovaConfig()
    assert config.optimization.candidates == 8
    assert config.engineer.provider == "openai_compatible"
    restored = NovaConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ConfigError):
        NovaConfig.from_dict({"unknown": 1})
    with pytest.raises(ConfigError):
        NovaConfig.from_dict({"optimization": {"nope": 1}})


def test_sandbox_backend_and_network_validation() -> None:
    with pytest.raises(ConfigError):
        NovaConfig.from_dict({"sandbox": {"backend": "docker"}})
    with pytest.raises(ConfigError):
        NovaConfig.from_dict({"sandbox": {"network": True}})


def test_load_and_save_json(tmp_path) -> None:
    config = NovaConfig.from_dict({"optimization": {"candidates": 4}, "run_dir": "myruns"})
    path = tmp_path / "config.json"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.optimization.candidates == 4
    assert loaded.run_dir == "myruns"


def test_load_yaml(tmp_path) -> None:
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "config.yaml"
    path.write_text(
        "optimization:\n  candidates: 3\nengineer:\n  model: test-model\n",
        encoding="utf-8",
    )
    loaded = load_config(path)
    assert loaded.optimization.candidates == 3
    assert loaded.engineer.model == "test-model"
    assert yaml is not None


def test_load_missing_file() -> None:
    with pytest.raises(ConfigError):
        load_config("/nonexistent/nova.json")


def test_api_key_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("NOVA_ENGINEER_API_KEY", "secret-value")
    config = NovaConfig()
    assert config.resolve_api_key() == "secret-value"
