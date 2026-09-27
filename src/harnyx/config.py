"""Typed configuration for Harnyx runs.

Secrets are never stored in config documents. Provider API keys are read from
environment variables (``engineer.api_key_env`` names the variable). YAML
documents are supported when PyYAML is installed; JSON always works.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from harnyx.core.agent import DEFAULT_SYSTEM_PROMPT
from harnyx.core.types import to_jsonable
from harnyx.errors import ConfigError


@dataclass(slots=True)
class AgentConfig:
    """Target-agent settings."""

    name: str = "generic"
    max_steps: int = 20
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass(slots=True)
class EngineerConfig:
    """Harness-engineer provider settings."""

    provider: str = "openai_compatible"
    model: str = ""
    base_url: str = ""
    api_key_env: str = "HARNYX_ENGINEER_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 12288
    prefill_think: bool = True
    require_think: bool = True
    include_response_template: bool = False


@dataclass(slots=True)
class OptimizationSettings:
    """Optimization-loop settings (defaults follow the paper's reference)."""

    candidates: int = 8
    iterations: int = 1
    reward_metric: str = "delta_average_reward"
    valid_bonus: float = 0.0
    accept_threshold: float = 0.0
    reject_regressions: bool = True
    allow_regressions: bool = False
    max_traces: int = 20
    selection_strategy: str = "round_robin"
    reward_threshold: float = 1.0
    smoke_test: bool = False


@dataclass(slots=True)
class SandboxConfig:
    """Sandbox settings."""

    backend: str = "local"  # "local" (in-process, AST-hardened) or "subprocess"
    hook_time_budget_ms: float = 50.0
    line_budget: int = 2000
    wall_timeout_seconds: float = 5.0
    memory_mb: int | None = 256
    cpu_seconds: int | None = 2
    network: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"local", "subprocess"}:
            raise ConfigError(f"unsupported sandbox backend: {self.backend!r}")
        if self.network:
            # The AST policy already forbids imports, so hooks cannot open
            # sockets; Harnyx does not implement a network-enabled sandbox.
            raise ConfigError("network access is not supported for harness hooks; leave sandbox.network=false")


@dataclass(slots=True)
class EvaluationConfig:
    """Evaluation settings."""

    benchmark: str = "local"
    max_tasks: int | None = None
    trajectory_dir: str | None = None


@dataclass(slots=True)
class HarnyxConfig:
    """Top-level Harnyx configuration."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    engineer: EngineerConfig = field(default_factory=EngineerConfig)
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    run_dir: str = "runs"

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> HarnyxConfig:
        if not isinstance(raw, dict):
            raise ConfigError("configuration must be a mapping")
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ConfigError(f"unknown top-level configuration keys: {sorted(unknown)}")
        nested = {
            "agent": AgentConfig,
            "engineer": EngineerConfig,
            "optimization": OptimizationSettings,
            "sandbox": SandboxConfig,
            "evaluation": EvaluationConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key in nested:
                if not isinstance(value, dict):
                    raise ConfigError(f"configuration section {key!r} must be a mapping")
                allowed = {f.name for f in fields(nested[key])}
                extra = set(value) - allowed
                if extra:
                    raise ConfigError(f"unknown keys in {key!r}: {sorted(extra)}")
                kwargs[key] = nested[key](**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def resolve_api_key(self) -> str | None:
        """Return the engineer API key from the configured environment variable."""
        return os.environ.get(self.engineer.api_key_env)


def load_config(path: str | os.PathLike[str]) -> HarnyxConfig:
    """Load a Harnyx config from a JSON or YAML file."""
    target = Path(path)
    if not target.exists():
        raise ConfigError(f"configuration file not found: {target}")
    text = target.read_text(encoding="utf-8")
    if target.suffix.lower() in {".yaml", ".yml"}:
        raw = _load_yaml(text)
    else:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON configuration: {exc}") from exc
    return HarnyxConfig.from_dict(raw)


def save_config(config: HarnyxConfig, path: str | os.PathLike[str]) -> Path:
    """Write a config as deterministic JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _load_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ConfigError("PyYAML is required for YAML configs; install harnyx[yaml] or use JSON") from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError("YAML configuration must be a mapping")
    return data
