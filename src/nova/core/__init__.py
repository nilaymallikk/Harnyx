"""Core data structures and protocols for NOVA.

This package holds the agent-agnostic primitives: tasks, trajectories, agents,
harnesses, and results. Nothing in ``nova.core`` depends on a specific model
provider, benchmark, or training framework.
"""

from __future__ import annotations

from nova.core.agent import Action, Agent, Environment, HarnessedAgent, Policy, StepResult
from nova.core.harness import (
    ACTION_EFFECT_KINDS,
    FEEDBACK_EFFECT_KINDS,
    HOOK_NAMES,
    BaseHarness,
    ExecutableHarness,
    Harness,
    HookContext,
    HookEffect,
    HookNames,
    HookSandbox,
)
from nova.core.result import AgentResult, EvaluationResult
from nova.core.task import Task
from nova.core.trajectory import Trajectory, TrajectoryRecorder, TrajectoryStep

__all__ = [
    "ACTION_EFFECT_KINDS",
    "Action",
    "Agent",
    "AgentResult",
    "BaseHarness",
    "Environment",
    "EvaluationResult",
    "ExecutableHarness",
    "FEEDBACK_EFFECT_KINDS",
    "HOOK_NAMES",
    "Harness",
    "HarnessedAgent",
    "HookContext",
    "HookEffect",
    "HookNames",
    "HookSandbox",
    "Policy",
    "StepResult",
    "Task",
    "Trajectory",
    "TrajectoryRecorder",
    "TrajectoryStep",
]
