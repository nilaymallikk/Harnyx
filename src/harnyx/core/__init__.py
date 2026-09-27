"""Core data structures and protocols for Harnyx.

This package holds the agent-agnostic primitives: tasks, trajectories, agents,
harnesses, and results. Nothing in ``harnyx.core`` depends on a specific model
provider, benchmark, or training framework.
"""

from __future__ import annotations

from harnyx.core.agent import Action, Agent, Environment, HarnessedAgent, Policy, StepResult
from harnyx.core.harness import (
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
from harnyx.core.result import AgentResult, EvaluationResult
from harnyx.core.task import Task
from harnyx.core.trajectory import Trajectory, TrajectoryRecorder, TrajectoryStep

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
