"""Harnyx: learn to improve executable AI-agent harnesses from failure trajectories.

Harnyx is an agent-agnostic, provider-agnostic Python implementation of the
Harness-R1 methodology. The public surface is intentionally small; see
``docs/architecture.md`` for the layer map and ``docs/reproduction.md`` for the
paper-to-implementation mapping.

Typical use::

    from harnyx import (
        ExecutableHarness, FailurePacket, HarnessOptimizer,
        LocalEvaluator, OutcomeReward, ScriptedHarnessEngineer,
    )
"""

from __future__ import annotations

from harnyx import errors
from harnyx.core.agent import Action, Agent, AgentResult, Environment, HarnessedAgent, Policy, StepResult
from harnyx.core.harness import (
    BaseHarness,
    ExecutableHarness,
    Harness,
    HookContext,
    HookEffect,
    HookNames,
)
from harnyx.core.result import EvaluationResult
from harnyx.core.task import Task
from harnyx.core.trajectory import Trajectory, TrajectoryRecorder, TrajectoryStep
from harnyx.engineering.harness_engineer import HarnessEngineer, LLMHarnessEngineer, ScriptedHarnessEngineer
from harnyx.engineering.patch import CodeHook, HarnessPatch
from harnyx.engineering.random_engineer import RandomHarnessEngineer
from harnyx.engineering.validation import PatchValidator, ValidationResult
from harnyx.evaluation.evaluator import Evaluator, LocalEvaluator
from harnyx.evaluation.harness_r1 import HarnessR1BenchmarkAdapter
from harnyx.optimization.failure_analysis import (
    FailureAnalyzer,
    FailureCase,
    FailurePacket,
    TraceFailureAnalyzer,
)
from harnyx.optimization.optimizer import HarnessOptimizer, OptimizationConfig, OptimizationResult
from harnyx.optimization.reward import OutcomeReward, RewardFunction, RewardResult
from harnyx.sandbox.limits import SandboxLimits
from harnyx.sandbox.runner import HookCompiler, LocalSandbox
from harnyx.versioning import HarnessVersion, HarnessVersionStore

__version__ = "0.1.0"

__all__ = [
    "Action",
    "Agent",
    "AgentResult",
    "BaseHarness",
    "CodeHook",
    "Environment",
    "EvaluationResult",
    "Evaluator",
    "ExecutableHarness",
    "FailureAnalyzer",
    "FailureCase",
    "FailurePacket",
    "Harness",
    "HarnessEngineer",
    "HarnessOptimizer",
    "HarnessPatch",
    "HarnessR1BenchmarkAdapter",
    "HarnessVersion",
    "HarnessVersionStore",
    "HarnessedAgent",
    "HookCompiler",
    "HookContext",
    "HookEffect",
    "HookNames",
    "LLMHarnessEngineer",
    "LocalEvaluator",
    "LocalSandbox",
    "OptimizationConfig",
    "OptimizationResult",
    "OutcomeReward",
    "PatchValidator",
    "Policy",
    "RandomHarnessEngineer",
    "RewardFunction",
    "RewardResult",
    "SandboxLimits",
    "ScriptedHarnessEngineer",
    "StepResult",
    "Task",
    "TraceFailureAnalyzer",
    "Trajectory",
    "TrajectoryRecorder",
    "TrajectoryStep",
    "ValidationResult",
    "errors",
    "__version__",
]
