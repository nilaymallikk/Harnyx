"""NOVA: learn to improve executable AI-agent harnesses from failure trajectories.

NOVA is an agent-agnostic, provider-agnostic Python implementation of the
Harness-R1 methodology. The public surface is intentionally small; see
``docs/architecture.md`` for the layer map and ``docs/reproduction.md`` for the
paper-to-implementation mapping.

Typical use::

    from nova import (
        ExecutableHarness, FailurePacket, HarnessOptimizer,
        LocalEvaluator, OutcomeReward, ScriptedHarnessEngineer,
    )
"""

from __future__ import annotations

from nova import errors
from nova.core.agent import Action, Agent, AgentResult, Environment, HarnessedAgent, Policy, StepResult
from nova.core.harness import (
    BaseHarness,
    ExecutableHarness,
    Harness,
    HookContext,
    HookEffect,
    HookNames,
)
from nova.core.result import EvaluationResult
from nova.core.task import Task
from nova.core.trajectory import Trajectory, TrajectoryRecorder, TrajectoryStep
from nova.engineering.harness_engineer import HarnessEngineer, LLMHarnessEngineer, ScriptedHarnessEngineer
from nova.engineering.patch import CodeHook, HarnessPatch
from nova.engineering.random_engineer import RandomHarnessEngineer
from nova.engineering.validation import PatchValidator, ValidationResult
from nova.evaluation.evaluator import Evaluator, LocalEvaluator
from nova.evaluation.harness_r1 import HarnessR1BenchmarkAdapter
from nova.optimization.failure_analysis import (
    FailureAnalyzer,
    FailureCase,
    FailurePacket,
    TraceFailureAnalyzer,
)
from nova.optimization.optimizer import HarnessOptimizer, OptimizationConfig, OptimizationResult
from nova.optimization.reward import OutcomeReward, RewardFunction, RewardResult
from nova.sandbox.limits import SandboxLimits
from nova.sandbox.runner import HookCompiler, LocalSandbox
from nova.versioning import HarnessVersion, HarnessVersionStore

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
