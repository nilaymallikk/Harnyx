"""Outcome-grounded harness optimization.

This package implements the Harness-R1 optimization loop minus the neural
training: failure extraction, candidate patch generation, sandboxed evaluation,
same-batch outcome reward, regression-protected selection, and versioning.
"""

from __future__ import annotations

from nova.optimization.failure_analysis import (
    FailureAnalyzer,
    FailureCase,
    FailurePacket,
    TraceFailureAnalyzer,
)
from nova.optimization.optimizer import (
    HarnessOptimizer,
    IterationReport,
    OptimizationConfig,
    OptimizationResult,
)
from nova.optimization.patch_generation import CandidatePatch, PatchGenerator
from nova.optimization.reward import OutcomeReward, RewardFunction, RewardResult
from nova.optimization.selection import CandidateSelector, SelectionResult

__all__ = [
    "CandidatePatch",
    "CandidateSelector",
    "FailureAnalyzer",
    "FailureCase",
    "FailurePacket",
    "HarnessOptimizer",
    "IterationReport",
    "OptimizationConfig",
    "OptimizationResult",
    "OutcomeReward",
    "PatchGenerator",
    "RewardFunction",
    "RewardResult",
    "SelectionResult",
    "TraceFailureAnalyzer",
]
