"""Harness engineering: patch representation, parsing, validation, generation."""

from __future__ import annotations

from nova.engineering.harness_engineer import (
    HarnessEngineer,
    LLMHarnessEngineer,
    ScriptedHarnessEngineer,
)
from nova.engineering.patch import (
    CodeHook,
    HarnessPatch,
    extract_json_object,
    extract_patch,
)
from nova.engineering.random_engineer import RandomHarnessEngineer
from nova.engineering.validation import PatchValidator, ValidationResult

__all__ = [
    "CodeHook",
    "HarnessEngineer",
    "HarnessPatch",
    "LLMHarnessEngineer",
    "PatchValidator",
    "RandomHarnessEngineer",
    "ScriptedHarnessEngineer",
    "ValidationResult",
    "extract_json_object",
    "extract_patch",
]
