"""Harness engineering: patch representation, parsing, validation, generation."""

from __future__ import annotations

from harnyx.engineering.harness_engineer import (
    HarnessEngineer,
    LLMHarnessEngineer,
    ScriptedHarnessEngineer,
)
from harnyx.engineering.patch import (
    CodeHook,
    HarnessPatch,
    extract_json_object,
    extract_patch,
)
from harnyx.engineering.random_engineer import RandomHarnessEngineer
from harnyx.engineering.validation import PatchValidator, ValidationResult

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
