"""Typed error hierarchy for NOVA.

Every public failure mode raises a subclass of :class:`NovaError` so callers can
distinguish library errors from ordinary bugs without string matching.
"""

from __future__ import annotations


class NovaError(Exception):
    """Base class for all NOVA errors."""


class ConfigError(NovaError):
    """Raised when a configuration document is missing or invalid."""


class ProviderError(NovaError):
    """Raised when a model provider cannot return a usable response."""


# --------------------------------------------------------------------------- #
# Patch handling
# --------------------------------------------------------------------------- #
class PatchError(NovaError):
    """Base class for patch parsing and validation failures."""


class PatchParseError(PatchError):
    """Raised when model output cannot be parsed into a patch object."""


class PatchValidationError(PatchError):
    """Raised when a parsed patch violates the harness patch contract."""


class PatchCompileError(PatchValidationError):
    """Raised when a hook body fails AST policy or compilation."""


# --------------------------------------------------------------------------- #
# Sandbox
# --------------------------------------------------------------------------- #
class SandboxError(NovaError):
    """Base class for sandbox failures."""


class SandboxTimeout(SandboxError):
    """Raised when untrusted code exceeds its execution budget."""


class SandboxViolation(SandboxError):
    """Raised when untrusted code violates an isolation policy."""


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
class TrajectoryError(NovaError):
    """Raised when a trajectory cannot be recorded or serialized."""


class EvaluationError(NovaError):
    """Raised when an evaluation cannot produce a comparable result."""


class EngineerError(NovaError):
    """Raised when a harness engineer cannot produce a candidate patch."""
