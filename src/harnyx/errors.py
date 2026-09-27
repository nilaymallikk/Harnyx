"""Typed error hierarchy for Harnyx.

Every public failure mode raises a subclass of :class:`HarnyxError` so callers can
distinguish library errors from ordinary bugs without string matching.
"""

from __future__ import annotations


class HarnyxError(Exception):
    """Base class for all Harnyx errors."""


class ConfigError(HarnyxError):
    """Raised when a configuration document is missing or invalid."""


class ProviderError(HarnyxError):
    """Raised when a model provider cannot return a usable response."""


# --------------------------------------------------------------------------- #
# Patch handling
# --------------------------------------------------------------------------- #
class PatchError(HarnyxError):
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
class SandboxError(HarnyxError):
    """Base class for sandbox failures."""


class SandboxTimeout(SandboxError):
    """Raised when untrusted code exceeds its execution budget."""


class SandboxViolation(SandboxError):
    """Raised when untrusted code violates an isolation policy."""


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
class TrajectoryError(HarnyxError):
    """Raised when a trajectory cannot be recorded or serialized."""


class EvaluationError(HarnyxError):
    """Raised when an evaluation cannot produce a comparable result."""


class EngineerError(HarnyxError):
    """Raised when a harness engineer cannot produce a candidate patch."""
