"""Resource limits applied to untrusted hook execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """Budgets and structural caps for a single hook source or invocation.

    The defaults reproduce the reference ``code_runner.py`` constants.
    """

    time_budget_s: float = 0.05
    line_budget: int = 2000
    max_source_chars: int = 8000
    max_ast_nodes: int = 1200
    max_str_chars: int = 700
    max_return_text_chars: int = 900
    max_helper_functions: int = 5
    # Process-level limits used by the subprocess sandbox.
    memory_mb: int | None = 256
    cpu_seconds: int | None = 2
    wall_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if self.time_budget_s <= 0:
            raise ValueError("time_budget_s must be positive")
        if self.line_budget <= 0:
            raise ValueError("line_budget must be positive")
        if self.wall_timeout_s <= 0:
            raise ValueError("wall_timeout_s must be positive")
