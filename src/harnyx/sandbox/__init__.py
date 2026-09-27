"""Executable-patch sandboxing.

Untrusted harness hooks are validated as a strict Python subset (no imports,
I/O, dunder access, dynamic evaluation, loops-with-exit, etc.), compiled with a
restricted builtin set, and executed under time/line budgets. A subprocess
sandbox provides process-level isolation for pre-flight validation.
"""

from __future__ import annotations

from harnyx.sandbox.isolation import SubprocessSandbox
from harnyx.sandbox.limits import SandboxLimits
from harnyx.sandbox.policy import HookPolicy, validate_hook_source
from harnyx.sandbox.runner import CompiledHook, HookCompiler, LocalSandbox

__all__ = [
    "CompiledHook",
    "HookCompiler",
    "HookPolicy",
    "LocalSandbox",
    "SandboxLimits",
    "SubprocessSandbox",
    "validate_hook_source",
]
