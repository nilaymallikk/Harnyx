"""Static and sandboxed validation of candidate harness patches.

A patch must pass three gates before it may be installed:

1. **Schema** — single JSON object, known benchmark, non-empty ``actions``, all
   actions are ``add_code_hook``, each lifecycle hook used at most once.
2. **AST policy** — every hook body passes :mod:`harnyx.sandbox.policy` (no
   imports, I/O, dunder access, dynamic execution, etc.).
3. **Sandbox smoke test** — optional, executes each hook once in an isolated
   process to catch runtime failures before rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harnyx.core.harness import HookNames
from harnyx.engineering.patch import HarnessPatch, extract_patch
from harnyx.errors import PatchError, PatchParseError, PatchValidationError
from harnyx.sandbox.runner import HookCompiler

MAX_HOOKS = len(HookNames)


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating a candidate patch."""

    ok: bool
    patch: HarnessPatch | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> HarnessPatch:
        if not self.ok or self.patch is None:
            raise PatchValidationError("; ".join(self.errors) or "patch is invalid")
        return self.patch


def _default_smoke_hook(hook_name: str) -> str:
    # on_before_action/on_post_step effects reference an action string; an empty
    # context still exercises the code path without touching the environment.
    return hook_name


class PatchValidator:
    """Validate candidate patches against the harness patch contract."""

    def __init__(
        self,
        *,
        compiler: HookCompiler | None = None,
        require_unique_hooks: bool = True,
        max_hooks: int = MAX_HOOKS,
        smoke_sandbox: Any | None = None,
    ) -> None:
        self.compiler = compiler or HookCompiler()
        self.require_unique_hooks = require_unique_hooks
        self.max_hooks = max_hooks
        self.smoke_sandbox = smoke_sandbox

    def validate(
        self,
        candidate: HarnessPatch | str | dict[str, Any],
        *,
        benchmark: str | None = None,
        require_think: bool = False,
        prefill_think: bool = False,
    ) -> ValidationResult:
        """Validate ``candidate`` and return a structured result (never raises)."""
        errors: list[str] = []
        warnings: list[str] = []
        try:
            patch = self._coerce(candidate, benchmark=benchmark, require_think=require_think, prefill_think=prefill_think)
        except (PatchError, PatchParseError, PatchValidationError) as exc:
            return ValidationResult(ok=False, errors=[str(exc)])

        if not patch.benchmark:
            errors.append("patch.benchmark is empty")
        hooks = patch.hooks
        if not hooks:
            errors.append("patch has no actions")
        if len(hooks) > self.max_hooks:
            errors.append(f"patch has {len(hooks)} hooks; at most {self.max_hooks} allowed")
        names = [hook.hook for hook in hooks]
        if self.require_unique_hooks and len(set(names)) != len(names):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            errors.append(f"each lifecycle hook may be used at most once; repeated: {duplicates}")

        for index, hook in enumerate(hooks):
            if hook.hook not in HookNames:
                errors.append(f"action {index}: unsupported hook {hook.hook!r}")
                continue
            try:
                self.compiler.compile(hook.code, benchmark=patch.benchmark)
            except PatchValidationError as exc:
                errors.append(f"action {index} ({hook.hook}): {exc}")
            except Exception as exc:  # defensive: compilation must never leak
                errors.append(f"action {index} ({hook.hook}): {type(exc).__name__}: {exc}")

        if self.smoke_sandbox is not None and not errors:
            errors.extend(self._smoke(patch))

        ok = not errors
        return ValidationResult(ok=ok, patch=patch if ok else patch, errors=errors, warnings=warnings)

    def _smoke(self, patch: HarnessPatch) -> list[str]:
        failures: list[str] = []
        for hook in patch.hooks:
            try:
                self.smoke_sandbox.smoke_test(hook.code, benchmark=patch.benchmark, hook_name=hook.hook)
            except Exception as exc:
                failures.append(f"smoke test failed for {hook.hook}: {type(exc).__name__}: {exc}")
        return failures

    @staticmethod
    def _coerce(
        candidate: HarnessPatch | str | dict[str, Any],
        *,
        benchmark: str | None,
        require_think: bool,
        prefill_think: bool,
    ) -> HarnessPatch:
        if isinstance(candidate, HarnessPatch):
            return candidate
        if isinstance(candidate, str):
            return extract_patch(
                candidate, benchmark=benchmark, require_think=require_think, prefill_think=prefill_think
            )
        if isinstance(candidate, dict):
            return HarnessPatch.from_dict(candidate, bench=benchmark)
        raise PatchValidationError(f"unsupported candidate type: {type(candidate).__name__}")
