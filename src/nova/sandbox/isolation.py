"""Process-isolated hook execution for untrusted patches.

The reference implementation executes AST-validated hooks in-process (see
:mod:`nova.sandbox.runner`). NOVA additionally offers a subprocess sandbox that
runs a hook in a fresh interpreter with:

* a sanitized environment (no API keys, no credentials),
* ``RLIMIT_AS``/``RLIMIT_CPU`` resource caps where the platform supports them,
* a wall-clock timeout,
* the same AST policy and restricted builtins.

Use this sandbox as a pre-flight gate for engineer-generated code, or as the
execution backend when process isolation is preferred over speed.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from nova.core.harness import HookEffect
from nova.errors import SandboxError, SandboxTimeout
from nova.sandbox.limits import SandboxLimits
from nova.sandbox.runner import CompiledHook, HookCompiler, LocalSandbox


def _nova_src_root() -> str:
    import nova

    return str(Path(nova.__file__).resolve().parents[1])


class SubprocessSandbox:
    """Run a single hook invocation in an isolated child interpreter."""

    def __init__(self, limits: SandboxLimits | None = None, *, python: str | None = None) -> None:
        self.limits = limits or SandboxLimits()
        self.python = python or sys.executable
        self.compiler = HookCompiler()
        self._local = LocalSandbox(self.limits)

    def compile(self, source: str, *, benchmark: str | None = None) -> CompiledHook:
        """Validate the source statically; execution happens in the child."""
        return self.compiler.compile(source, benchmark=benchmark)

    def run(
        self,
        compiled: CompiledHook,
        ctx: Mapping[str, Any],
        nb: MutableMapping[str, Any],
        *,
        hook_name: str,
    ) -> HookEffect | None:
        payload = {
            "source": compiled.source,
            "benchmark": compiled.benchmark,
            "hook_name": hook_name,
            "ctx": _jsonable(ctx),
            "nb": _jsonable(nb),
            "limits": dataclasses.asdict(self.limits),
        }
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": _nova_src_root(),
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "LANG": "C.UTF-8",
        }
        cmd = [self.python, "-c", "from nova.sandbox.isolation import _child_main; _child_main()"]
        try:
            proc = subprocess.run(
                cmd,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.limits.wall_timeout_s,
                env=env,
                cwd=str(Path(_nova_src_root()).parent),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxTimeout(f"hook exceeded {self.limits.wall_timeout_s}s wall timeout") from exc
        if proc.returncode != 0:
            raise SandboxError(f"hook subprocess failed (rc={proc.returncode}): {proc.stderr.strip()[:400]}")
        try:
            out = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise SandboxError("hook subprocess produced no parseable result") from exc

        # Reflect notebook mutations back into the caller's mapping.
        returned_nb = out.get("nb")
        if isinstance(returned_nb, dict):
            nb.clear()
            nb.update(returned_nb)

        effect = out.get("effect")
        if not isinstance(effect, dict):
            return None
        return HookEffect(
            kind=str(effect.get("kind", "")),
            message=str(effect.get("message", "")),
            action=str(effect.get("action", "")),
            skills=tuple(effect.get("skills", []) or []),
            tool_hint=str(effect.get("tool_hint", "")),
        )

    def smoke_test(self, source: str, *, benchmark: str | None = None, hook_name: str = "on_init") -> None:
        """Execute ``source`` once in isolation to catch runtime failures early.

        Raises:
            SandboxError/SandboxTimeout: if the hook cannot run safely.
        """
        compiled = self.compile(source, benchmark=benchmark)
        self.run(compiled, {}, {}, hook_name=hook_name)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _apply_resource_limits(limits: SandboxLimits) -> None:
    try:
        import resource

        if limits.memory_mb:
            soft = limits.memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
        if limits.cpu_seconds:
            resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
    except (ImportError, ValueError, OSError):
        # Resource limits are best-effort; the wall timeout still applies.
        return


def _child_main() -> None:  # pragma: no cover - exercised via subprocess
    """Child-process entry point. Reads a JSON payload on stdin, writes JSON out."""
    payload = json.loads(sys.stdin.read())
    limits = SandboxLimits(**payload["limits"])
    _apply_resource_limits(limits)
    sandbox = LocalSandbox(limits)
    compiled = sandbox.compile(payload["source"], benchmark=payload.get("benchmark"))
    nb = payload.get("nb") or {}
    effect = sandbox.run(compiled, payload.get("ctx") or {}, nb, hook_name=payload["hook_name"])
    json.dump({"effect": effect.to_dict() if effect is not None else None, "nb": nb}, sys.stdout)
