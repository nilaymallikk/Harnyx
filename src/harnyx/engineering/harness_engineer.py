"""Harness engineers: components that diagnose failures and propose patches.

A harness engineer is *not* a task solver. It reads a failure packet and emits an
executable harness overlay. Two implementations ship:

* :class:`LLMHarnessEngineer` — provider-agnostic, samples candidates from a
  model using the released ``prefill_think_patch`` protocol.
* :class:`ScriptedHarnessEngineer` — deterministic, for the toy demo and tests.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from harnyx.engineering.patch import HarnessPatch, extract_patch
from harnyx.engineering.prompt import build_engineer_messages
from harnyx.errors import EngineerError, PatchError
from harnyx.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


@runtime_checkable
class HarnessEngineer(Protocol):
    """Structural interface for a harness engineer."""

    name: str

    def generate_patch(self, packet: Any) -> HarnessPatch:
        """Return one candidate patch for ``packet``."""

    def generate_candidates(self, packet: Any, n: int) -> list[HarnessPatch]:
        """Return up to ``n`` candidate patches for ``packet``."""


class BaseHarnessEngineer:
    """Shared candidate-generation helper."""

    name = "base-engineer"

    def generate_patch(self, packet: Any) -> HarnessPatch:  # pragma: no cover - interface
        raise NotImplementedError

    def generate_candidates(self, packet: Any, n: int) -> list[HarnessPatch]:
        if n < 1:
            raise ValueError("n must be >= 1")
        candidates: list[HarnessPatch] = []
        errors: list[str] = []
        for _ in range(n):
            try:
                candidates.append(self.generate_patch(packet))
            except (EngineerError, PatchError) as exc:
                errors.append(str(exc))
        if not candidates:
            raise EngineerError(f"engineer produced no parseable candidate ({'; '.join(errors) or 'unknown'})")
        return candidates


class LLMHarnessEngineer(BaseHarnessEngineer):
    """A model-driven harness engineer."""

    name = "llm-engineer"

    def __init__(
        self,
        provider: LLMProvider,
        *,
        benchmark: str = "",
        temperature: float = 0.7,
        max_tokens: int = 12288,
        prefill_think: bool = True,
        require_think: bool = True,
        include_response_template: bool = False,
        provider_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.benchmark = benchmark
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prefill_think = prefill_think
        self.require_think = require_think
        self.include_response_template = include_response_template
        self.provider_kwargs = dict(provider_kwargs or {})

    def build_messages(self, packet: Any) -> list[dict[str, str]]:
        return build_engineer_messages(
            packet,
            benchmark=self.benchmark or None,
            include_response_template=self.include_response_template,
        )

    def generate_patch(self, packet: Any, *, temperature: float | None = None) -> HarnessPatch:
        bench = self.benchmark or getattr(packet, "benchmark", "") or ""
        messages = self.build_messages(packet)
        response = self.provider.complete(
            messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            **self.provider_kwargs,
        )
        try:
            return extract_patch(
                response.text,
                benchmark=bench or None,
                require_think=self.require_think,
                prefill_think=self.prefill_think,
                source=getattr(self.provider, "name", "engineer"),
            )
        except PatchError as exc:
            raise EngineerError(f"engineer produced an unparseable patch: {exc}") from exc

    def generate_candidates(self, packet: Any, n: int) -> list[HarnessPatch]:
        if n < 1:
            raise ValueError("n must be >= 1")
        candidates: list[HarnessPatch] = []
        errors: list[str] = []
        for _ in range(n):
            try:
                candidates.append(self.generate_patch(packet))
            except (EngineerError, PatchError) as exc:
                errors.append(str(exc))
                logger.warning("candidate generation failed: %s", exc)
        if not candidates:
            raise EngineerError(f"engineer produced no parseable candidate ({'; '.join(errors) or 'unknown'})")
        return candidates


class ScriptedHarnessEngineer(BaseHarnessEngineer):
    """Return deterministic patches, cycling through a fixed list.

    This makes the end-to-end demo and CI test fully reproducible without any
    external model, while exercising the real parser, validator, sandbox, and
    reward path.
    """

    name = "scripted-engineer"

    def __init__(self, patches: Sequence[HarnessPatch], *, loop: bool = True) -> None:
        if not patches:
            raise ValueError("ScriptedHarnessEngineer requires at least one patch")
        self._patches = list(patches)
        self._loop = loop
        self._index = 0

    def generate_patch(self, packet: Any) -> HarnessPatch:
        if self._index >= len(self._patches):
            if not self._loop:
                raise EngineerError("scripted engineer exhausted")
            self._index = 0
        patch = self._patches[self._index]
        self._index += 1
        return patch

    def generate_candidates(self, packet: Any, n: int) -> list[HarnessPatch]:
        return [self.generate_patch(packet) for _ in range(max(0, n))]
