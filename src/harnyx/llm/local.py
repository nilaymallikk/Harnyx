"""Deterministic providers for tests and offline demos."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from harnyx.errors import ProviderError
from harnyx.llm.provider import LLMResponse, coerce_messages


class ScriptedProvider:
    """Return a fixed, pre-queued list of responses in order.

    Used by the deterministic toy demo and the test suite. Raises
    :class:`ProviderError` when the script is exhausted so silent fallbacks
    cannot mask a bug.
    """

    name = "scripted"

    def __init__(self, responses: Sequence[str], *, model: str = "scripted") -> None:
        self._responses = list(responses)
        self._index = 0
        self.model = model
        self.calls: list[list[dict[str, str]]] = []

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(coerce_messages(messages))
        if self._index >= len(self._responses):
            raise ProviderError("scripted provider exhausted")
        text = self._responses[self._index]
        self._index += 1
        return LLMResponse(text=text, model=self.model, finish_reason="stop")

    @property
    def remaining(self) -> int:
        return len(self._responses) - self._index
