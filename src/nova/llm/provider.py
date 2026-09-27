"""Provider protocol and shared response types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from nova.core.types import to_jsonable


@dataclass(frozen=True, slots=True)
class Message:
    """A chat message."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class LLMResponse:
    """A single completion."""

    text: str
    model: str = ""
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@runtime_checkable
class LLMProvider(Protocol):
    """Structural interface every model provider must satisfy."""

    name: str

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Return one completion for ``messages``."""


def coerce_messages(messages: Sequence[Mapping[str, Any] | Message]) -> list[dict[str, str]]:
    """Normalize a message sequence into OpenAI-style dicts."""
    out: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, Message):
            out.append(message.to_dict())
        elif isinstance(message, Mapping):
            out.append({"role": str(message.get("role", "user")), "content": str(message.get("content", ""))})
        else:
            raise TypeError(f"unsupported message type: {type(message).__name__}")
    return out
