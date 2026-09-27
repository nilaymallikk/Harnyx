"""Model-provider abstraction for the harness engineer.

NOVA is provider-agnostic. The only requirement for a provider is the
:class:`~nova.llm.provider.LLMProvider` structural interface. OpenAI-compatible
endpoints (including OpenRouter, vLLM, SGLang, and local servers) use
:class:`~nova.llm.openai.OpenAICompatibleProvider`; deterministic tests use
:class:`~nova.llm.local.ScriptedProvider`.
"""

from __future__ import annotations

from nova.llm.local import CallableProvider, ScriptedProvider
from nova.llm.openai import OpenAICompatibleProvider, OpenRouterProvider
from nova.llm.provider import LLMProvider, LLMResponse, Message

__all__ = [
    "CallableProvider",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "OpenAICompatibleProvider",
    "OpenRouterProvider",
    "ScriptedProvider",
]
