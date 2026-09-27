"""Model-provider abstraction for the harness engineer.

Harnyx is provider-agnostic. The only requirement for a provider is the
:class:`~harnyx.llm.provider.LLMProvider` structural interface. OpenAI-compatible
endpoints (including OpenRouter, vLLM, SGLang, and local servers) use
:class:`~harnyx.llm.openai.OpenAICompatibleProvider`; deterministic tests use
:class:`~harnyx.llm.local.ScriptedProvider`.
"""

from __future__ import annotations

from harnyx.llm.local import CallableProvider, ScriptedProvider
from harnyx.llm.openai import OpenAICompatibleProvider, OpenRouterProvider
from harnyx.llm.provider import LLMProvider, LLMResponse, Message

__all__ = [
    "CallableProvider",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "OpenAICompatibleProvider",
    "OpenRouterProvider",
    "ScriptedProvider",
]
