"""OpenAI-compatible chat-completions provider (stdlib only).

Any server exposing ``POST {base_url}/chat/completions`` works: OpenAI,
OpenRouter, vLLM, SGLang, llama.cpp, Ollama's OpenAI shim, and the Harness-R1
reference checkpoints served behind an OpenAI-compatible endpoint.

API keys are read from the constructor or an environment variable and are never
hard-coded or logged.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from harnyx.errors import ProviderError
from harnyx.llm.provider import LLMResponse, coerce_messages

DEFAULT_ENV_KEY = "HARNYX_ENGINEER_API_KEY"


class OpenAICompatibleProvider:
    """A minimal, dependency-free OpenAI-compatible chat client."""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        env_key: str = DEFAULT_ENV_KEY,
        timeout: float = 180.0,
        default_temperature: float = 0.0,
        default_max_tokens: int | None = None,
        extra_headers: Mapping[str, str] | None = None,
        provider_name: str = "openai_compatible",
    ) -> None:
        if not base_url:
            raise ProviderError("base_url is required")
        if not model:
            raise ProviderError("model is required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get(env_key)
        self.timeout = timeout
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.extra_headers = dict(extra_headers or {})
        self.name = provider_name

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": coerce_messages(messages),
            "temperature": self.default_temperature if temperature is None else temperature,
            "stream": False,
        }
        resolved_max = self.default_max_tokens if max_tokens is None else max_tokens
        if resolved_max is not None:
            payload["max_tokens"] = resolved_max
        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value

        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:600]
            raise ProviderError(f"HTTP {exc.code} from {self.endpoint}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"cannot reach {self.endpoint}: {exc.reason}") from exc

        try:
            data = json.loads(body)
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content") or message.get("reasoning_content") or ""
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape from {self.endpoint}: {body[:300]}") from exc

        return LLMResponse(
            text=text,
            model=str(data.get("model", self.model)),
            finish_reason=str(choice.get("finish_reason", "")),
            usage=dict(data.get("usage", {}) or {}),
            raw=data if isinstance(data, dict) else {},
        )
