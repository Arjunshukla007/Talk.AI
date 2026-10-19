"""Thin LLM clients for OpenAI and Anthropic with retries, streaming and friendly errors."""
from __future__ import annotations

import time
from typing import Callable, Iterator, Protocol, TypeVar

from talkai.config import Settings

T = TypeVar("T")
RETRY_CODES = {429, 500, 502, 503, 529}


class LLMError(Exception):
    """The model provider failed (auth, rate limit, network...). Message is user-safe."""


class LLMClient(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 900) -> str: ...
    def stream(self, system: str, user: str, max_tokens: int = 900) -> Iterator[str]: ...


def _is_retriable(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) in RETRY_CODES or type(exc).__name__ in {
        "APIConnectionError", "APITimeoutError",
    }


def with_retry(fn: Callable[[], T], attempts: int = 3, base_delay: float = 1.0, sleep=time.sleep) -> T:
    """Retry transient failures (rate limits, 5xx, network) with exponential backoff."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if i == attempts - 1 or not _is_retriable(exc):
                raise
            sleep(base_delay * 2**i)
    raise RuntimeError("unreachable")


def _friendly(provider: str, exc: Exception) -> LLMError:
    code = getattr(exc, "status_code", None)
    if code in (401, 403):
        return LLMError(f"{provider} rejected the API key. Check that it is correct and has access to the model.")
    if code == 404:
        return LLMError(f"{provider} could not find the configured model name. Check OPENAI_MODEL / ANTHROPIC_MODEL.")
    if code == 429:
        return LLMError(f"{provider} rate limit or quota reached. Wait a moment or check your billing.")
    return LLMError(f"{provider} request failed: {exc}")


class OpenAIChat:
    def __init__(self, settings: Settings):
        from openai import OpenAI

        self.model = settings.openai_model
        self.client = OpenAI(api_key=settings.openai_api_key, timeout=60)

    def _messages(self, system: str, user: str) -> list[dict]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def complete(self, system: str, user: str, max_tokens: int = 900) -> str:
        try:
            resp = with_retry(lambda: self.client.chat.completions.create(
                model=self.model, temperature=0.1, max_tokens=max_tokens, messages=self._messages(system, user)))
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            raise _friendly("OpenAI", exc) from exc

    def stream(self, system: str, user: str, max_tokens: int = 900) -> Iterator[str]:
        try:
            events = self.client.chat.completions.create(
                model=self.model, temperature=0.1, max_tokens=max_tokens, stream=True,
                messages=self._messages(system, user))
            for event in events:
                if event.choices and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
        except Exception as exc:
            raise _friendly("OpenAI", exc) from exc


class AnthropicChat:
    def __init__(self, settings: Settings):
        import anthropic

        self.model = settings.anthropic_model
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=60)

    def _kwargs(self, system: str, user: str, max_tokens: int) -> dict:
        return dict(model=self.model, max_tokens=max_tokens, temperature=0.1, system=system,
                    messages=[{"role": "user", "content": user}])

    def complete(self, system: str, user: str, max_tokens: int = 900) -> str:
        try:
            resp = with_retry(lambda: self.client.messages.create(**self._kwargs(system, user, max_tokens)))
            return "".join(b.text for b in resp.content if getattr(b, "text", None)).strip()
        except Exception as exc:
            raise _friendly("Anthropic", exc) from exc

    def stream(self, system: str, user: str, max_tokens: int = 900) -> Iterator[str]:
        try:
            with self.client.messages.stream(**self._kwargs(system, user, max_tokens)) as s:
                yield from s.text_stream
        except Exception as exc:
            raise _friendly("Anthropic", exc) from exc


def build_llm(settings: Settings) -> LLMClient:
    settings.validate_llm()
    return OpenAIChat(settings) if settings.provider == "openai" else AnthropicChat(settings)
