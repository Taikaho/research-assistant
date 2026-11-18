"""Adapter for OpenAI-compatible chat-completion APIs (cloud models).

Works with any service exposing /chat/completions and /models in the OpenAI
format (OpenAI, OpenRouter, Groq, Together, vLLM, LM Studio, ...). The API key
is read from settings (environment) and is never logged or returned.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from app.llm.base import (
    ChatMessage,
    GenerationParams,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
)

# Upper bound on a single wait between retries, whatever Retry-After says.
MAX_RETRY_WAIT_SECONDS = 30.0


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        provider_name: str = "openai",
        default_model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        token_param: str = "max_tokens",
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._name = provider_name
        self._default_model = default_model
        self._max_retries = max(0, max_retries)
        self._token_param = token_param
        self._sleep = sleep
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=10.0),
            transport=transport,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_local(self) -> bool:
        return False

    @property
    def default_model(self) -> str | None:
        return self._default_model

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        attempt = 0
        while True:
            try:
                response = await self._client.request(method, path, **kwargs)
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(f"{self._name} did not respond in time") from exc
            except httpx.TransportError as exc:
                raise LLMConnectionError(f"Cannot reach {self._name} API") from exc

            if response.status_code < 400:
                try:
                    return response.json()
                except ValueError as exc:
                    raise LLMResponseError(f"{self._name} returned a non-JSON response") from exc

            message, code = _error_details(response)
            retryable = (response.status_code == 429 and code != "insufficient_quota") or (
                response.status_code >= 500
            )
            if retryable and attempt < self._max_retries:
                await self._sleep(_retry_delay(response, attempt))
                attempt += 1
                continue

            status = response.status_code
            if status in (401, 403):
                raise LLMAuthenticationError(f"{self._name} rejected the credentials: {message}")
            if status == 404:
                raise LLMModelNotFoundError(f"{self._name}: {message}")
            if status == 429:
                raise LLMRateLimitError(
                    f"{self._name} rate limit or quota exceeded: {message}",
                    retry_after=_retry_after(response),
                )
            raise LLMResponseError(f"{self._name} error {status}: {message}")

    async def generate(
        self, model: str, messages: list[ChatMessage], params: GenerationParams
    ) -> LLMResponse:
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": params.temperature,
            self._token_param: params.max_tokens,
        }
        data = await self._request("POST", "/chat/completions", json=payload)
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(f"{self._name} response has no message content") from exc
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
        )

    async def list_models(self) -> list[str]:
        data = await self._request("GET", "/models")
        try:
            return sorted(m["id"] for m in data.get("data", []))
        except (KeyError, TypeError) as exc:
            raise LLMResponseError(f"Unexpected /models response from {self._name}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


def _error_details(response: httpx.Response) -> tuple[str, str | None]:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500] or response.reason_phrase, None
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)[:500], error.get("code")
    if isinstance(error, str):
        return error[:500], None
    return str(body)[:500], None


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # HTTP-date form; fall back to exponential backoff


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    delay = _retry_after(response)
    if delay is None:
        delay = 1.0 * 2**attempt
    return min(delay, MAX_RETRY_WAIT_SECONDS)
