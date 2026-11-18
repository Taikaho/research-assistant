"""Ollama adapter (local models) using the native /api/chat endpoint."""

from typing import Any

import httpx

from app.llm.base import (
    ChatMessage,
    GenerationParams,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMProvider,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
)

# Nanosecond timings Ollama reports with each response.
_TIMING_FIELDS = ("total_duration", "load_duration", "eval_duration")


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        base_url: str,
        default_model: str | None = None,
        timeout: float = 180.0,
        num_ctx: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._default_model = default_model
        self._num_ctx = num_ctx
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            # Generous read timeout: a cold model load plus a long answer on CPU can take
            # minutes. Connecting should be near-instant for a local service.
            timeout=httpx.Timeout(timeout, connect=5.0),
            transport=transport,
        )

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def is_local(self) -> bool:
        return True

    @property
    def default_model(self) -> str | None:
        return self._default_model

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Ollama did not respond in time ({path})") from exc
        except httpx.TransportError as exc:
            raise LLMConnectionError(
                f"Cannot reach Ollama at {self._client.base_url}. Is it running?"
            ) from exc

        if response.status_code >= 400:
            message = _error_message(response)
            if response.status_code == 404:
                raise LLMModelNotFoundError(message)
            raise LLMResponseError(f"Ollama error {response.status_code}: {message}")
        try:
            return response.json()
        except ValueError as exc:
            raise LLMResponseError("Ollama returned a non-JSON response") from exc

    async def generate(
        self, model: str, messages: list[ChatMessage], params: GenerationParams
    ) -> LLMResponse:
        options: dict[str, Any] = {
            "temperature": params.temperature,
            "num_predict": params.max_tokens,
        }
        if self._num_ctx:
            options["num_ctx"] = self._num_ctx
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": options,
        }
        try:
            data = await self._request("POST", "/api/chat", json=payload)
        except LLMModelNotFoundError as exc:
            raise LLMModelNotFoundError(
                f"Ollama model {model!r} is not installed. Run: ollama pull {model}"
            ) from exc

        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMResponseError("Ollama response has no message content") from exc
        return LLMResponse(
            text=text,
            model=data.get("model", model),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            finish_reason=data.get("done_reason"),
            raw_metadata={k: data[k] for k in _TIMING_FIELDS if k in data},
        )

    async def list_models(self) -> list[str]:
        data = await self._request("GET", "/api/tags")
        try:
            return sorted(m["name"] for m in data.get("models", []))
        except (KeyError, TypeError) as exc:
            raise LLMError("Unexpected /api/tags response from Ollama") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500] or response.reason_phrase
    if isinstance(body, dict) and isinstance(body.get("error"), str):
        return str(body["error"])
    return str(body)[:500]
