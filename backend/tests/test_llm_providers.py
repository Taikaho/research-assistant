"""LLM adapters (HTTP mocked with httpx.MockTransport) and the provider registry."""

import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import settings
from app.llm.base import (
    ChatMessage,
    GenerationParams,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from app.llm.ollama import OllamaProvider
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.llm.registry import LLMRegistry, UnknownProviderError, create_llm_registry
from tests.fakes import FakeLLMProvider

MESSAGES = [ChatMessage("system", "be grounded"), ChatMessage("user", "question")]
PARAMS = GenerationParams(temperature=0.2, max_tokens=300)

Handler = Callable[[httpx.Request], httpx.Response]


def _ollama(handler: Handler, **kwargs: object) -> OllamaProvider:
    return OllamaProvider(
        "http://ollama.test", transport=httpx.MockTransport(handler), **kwargs  # type: ignore[arg-type]
    )


def _openai(handler: Handler, sleeps: list[float] | None = None, **kwargs: object) -> (
    OpenAICompatibleProvider
):
    async def fake_sleep(seconds: float) -> None:
        if sleeps is not None:
            sleeps.append(seconds)

    return OpenAICompatibleProvider(
        api_key="sk-test-secret",
        base_url="https://cloud.test/v1",
        transport=httpx.MockTransport(handler),
        sleep=fake_sleep,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOllamaProvider:
    async def test_generate_sends_messages_and_options(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "model": "llama3.2:3b",
                    "message": {"role": "assistant", "content": "Answer [1]."},
                    "done_reason": "stop",
                    "prompt_eval_count": 812,
                    "eval_count": 41,
                    "total_duration": 123,
                },
            )

        provider = _ollama(handler, num_ctx=8192)
        result = await provider.generate("llama3.2:3b", MESSAGES, PARAMS)

        body = seen["body"]
        assert seen["path"] == "/api/chat"
        assert isinstance(body, dict)
        assert body["stream"] is False
        assert body["messages"] == [
            {"role": "system", "content": "be grounded"},
            {"role": "user", "content": "question"},
        ]
        assert body["options"] == {"temperature": 0.2, "num_predict": 300, "num_ctx": 8192}
        assert result.text == "Answer [1]."
        assert (result.prompt_tokens, result.completion_tokens) == (812, 41)
        assert result.finish_reason == "stop"
        assert provider.is_local

    async def test_missing_model_raises_model_not_found_with_pull_hint(self) -> None:
        provider = _ollama(lambda r: httpx.Response(404, json={"error": "model 'x' not found"}))
        with pytest.raises(LLMModelNotFoundError, match="ollama pull x"):
            await provider.generate("x", MESSAGES, PARAMS)

    async def test_connection_refused_raises_connection_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(LLMConnectionError, match="Is it running"):
            await _ollama(handler).generate("m", MESSAGES, PARAMS)

    async def test_timeout_raises_timeout_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        with pytest.raises(LLMTimeoutError):
            await _ollama(handler).generate("m", MESSAGES, PARAMS)

    async def test_server_error_raises_response_error(self) -> None:
        provider = _ollama(lambda r: httpx.Response(500, json={"error": "out of memory"}))
        with pytest.raises(LLMResponseError, match="out of memory"):
            await provider.generate("m", MESSAGES, PARAMS)

    async def test_list_models_and_health(self) -> None:
        provider = _ollama(
            lambda r: httpx.Response(200, json={"models": [{"name": "b:1"}, {"name": "a:2"}]})
        )
        assert await provider.list_models() == ["a:2", "b:1"]
        assert (await provider.health()).available

    async def test_health_reports_unavailable_service(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        health = await _ollama(handler).health()
        assert not health.available and health.detail

    async def test_resolve_model_prefers_request_then_default_then_first_installed(self) -> None:
        provider = _ollama(lambda r: httpx.Response(200, json={"models": [{"name": "only:1"}]}))
        assert await provider.resolve_model("asked") == "asked"
        assert await provider.resolve_model(None) == "only:1"
        configured = _ollama(lambda r: httpx.Response(500), default_model="conf")
        assert await configured.resolve_model(None) == "conf"

    async def test_resolve_model_without_installed_models_fails(self) -> None:
        provider = _ollama(lambda r: httpx.Response(200, json={"models": []}))
        with pytest.raises(LLMModelNotFoundError):
            await provider.resolve_model(None)


# ---------------------------------------------------------------------------
# OpenAI-compatible
# ---------------------------------------------------------------------------


def _completion(text: str = "Cloud answer [2].") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "gpt-test-2026",
            "choices": [
                {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 900, "completion_tokens": 30},
        },
    )


@pytest.mark.unit
class TestOpenAICompatibleProvider:
    async def test_generate_request_and_response(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["authorization"]
            seen["body"] = json.loads(request.content)
            return _completion()

        provider = _openai(handler)
        result = await provider.generate("gpt-test", MESSAGES, PARAMS)

        assert seen["url"] == "https://cloud.test/v1/chat/completions"
        assert seen["auth"] == "Bearer sk-test-secret"
        assert seen["body"] == {
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "be grounded"},
                {"role": "user", "content": "question"},
            ],
            "temperature": 0.2,
            "max_tokens": 300,
        }
        assert result.text == "Cloud answer [2]."
        assert (result.prompt_tokens, result.completion_tokens) == (900, 30)
        assert not provider.is_local

    async def test_token_param_is_configurable(self) -> None:
        bodies: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return _completion()

        await _openai(handler, token_param="max_completion_tokens").generate(
            "o-test", MESSAGES, PARAMS
        )
        assert bodies[0]["max_completion_tokens"] == 300
        assert "max_tokens" not in bodies[0]

    async def test_rate_limit_is_retried_honouring_retry_after(self) -> None:
        responses = [
            httpx.Response(429, headers={"retry-after": "3"}, json={"error": {"message": "slow"}}),
            _completion(),
        ]
        sleeps: list[float] = []
        provider = _openai(lambda r: responses.pop(0), sleeps=sleeps)
        result = await provider.generate("m", MESSAGES, PARAMS)
        assert result.text == "Cloud answer [2]."
        assert sleeps == [3.0]

    async def test_rate_limit_after_retries_raises_with_retry_after(self) -> None:
        sleeps: list[float] = []
        provider = _openai(
            lambda r: httpx.Response(
                429, headers={"retry-after": "7"}, json={"error": {"message": "Too many"}}
            ),
            sleeps=sleeps,
            max_retries=2,
        )
        with pytest.raises(LLMRateLimitError) as info:
            await provider.generate("m", MESSAGES, PARAMS)
        assert info.value.retry_after == 7.0
        assert len(sleeps) == 2

    async def test_exhausted_quota_is_not_retried(self) -> None:
        sleeps: list[float] = []
        provider = _openai(
            lambda r: httpx.Response(
                429, json={"error": {"message": "quota", "code": "insufficient_quota"}}
            ),
            sleeps=sleeps,
        )
        with pytest.raises(LLMRateLimitError):
            await provider.generate("m", MESSAGES, PARAMS)
        assert sleeps == []

    async def test_server_errors_are_retried_with_backoff(self) -> None:
        responses = [httpx.Response(503), httpx.Response(502), _completion()]
        sleeps: list[float] = []
        provider = _openai(lambda r: responses.pop(0), sleeps=sleeps)
        await provider.generate("m", MESSAGES, PARAMS)
        assert sleeps == [1.0, 2.0]

    @pytest.mark.parametrize(
        ("status", "error"),
        [
            (401, LLMAuthenticationError),
            (403, LLMAuthenticationError),
            (404, LLMModelNotFoundError),
            (400, LLMResponseError),
        ],
    )
    async def test_client_errors_map_to_provider_neutral_errors(
        self, status: int, error: type[Exception]
    ) -> None:
        provider = _openai(lambda r: httpx.Response(status, json={"error": {"message": "nope"}}))
        with pytest.raises(error):
            await provider.generate("m", MESSAGES, PARAMS)

    async def test_api_key_never_appears_in_errors(self) -> None:
        provider = _openai(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}))
        with pytest.raises(LLMAuthenticationError) as info:
            await provider.generate("m", MESSAGES, PARAMS)
        assert "sk-test-secret" not in str(info.value)

    async def test_timeout_and_connection_errors(self) -> None:
        def timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        def refused(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns", request=request)

        with pytest.raises(LLMTimeoutError):
            await _openai(timeout).generate("m", MESSAGES, PARAMS)
        with pytest.raises(LLMConnectionError):
            await _openai(refused).generate("m", MESSAGES, PARAMS)

    async def test_malformed_response_raises_response_error(self) -> None:
        provider = _openai(lambda r: httpx.Response(200, json={"choices": []}))
        with pytest.raises(LLMResponseError):
            await provider.generate("m", MESSAGES, PARAMS)

    async def test_list_models(self) -> None:
        provider = _openai(
            lambda r: httpx.Response(200, json={"data": [{"id": "z"}, {"id": "a"}]})
        )
        assert await provider.list_models() == ["a", "z"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegistry:
    def test_get_uses_default_and_rejects_unknown(self) -> None:
        local = FakeLLMProvider(name="local")
        registry = LLMRegistry([local], default_provider="local")
        assert registry.get(None) is local
        assert registry.get("local") is local
        with pytest.raises(UnknownProviderError, match="configured: local"):
            registry.get("openai")

    def test_cloud_provider_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "openai_api_key", None)
        monkeypatch.setattr(settings, "llm_local_only", False)
        names = [p.name for p in create_llm_registry().providers]
        assert names == ["ollama"]

    def test_cloud_provider_registered_with_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "openai_api_key", SecretStr("sk-x"))
        monkeypatch.setattr(settings, "openai_provider_name", "openai")
        monkeypatch.setattr(settings, "llm_local_only", False)
        registry = create_llm_registry()
        assert [(p.name, p.is_local) for p in registry.providers] == [
            ("ollama", True),
            ("openai", False),
        ]

    def test_local_only_mode_never_registers_cloud_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "openai_api_key", SecretStr("sk-x"))
        monkeypatch.setattr(settings, "llm_local_only", True)
        registry = create_llm_registry()
        assert all(p.is_local for p in registry.providers)
        with pytest.raises(UnknownProviderError):
            registry.get("openai")
