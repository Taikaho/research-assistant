"""Provider registry built from application configuration.

Provider selection lives here so the RAG pipeline only asks for "a provider by
name" and never constructs adapters itself.
"""

from functools import lru_cache

from app.core.config import settings
from app.llm.base import LLMProvider


class UnknownProviderError(LookupError):
    """The requested provider is not configured (or disabled by local-only mode)."""


class LLMRegistry:
    def __init__(self, providers: list[LLMProvider], default_provider: str) -> None:
        self._providers = {p.name: p for p in providers}
        self.default_provider = default_provider

    @property
    def providers(self) -> list[LLMProvider]:
        return list(self._providers.values())

    def get(self, name: str | None) -> LLMProvider:
        name = name or self.default_provider
        provider = self._providers.get(name)
        if provider is None:
            configured = ", ".join(sorted(self._providers)) or "none"
            raise UnknownProviderError(
                f"LLM provider {name!r} is not configured (configured: {configured})"
            )
        return provider

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()


def create_llm_registry() -> LLMRegistry:
    from app.llm.ollama import OllamaProvider

    providers: list[LLMProvider] = [
        OllamaProvider(
            settings.ollama_base_url,
            default_model=settings.ollama_default_model,
            timeout=settings.ollama_timeout_seconds,
            num_ctx=settings.ollama_num_ctx,
        )
    ]

    # Cloud providers are registered only with credentials and never in local-only
    # mode, so a request can't route document text to the cloud by accident.
    if settings.openai_api_key and not settings.llm_local_only:
        from app.llm.openai_compatible import OpenAICompatibleProvider

        providers.append(
            OpenAICompatibleProvider(
                api_key=settings.openai_api_key.get_secret_value(),
                base_url=settings.openai_base_url,
                provider_name=settings.openai_provider_name,
                default_model=settings.openai_default_model,
                timeout=settings.openai_timeout_seconds,
                max_retries=settings.openai_max_retries,
                token_param=settings.openai_token_param,
            )
        )
    return LLMRegistry(providers, settings.llm_default_provider)


@lru_cache(maxsize=1)
def get_llm_registry() -> LLMRegistry:
    """FastAPI dependency returning the process-wide registry."""
    return create_llm_registry()
