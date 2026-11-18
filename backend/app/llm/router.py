import asyncio

from fastapi import APIRouter, Depends

from app.core.config import settings
from app.llm.base import LLMError, LLMProvider
from app.llm.registry import LLMRegistry, get_llm_registry
from app.llm.schemas import (
    ModelsHealthResponse,
    ModelsResponse,
    ProviderHealthInfo,
    ProviderInfo,
    ProviderModels,
    ProvidersResponse,
)

router = APIRouter(prefix="/models", tags=["models"])


def _info(provider: LLMProvider, registry: LLMRegistry) -> ProviderInfo:
    return ProviderInfo(
        name=provider.name,
        is_local=provider.is_local,
        default_model=provider.default_model,
        is_default=provider.name == registry.default_provider,
    )


@router.get("", response_model=ModelsResponse)
async def list_models(registry: LLMRegistry = Depends(get_llm_registry)) -> ModelsResponse:
    """Configured providers and the models each currently offers."""

    async def describe(provider: LLMProvider) -> ProviderModels:
        base = _info(provider, registry).model_dump()
        try:
            models = await provider.list_models()
        except LLMError as exc:
            return ProviderModels(**base, available=False, models=[], error=str(exc))
        return ProviderModels(**base, available=True, models=models)

    providers = await asyncio.gather(*(describe(p) for p in registry.providers))
    return ModelsResponse(
        default_provider=registry.default_provider,
        local_only=settings.llm_local_only,
        providers=list(providers),
    )


@router.get("/providers", response_model=ProvidersResponse)
async def list_providers(registry: LLMRegistry = Depends(get_llm_registry)) -> ProvidersResponse:
    """Configured providers (no network calls). Credentials are never included."""
    return ProvidersResponse(
        default_provider=registry.default_provider,
        local_only=settings.llm_local_only,
        providers=[_info(p, registry) for p in registry.providers],
    )


@router.get("/health", response_model=ModelsHealthResponse)
async def providers_health(
    registry: LLMRegistry = Depends(get_llm_registry),
) -> ModelsHealthResponse:
    async def check(provider: LLMProvider) -> ProviderHealthInfo:
        health = await provider.health()
        return ProviderHealthInfo(
            name=provider.name,
            is_local=provider.is_local,
            available=health.available,
            detail=health.detail,
        )

    results = await asyncio.gather(*(check(p) for p in registry.providers))
    return ModelsHealthResponse(providers=list(results))
