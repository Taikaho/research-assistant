from pydantic import BaseModel


class ProviderInfo(BaseModel):
    name: str
    is_local: bool
    default_model: str | None
    is_default: bool


class ProvidersResponse(BaseModel):
    default_provider: str
    # When true, cloud providers are disabled and no document text leaves this machine.
    local_only: bool
    providers: list[ProviderInfo]


class ProviderModels(ProviderInfo):
    available: bool
    models: list[str]
    error: str | None = None


class ModelsResponse(BaseModel):
    default_provider: str
    local_only: bool
    providers: list[ProviderModels]


class ProviderHealthInfo(BaseModel):
    name: str
    is_local: bool
    available: bool
    detail: str | None = None


class ModelsHealthResponse(BaseModel):
    providers: list[ProviderHealthInfo]
