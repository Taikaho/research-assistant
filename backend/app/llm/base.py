"""LLM provider interface and provider-neutral errors.

The RAG pipeline depends only on this module. It never knows whether a provider
runs locally or in the cloud, and never sees provider-specific exceptions:
adapters translate their failures into the LLMError hierarchy below.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class GenerationParams:
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    # Token counts as reported by the provider; None when it does not report them.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # e.g. "stop", "length"; None when unknown.
    finish_reason: str | None = None
    raw_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderHealth:
    available: bool
    detail: str | None = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base class for all provider failures."""


class LLMConnectionError(LLMError):
    """The provider could not be reached (service down, DNS, refused connection)."""


class LLMTimeoutError(LLMError):
    """The provider did not respond in time."""


class LLMModelNotFoundError(LLMError):
    """The requested model is not installed / not offered by the provider."""


class LLMAuthenticationError(LLMError):
    """Credentials are missing, invalid or lack permission."""


class LLMRateLimitError(LLMError):
    """The provider rejected the request because of rate limits or quota."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMResponseError(LLMError):
    """The provider returned an error or a response that could not be used."""


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """A chat-completion backend (local or cloud)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier used in API requests, e.g. "ollama"."""

    @property
    @abstractmethod
    def is_local(self) -> bool:
        """True when prompts never leave this machine."""

    @property
    @abstractmethod
    def default_model(self) -> str | None:
        """Configured default model, if any."""

    @abstractmethod
    async def generate(
        self, model: str, messages: list[ChatMessage], params: GenerationParams
    ) -> LLMResponse:
        """Generate a reply. Raises LLMError subclasses on failure."""

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Models currently available. Raises LLMError subclasses on failure."""

    async def health(self) -> ProviderHealth:
        try:
            await self.list_models()
        except LLMError as exc:
            return ProviderHealth(available=False, detail=str(exc))
        return ProviderHealth(available=True)

    async def resolve_model(self, requested: str | None) -> str:
        """Pick the model to use: the requested one, else the default, else the first listed."""
        if requested:
            return requested
        if self.default_model:
            return self.default_model
        models = await self.list_models()
        if not models:
            raise LLMModelNotFoundError(f"Provider {self.name!r} has no models available")
        return models[0]

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""
