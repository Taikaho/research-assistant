"""Test doubles."""

import hashlib
import math
import re
from collections.abc import Sequence

from app.embeddings.base import EmbeddingProvider
from app.llm.base import ChatMessage, GenerationParams, LLMProvider, LLMResponse

_WORD = re.compile(r"\w+")


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic bag-of-words embeddings; no model download.

    Each word is hashed into one of `dimensions` buckets, so texts that share
    words have high cosine similarity. Good enough to test ranking and scoping.
    """

    def __init__(self, dimensions: int = 64, name: str = "fake-model", version: str = "1") -> None:
        self._dimensions = dimensions
        self._name = name
        self._version = version
        self.document_calls = 0
        self.query_calls = 0

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for word in _WORD.findall(text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._embed(text)


class FakeLLMProvider(LLMProvider):
    """Scripted LLM: returns queued replies (or raises queued errors) and records calls."""

    def __init__(
        self,
        name: str = "fake-llm",
        is_local: bool = True,
        models: Sequence[str] = ("fake-model-1", "fake-model-2"),
        default_model: str | None = "fake-model-1",
    ) -> None:
        self._name = name
        self._is_local = is_local
        self._models = list(models)
        self._default_model = default_model
        self.replies: list[str | Exception] = []
        self.default_reply = "Linear discriminant analysis separates classes [1]."
        self.calls: list[tuple[str, list[ChatMessage], GenerationParams]] = []
        self.list_error: Exception | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_local(self) -> bool:
        return self._is_local

    @property
    def default_model(self) -> str | None:
        return self._default_model

    async def generate(
        self, model: str, messages: list[ChatMessage], params: GenerationParams
    ) -> LLMResponse:
        self.calls.append((model, messages, params))
        reply = self.replies.pop(0) if self.replies else self.default_reply
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(
            text=reply, model=model, prompt_tokens=100, completion_tokens=20, finish_reason="stop"
        )

    async def list_models(self) -> list[str]:
        if self.list_error is not None:
            raise self.list_error
        return list(self._models)
