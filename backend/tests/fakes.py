"""Test doubles."""

import hashlib
import math
import re
from collections.abc import Sequence

from app.embeddings.base import EmbeddingProvider

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
