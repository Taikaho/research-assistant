"""Embedding provider interface.

Business logic depends only on this interface, never on a concrete library.
Implementations are synchronous and may be CPU/GPU bound; async callers should
run them in a worker thread (see app.embeddings.service).
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingProvider(ABC):
    """Generates document and query embeddings in a single vector space."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier, e.g. a Hugging Face repo id."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Identifies the exact weights (e.g. commit hash). May load the model."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Output vector dimension. May load the model."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages for storage. Vectors are L2-normalised."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a search query. Vectors are L2-normalised."""
