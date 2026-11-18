"""Local embedding generation with Sentence Transformers.

sentence_transformers (and torch) are imported lazily on first use so that the
application starts, and unit tests run, without loading the model.
"""

import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        device: str | None = None,
        query_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._revision = revision
        self._cache_dir = cache_dir
        self._device = device
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._model: Any = None
        self._version: str | None = None
        self._lock = threading.Lock()

    # -- model loading -------------------------------------------------------

    def _load(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s", self._model_name)
                    model = SentenceTransformer(
                        self._model_name,
                        revision=self._revision,
                        cache_folder=self._cache_dir,
                        device=self._device,
                    )
                    self._version = self._resolve_version()
                    self._model = model
        return self._model

    def _resolve_version(self) -> str:
        """Return the Hugging Face commit hash of the cached weights.

        The model has just been loaded, so its snapshot is in the local cache;
        the snapshot directory is named after the commit hash.
        """
        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                self._model_name,
                revision=self._revision,
                cache_dir=self._cache_dir,
                local_files_only=True,
            )
            return Path(path).name
        except Exception:
            # A local model directory or an unexpected cache layout.
            pass
        if self._revision:
            return self._revision
        logger.warning(
            "Could not determine the revision of embedding model %s; recording 'unknown'. "
            "Set EMBEDDING_MODEL_REVISION to pin it.",
            self._model_name,
        )
        return "unknown"

    # -- EmbeddingProvider ---------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        self._load()
        assert self._version is not None
        return self._version

    @property
    def dimensions(self) -> int:
        model = self._load()
        getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension"
        )
        dim = getter()
        if not dim:
            raise RuntimeError(f"Embedding model {self._model_name} reports no dimension")
        return int(dim)

    @property
    def max_seq_length(self) -> int | None:
        value = getattr(self._load(), "max_seq_length", None)
        return int(value) if value else None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        self._warn_on_truncation(model, texts)
        vectors = model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        vector = model.encode(
            self._query_prefix + text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return list(vector.tolist())

    def _warn_on_truncation(self, model: Any, texts: Sequence[str]) -> None:
        """Log when passages exceed the model input length (the excess is ignored)."""
        limit = getattr(model, "max_seq_length", None)
        tokenizer = getattr(model, "tokenizer", None)
        if not limit or tokenizer is None:
            return
        try:
            lengths = [len(ids) for ids in tokenizer(list(texts), verbose=False)["input_ids"]]
        except Exception:  # pragma: no cover - tokenizer specific
            return
        truncated = sum(1 for n in lengths if n > limit)
        if truncated:
            logger.warning(
                "%d of %d passages exceed the %d-token limit of %s and were truncated "
                "(longest: %d tokens). Consider a smaller CHUNK_SIZE.",
                truncated,
                len(texts),
                limit,
                self._model_name,
                max(lengths),
            )
