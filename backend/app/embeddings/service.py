"""Embedding orchestration: provider selection, model registry and (re)indexing.

Model inference runs in a worker thread so it never blocks the event loop.
"""

import asyncio
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.models import Document, DocumentChunk, EmbeddingModel
from app.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

# Chunks loaded and committed per re-index batch.
REINDEX_BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def create_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider == "sentence_transformers":
        from app.embeddings.sentence_transformer import SentenceTransformerProvider

        return SentenceTransformerProvider(
            settings.embedding_model_name,
            revision=settings.embedding_model_revision,
            cache_dir=settings.embedding_cache_dir,
            device=settings.embedding_device,
            query_prefix=settings.embedding_query_prefix,
            batch_size=settings.embedding_batch_size,
        )
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider!r}")


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """FastAPI dependency returning the process-wide provider (model loads lazily)."""
    return create_embedding_provider()


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


async def register_model(db: AsyncSession, provider: EmbeddingProvider) -> EmbeddingModel:
    """Return the registry row for the provider's model, creating it if needed.

    The row is marked active and every other model inactive. Does not commit.
    Loads the model (in a worker thread) to resolve its version and dimension.
    """
    name = provider.model_name
    version, dimensions = await asyncio.to_thread(
        lambda: (provider.model_version, provider.dimensions)
    )

    await db.execute(
        pg_insert(EmbeddingModel)
        .values(id=uuid.uuid4(), name=name, version=version, dimensions=dimensions)
        .on_conflict_do_nothing(constraint="uq_embedding_model_name_version")
    )
    model = (
        await db.execute(
            select(EmbeddingModel).where(
                EmbeddingModel.name == name, EmbeddingModel.version == version
            )
        )
    ).scalar_one()

    if model.dimensions != dimensions:
        raise RuntimeError(
            f"Embedding model {name}@{version} is registered with {model.dimensions} "
            f"dimensions but produced {dimensions}."
        )

    if not model.is_active:
        await db.execute(
            update(EmbeddingModel)
            .where(EmbeddingModel.id != model.id, EmbeddingModel.is_active.is_(True))
            .values(is_active=False)
        )
        model.is_active = True
    return model


# ---------------------------------------------------------------------------
# Embedding chunks
# ---------------------------------------------------------------------------


async def embed_chunks(
    provider: EmbeddingProvider,
    model: EmbeddingModel,
    chunks: Sequence[DocumentChunk],
) -> None:
    """Compute and assign embeddings for the given chunks. Does not commit."""
    if not chunks:
        return
    vectors = await asyncio.to_thread(provider.embed_documents, [c.text for c in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError(f"Expected {len(chunks)} embeddings, got {len(vectors)}")
    for chunk, vector in zip(chunks, vectors, strict=True):
        if len(vector) != model.dimensions:
            raise RuntimeError(
                f"Embedding has {len(vector)} dimensions, expected {model.dimensions}"
            )
        chunk.embedding = vector
        chunk.embedding_model_id = model.id


def _stale_chunk_filter(model_id: uuid.UUID):  # type: ignore[no-untyped-def]
    """Chunks that have no embedding from the given model."""
    return or_(
        DocumentChunk.embedding.is_(None),
        DocumentChunk.embedding_model_id.is_(None),
        DocumentChunk.embedding_model_id != model_id,
    )


@dataclass
class ReindexResult:
    model: EmbeddingModel
    documents: int
    chunks: int


async def reindex(
    db: AsyncSession,
    provider: EmbeddingProvider,
    document_id: uuid.UUID | None = None,
) -> ReindexResult:
    """Embed every completed document's chunks that lack a current-model embedding.

    Commits after each batch so progress survives interruption; re-running is
    safe and only processes what is still stale.
    """
    model = await register_model(db, provider)
    await db.commit()

    base = (
        select(DocumentChunk)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(Document.processing_status == "completed", _stale_chunk_filter(model.id))
    )
    if document_id is not None:
        base = base.where(DocumentChunk.document_id == document_id)

    total_chunks = 0
    documents: set[uuid.UUID] = set()
    while True:
        batch = (
            (
                await db.execute(
                    base.order_by(DocumentChunk.document_id, DocumentChunk.chunk_index).limit(
                        REINDEX_BATCH_SIZE
                    )
                )
            )
            .scalars()
            .all()
        )
        if not batch:
            break
        await embed_chunks(provider, model, batch)
        await db.commit()
        total_chunks += len(batch)
        documents.update(c.document_id for c in batch)
        logger.info("Re-indexed %d chunks so far", total_chunks)

    return ReindexResult(model=model, documents=len(documents), chunks=total_chunks)


@dataclass
class IndexStatus:
    model: EmbeddingModel | None
    total_chunks: int
    indexed_chunks: int


async def index_status(db: AsyncSession) -> IndexStatus:
    """Chunk counts relative to the active model (completed documents only)."""
    model = (
        await db.execute(select(EmbeddingModel).where(EmbeddingModel.is_active.is_(True)))
    ).scalar_one_or_none()

    total_stmt = (
        select(func.count())
        .select_from(DocumentChunk)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(Document.processing_status == "completed")
    )
    total = (await db.execute(total_stmt)).scalar_one()
    indexed = 0
    if model is not None:
        indexed = (
            await db.execute(
                total_stmt.where(
                    DocumentChunk.embedding_model_id == model.id,
                    DocumentChunk.embedding.is_not(None),
                )
            )
        ).scalar_one()
    return IndexStatus(model=model, total_chunks=total, indexed_chunks=indexed)
