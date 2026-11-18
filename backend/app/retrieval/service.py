"""Semantic retrieval over document chunks with pgvector.

Guarantees:
- Only chunks of completed documents that belong to at least one selected
  collection are considered (enforced in SQL, not post-filtered).
- Only chunks embedded by the active model are compared with the query, so
  vectors from different models are never mixed.
- Near-duplicate passages are removed before the top-k cut.
"""

import asyncio
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.models import CollectionDocument, Document, DocumentChunk, EmbeddingModel
from app.embeddings.base import EmbeddingProvider
from app.embeddings.service import register_model


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    document: Document
    score: float


@dataclass
class RetrievalResult:
    model: EmbeddingModel
    chunks: list[RetrievedChunk]
    unindexed_document_count: int


def _in_scope(collection_ids: Sequence[uuid.UUID]):  # type: ignore[no-untyped-def]
    """EXISTS filter: the chunk's document is in at least one selected collection."""
    return exists().where(
        CollectionDocument.document_id == DocumentChunk.document_id,
        CollectionDocument.collection_id.in_(collection_ids),
    )


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------

_WORD = re.compile(r"\w+")


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def remove_near_duplicates(
    candidates: Sequence[RetrievedChunk], threshold: float, limit: int
) -> list[RetrievedChunk]:
    """Keep the highest-scoring passage of each near-duplicate group, up to `limit`.

    Candidates must be sorted by descending score.
    """
    kept: list[RetrievedChunk] = []
    kept_shingles: list[set[tuple[str, ...]]] = []
    for candidate in candidates:
        sh = _shingles(candidate.chunk.text)
        if any(jaccard(sh, other) >= threshold for other in kept_shingles):
            continue
        kept.append(candidate)
        kept_shingles.append(sh)
        if len(kept) >= limit:
            break
    return kept


def merge_results(
    result_lists: Sequence[Sequence[RetrievedChunk]], limit: int
) -> list[RetrievedChunk]:
    """Combine results of several queries: best score per chunk, best first, up to `limit`.

    Near-duplicates across the lists are removed the same way as within one search.
    """
    best: dict[uuid.UUID, RetrievedChunk] = {}
    for results in result_lists:
        for item in results:
            current = best.get(item.chunk.id)
            if current is None or item.score > current.score:
                best[item.chunk.id] = item
    ranked = sorted(best.values(), key=lambda c: (-c.score, str(c.chunk.id)))
    return remove_near_duplicates(ranked, settings.retrieval_dedup_threshold, limit)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


async def search(
    db: AsyncSession,
    provider: EmbeddingProvider,
    query: str,
    collection_ids: Sequence[uuid.UUID],
    top_k: int | None = None,
    min_score: float | None = None,
) -> RetrievalResult:
    top_k = top_k or settings.retrieval_top_k
    min_score = settings.retrieval_min_score if min_score is None else min_score
    collection_ids = list(dict.fromkeys(collection_ids))

    model = await register_model(db, provider)
    await db.commit()
    query_vector = await asyncio.to_thread(provider.embed_query, query)

    distance = DocumentChunk.embedding.cosine_distance(query_vector)
    candidate_limit = top_k * max(1, settings.retrieval_candidate_multiplier)
    stmt = (
        select(DocumentChunk, Document, distance.label("distance"))
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            Document.processing_status == "completed",
            DocumentChunk.embedding_model_id == model.id,
            DocumentChunk.embedding.is_not(None),
            _in_scope(collection_ids),
        )
        .order_by(distance, DocumentChunk.id)
        .limit(candidate_limit)
    )
    rows = (await db.execute(stmt)).all()

    candidates = [
        RetrievedChunk(chunk=chunk, document=document, score=1.0 - float(dist))
        for chunk, document, dist in rows
    ]
    candidates = [c for c in candidates if c.score >= min_score]
    results = remove_near_duplicates(candidates, settings.retrieval_dedup_threshold, top_k)

    unindexed = await count_unindexed_documents(db, model.id, collection_ids)
    return RetrievalResult(model=model, chunks=results, unindexed_document_count=unindexed)


async def count_unindexed_documents(
    db: AsyncSession, model_id: uuid.UUID, collection_ids: Sequence[uuid.UUID]
) -> int:
    """Completed in-scope documents with at least one chunk the model has not embedded."""
    stmt = (
        select(func.count(func.distinct(DocumentChunk.document_id)))
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            Document.processing_status == "completed",
            _in_scope(collection_ids),
            (DocumentChunk.embedding_model_id.is_distinct_from(model_id))
            | DocumentChunk.embedding.is_(None),
        )
    )
    return int((await db.execute(stmt)).scalar_one())
