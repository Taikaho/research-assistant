"""Evaluation run orchestration.

1. Prepare the corpus: ingest the dataset PDFs into an `eval:<dataset>` collection
   (re-processing when the chunking settings differ) and embed every chunk with
   the experiment's embedding model.
2. Check the dataset against the extracted text (pages and passages exist).
3. Retrieval: search each question, judge each retrieved chunk against the
   dataset passages, compute Recall@k, Precision@k, hit rate, MRR and the
   relevance-gate calibration.
4. Generation (optional): ask each question through the production chat
   pipeline and measure abstention, citation correctness, latency and tokens;
   optionally score answers with an LLM judge.

Everything runs against the database the session is bound to. Use a dedicated
evaluation database: switching the embedding model re-embeds every chunk there.
"""

import hashlib
import io
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat import service as chat_service
from app.chat.schemas import ChatRequest
from app.collections.schemas import CollectionCreate
from app.collections.service import create_collection, link_document
from app.core.config import Settings
from app.database.models import Collection, DocumentChunk, DocumentPage, EmbeddingModel, Message
from app.documents import service as documents_service
from app.embeddings.base import EmbeddingProvider
from app.embeddings.service import reindex
from app.evaluation.config import ExperimentConfig
from app.evaluation.dataset import LoadedDataset, Question
from app.evaluation.judge import judge_answer
from app.evaluation.matching import ChunkRef, matched_items, passage_coverage, passage_on_page
from app.evaluation.metrics import hit_at_k, precision_at_k, recall_at_k, reciprocal_rank
from app.llm.base import LLMError
from app.llm.registry import LLMRegistry
from app.retrieval import service as retrieval

logger = logging.getLogger(__name__)

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Corpus preparation
# ---------------------------------------------------------------------------


@dataclass
class Corpus:
    collection_id: uuid.UUID
    # Database document id -> dataset document id.
    documents: dict[uuid.UUID, str]
    embedding_model: EmbeddingModel
    reprocessed: list[str] = field(default_factory=list)


async def _get_or_create_collection(db: AsyncSession, name: str) -> Collection:
    existing = (
        await db.execute(select(Collection).where(Collection.name == name))
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    return await create_collection(
        db, CollectionCreate(name=name, description="Evaluation corpus (created by app.evaluation)")
    )


async def _chunking_matches(db: AsyncSession, document_id: uuid.UUID, settings: Settings) -> bool:
    meta = (
        await db.execute(
            select(DocumentChunk.chunk_metadata)
            .where(DocumentChunk.document_id == document_id)
            .limit(1)
        )
    ).scalar_one_or_none()
    if not isinstance(meta, dict):
        return True  # no chunks (e.g. an image-only PDF): nothing to re-chunk
    return bool(
        meta.get("chunk_size") == settings.chunk_size
        and meta.get("chunk_overlap") == settings.chunk_overlap
    )


async def prepare_corpus(
    db: AsyncSession,
    loaded: LoadedDataset,
    embedder: EmbeddingProvider,
    settings: Settings,
    progress: Progress = _noop,
) -> Corpus:
    collection = await _get_or_create_collection(db, f"eval:{loaded.dataset.name}")
    documents: dict[uuid.UUID, str] = {}
    reprocessed: list[str] = []

    for spec in loaded.dataset.documents:
        path = loaded.pdf_path(spec.id)
        data = path.read_bytes()
        existing = await documents_service.get_document_by_hash(
            db, hashlib.sha256(data).hexdigest()
        )
        if existing is None:
            progress(f"Ingesting {spec.file}")
            upload = UploadFile(io.BytesIO(data), filename=path.name)
            document, _ = await documents_service.ingest_document(
                db, upload, embedder, [collection.id]
            )
        else:
            document = existing
            await link_document(db, [collection.id], document.id)
            await db.commit()
            if document.processing_status != "completed" or not await _chunking_matches(
                db, document.id, settings
            ):
                progress(f"Re-processing {spec.file} with the current chunking settings")
                document = await documents_service.reprocess_document(db, document, embedder)
                reprocessed.append(spec.id)
        if document.processing_status != "completed":
            raise RuntimeError(f"Document {spec.file} could not be processed")
        documents[document.id] = spec.id

    progress(f"Embedding chunks with {embedder.model_name} (only those not yet embedded)")
    result = await reindex(db, embedder)
    return Corpus(collection.id, documents, result.model, reprocessed)


# ---------------------------------------------------------------------------
# Dataset checks
# ---------------------------------------------------------------------------


async def check_dataset(
    db: AsyncSession, loaded: LoadedDataset, corpus: Corpus, threshold: float
) -> list[dict[str, Any]]:
    """Find dataset entries that do not match the extracted text.

    A passage that is not on its stated page (or not in the document at all)
    makes every retrieval judgement for that question wrong, so these are
    reported before any metric is trusted.
    """
    by_dataset_id = {v: k for k, v in corpus.documents.items()}
    issues: list[dict[str, Any]] = []
    for q in loaded.dataset.questions:
        for r in q.relevant:
            document_id = by_dataset_id[r.document]
            pages = {
                p.pdf_page_number: p.extracted_text
                for p in (
                    await db.execute(
                        select(DocumentPage).where(DocumentPage.document_id == document_id)
                    )
                ).scalars()
            }
            if r.page_index not in pages:
                issues.append(
                    {"question": q.id, "problem": f"PDF page {r.pdf_page} does not exist"}
                )
                continue
            if r.passage is None:
                continue
            if passage_on_page(
                pages[r.page_index], pages.get(r.page_index + 1, ""), r.passage, threshold
            ):
                continue
            found = [
                n + 1
                for n, t in sorted(pages.items())
                if passage_coverage(t, r.passage) >= threshold
            ]
            issues.append(
                {
                    "question": q.id,
                    "problem": (
                        f"passage not found on PDF page {r.pdf_page}"
                        + (
                            f"; found on page(s) {found}"
                            if found
                            else "; not found in the document"
                        )
                    ),
                }
            )
    return issues


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


async def evaluate_retrieval(
    db: AsyncSession,
    embedder: EmbeddingProvider,
    questions: Sequence[Question],
    corpus: Corpus,
    config: ExperimentConfig,
    progress: Progress = _noop,
) -> list[dict[str, Any]]:
    threshold = config.retrieval.passage_match_threshold
    records = []
    for i, q in enumerate(questions, start=1):
        progress(f"Retrieval {i}/{len(questions)}: {q.id}")
        started = time.perf_counter()
        result = await retrieval.search(
            db,
            embedder,
            q.question,
            [corpus.collection_id],
            top_k=config.search_depth,
            min_score=-1.0,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        ranked: list[set[int]] = []
        hits = []
        for rank, c in enumerate(result.chunks, start=1):
            ref = ChunkRef(
                corpus.documents.get(c.document.id), c.chunk.page_numbers or [], c.chunk.text
            )
            matched = matched_items(ref, q.relevant, threshold)
            ranked.append(matched)
            hits.append(
                {
                    "rank": rank,
                    "document": ref.dataset_document,
                    "pdf_pages": [p + 1 for p in ref.pages],
                    "chunk_index": c.chunk.chunk_index,
                    "score": round(c.score, 4),
                    "matched": sorted(matched),
                }
            )

        record: dict[str, Any] = {
            "id": q.id,
            "language": q.language,
            "answerable": q.answerable,
            "category": q.category,
            "top_score": round(result.chunks[0].score, 4) if result.chunks else None,
            "latency_ms": round(elapsed_ms, 1),
            "results": hits,
        }
        if q.answerable:
            record["metrics"] = {
                **{
                    f"recall@{k}": recall_at_k(ranked, len(q.relevant), k)
                    for k in config.retrieval.k_values
                },
                **{f"precision@{k}": precision_at_k(ranked, k) for k in config.retrieval.k_values},
                **{f"hit@{k}": hit_at_k(ranked, k) for k in config.retrieval.k_values},
                "rr": reciprocal_rank(ranked),
            }
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


async def evaluate_generation(
    db: AsyncSession,
    embedder: EmbeddingProvider,
    registry: LLMRegistry,
    questions: Sequence[Question],
    corpus: Corpus,
    config: ExperimentConfig,
    progress: Progress = _noop,
) -> list[dict[str, Any]]:
    gen = config.generation
    threshold = config.retrieval.passage_match_threshold
    judge = config.judge
    judge_provider = registry.get(judge.provider) if judge.enabled else None
    judge_model = await judge_provider.resolve_model(judge.model) if judge_provider else None

    records = []
    for i, q in enumerate(questions, start=1):
        progress(f"Generation {i}/{len(questions)}: {q.id}")
        record: dict[str, Any] = {
            "id": q.id,
            "language": q.language,
            "answerable": q.answerable,
            "category": q.category,
            "question": q.question,
            "expected_answer": q.expected_answer,
        }
        request = ChatRequest(
            question=q.question,
            collection_ids=[corpus.collection_id],
            model_provider=gen.provider,
            model_name=gen.model,
            temperature=gen.temperature,
            max_tokens=gen.max_tokens,
            top_k=gen.top_k,
        )
        started = time.perf_counter()
        try:
            response = await chat_service.answer_question(db, embedder, registry, request)
        except (LLMError, chat_service.ChatError) as exc:
            await db.rollback()
            record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            continue
        total_ms = (time.perf_counter() - started) * 1000

        message = await db.get(Message, response.message_id)
        metadata: dict[str, Any] = dict(message.generation_metadata or {}) if message else {}
        latency = metadata.get("latency_ms") or {}

        sources = []
        for s in response.sources:
            ref = ChunkRef(corpus.documents.get(s.document_id), s.pdf_page_numbers, s.excerpt)
            matched = matched_items(ref, q.relevant, threshold)
            sources.append(
                {
                    "number": s.citation_number,
                    "cited": s.cited,
                    "document": ref.dataset_document,
                    "pdf_pages": [p + 1 for p in s.pdf_page_numbers],
                    "score": round(s.relevance_score, 4),
                    "relevant": bool(matched),
                }
            )
        cited = [s for s in sources if s["cited"]]
        cited_relevant = [s for s in cited if s["relevant"]]
        answered = not response.insufficient_evidence

        record.update(
            {
                "answer": response.answer,
                "abstained": response.insufficient_evidence,
                "abstain_reason": response.insufficient_evidence_reason,
                "model": response.model.model_dump() if response.model else None,
                "sources": sources,
                "cited_count": len(cited),
                "cited_relevant_count": len(cited_relevant),
                "citation_precision": len(cited_relevant) / len(cited) if cited else None,
                "citation_hit": bool(cited_relevant) if q.answerable and answered else None,
                "uncited_answer": answered and not cited,
                "context_hit": any(s["relevant"] for s in sources) if q.answerable else None,
                "invalid_citation_numbers": metadata.get("invalid_citation_numbers", []),
                "latency_ms": {
                    "retrieval": latency.get("retrieval"),
                    "generation": latency.get("generation"),
                    "total": round(total_ms, 1),
                },
                "usage": response.usage.model_dump() if response.usage else None,
            }
        )

        if judge_provider and judge_model and q.answerable and answered and q.expected_answer:
            try:
                scores = await judge_answer(
                    judge_provider,
                    judge_model,
                    q.question,
                    q.expected_answer,
                    [s.excerpt for s in response.sources],
                    response.answer,
                )
                record["judge"] = {
                    "correctness": scores.correctness,
                    "faithfulness": scores.faithfulness,
                    "relevance": scores.relevance,
                    "explanation": scores.explanation,
                }
            except (LLMError, ValueError) as exc:
                record["judge"] = {"error": f"{type(exc).__name__}: {exc}"}
        records.append(record)
    return records
