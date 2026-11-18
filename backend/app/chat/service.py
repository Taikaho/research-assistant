"""Retrieval-augmented question answering.

Pipeline: resolve scope -> retrieve passages -> relevance gate -> prompt ->
LLM -> parse citations -> assemble sources from retrieval metadata -> persist.

The pipeline talks to LLMs only through app.llm.base.LLMProvider and never
branches on local vs cloud; it only reports which one was used.
"""

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import case, delete, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.citations import parse_answer, strip_citations
from app.chat.prompts import PROMPT_VERSION, PromptPassage, build_messages
from app.chat.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
    InsufficientReason,
    MessageRead,
    ModelInfo,
    SourceReference,
    TokenUsage,
)
from app.collections.service import find_missing_collection_ids
from app.core.config import settings
from app.database.models import (
    Conversation,
    ConversationCollection,
    Document,
    DocumentChunk,
    DocumentPage,
    Message,
    MessageSource,
)
from app.embeddings.base import EmbeddingProvider
from app.llm.base import ChatMessage, GenerationParams, LLMResponseError
from app.llm.registry import LLMRegistry
from app.retrieval import service as retrieval

NO_EVIDENCE_ANSWER = (
    "I could not find passages in the selected collections that are relevant enough "
    "to answer this question."
)
MODEL_INSUFFICIENT_ANSWER = (
    "The retrieved passages do not contain enough information to answer this question."
)


# ---------------------------------------------------------------------------
# Errors (mapped to HTTP responses by the router)
# ---------------------------------------------------------------------------


class ChatError(Exception):
    pass


class ConversationNotFoundError(ChatError):
    pass


class CollectionsNotFoundError(ChatError):
    def __init__(self, missing: Sequence[uuid.UUID]) -> None:
        super().__init__(f"Collections not found: {', '.join(str(m) for m in missing)}")
        self.missing = list(missing)


class MissingScopeError(ChatError):
    pass


class RetrievalFailedError(ChatError):
    pass


# ---------------------------------------------------------------------------
# Source assembly
# ---------------------------------------------------------------------------


@dataclass
class _Passage:
    number: int
    chunk: DocumentChunk
    document: Document
    score: float
    cited: bool = False


async def _printed_pages(
    db: AsyncSession, chunks: Sequence[DocumentChunk]
) -> dict[tuple[uuid.UUID, int], str | None]:
    keys = {(c.document_id, int(p)) for c in chunks for p in (c.page_numbers or [])}
    if not keys:
        return {}
    rows = await db.execute(
        select(
            DocumentPage.document_id, DocumentPage.pdf_page_number, DocumentPage.printed_page_number
        ).where(tuple_(DocumentPage.document_id, DocumentPage.pdf_page_number).in_(list(keys)))
    )
    return {(doc_id, page): printed for doc_id, page, printed in rows.all()}


def _source(
    passage: _Passage, printed: dict[tuple[uuid.UUID, int], str | None]
) -> SourceReference:
    pages = [int(p) for p in (passage.chunk.page_numbers or [])]
    return SourceReference(
        citation_number=passage.number,
        cited=passage.cited,
        document_id=passage.document.id,
        document_title=passage.document.title,
        filename=passage.document.filename,
        chunk_id=passage.chunk.id,
        pdf_page_numbers=pages,
        printed_page_numbers=[printed.get((passage.document.id, p)) for p in pages],
        relevance_score=passage.score,
        excerpt=passage.chunk.text,
    )


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

_ROLE_ORDER = case((Message.role == "user", 0), else_=1)


async def _conversation_scope(db: AsyncSession, conversation_id: uuid.UUID) -> list[uuid.UUID]:
    rows = await db.execute(
        select(ConversationCollection.collection_id).where(
            ConversationCollection.conversation_id == conversation_id
        )
    )
    return list(rows.scalars().all())


async def _set_conversation_scope(
    db: AsyncSession, conversation_id: uuid.UUID, collection_ids: Sequence[uuid.UUID]
) -> None:
    await db.execute(
        delete(ConversationCollection).where(
            ConversationCollection.conversation_id == conversation_id
        )
    )
    db.add_all(
        ConversationCollection(conversation_id=conversation_id, collection_id=cid)
        for cid in collection_ids
    )


async def _history(db: AsyncSession, conversation_id: uuid.UUID) -> list[ChatMessage]:
    """Recent turns for follow-up questions, oldest first.

    Citation markers are stripped from earlier answers: their numbers referred to
    that turn's passages and would be confused with the current numbering.
    """
    limit = settings.chat_history_messages
    if limit <= 0:
        return []
    rows = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id, Message.role.in_(["user", "assistant"]))
        .order_by(Message.created_at.desc(), _ROLE_ORDER.desc())
        .limit(limit)
    )
    messages = list(reversed(rows.scalars().all()))
    history: list[ChatMessage] = []
    for m in messages:
        if m.role == "assistant":
            history.append(ChatMessage(role="assistant", content=strip_citations(m.content)))
        else:
            history.append(ChatMessage(role="user", content=m.content))
    return history


def _previous_question(history: Sequence[ChatMessage]) -> str | None:
    for message in reversed(history):
        if message.role == "user":
            return message.content
    return None


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------


async def answer_question(
    db: AsyncSession,
    embedder: EmbeddingProvider,
    registry: LLMRegistry,
    request: ChatRequest,
) -> ChatResponse:
    # 1. Scope
    conversation: Conversation | None = None
    if request.conversation_id is not None:
        conversation = await db.get(Conversation, request.conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(f"Conversation {request.conversation_id} not found")

    if request.collection_ids:
        scope = list(dict.fromkeys(request.collection_ids))
    elif conversation is not None:
        scope = await _conversation_scope(db, conversation.id)
    else:
        scope = []
    if not scope:
        raise MissingScopeError("Select at least one collection (collection_ids)")
    missing = await find_missing_collection_ids(db, scope)
    if missing:
        raise CollectionsNotFoundError(missing)

    # 2. Model (fail fast on an unknown provider / model before doing any work)
    provider = registry.get(request.model_provider)
    model_name = await provider.resolve_model(request.model_name)
    params = GenerationParams(
        temperature=(
            request.temperature if request.temperature is not None else settings.llm_temperature
        ),
        max_tokens=request.max_tokens or settings.llm_max_tokens,
    )

    history = await _history(db, conversation.id) if conversation is not None else []

    # 3. Retrieval. A follow-up such as "which one was more accurate?" often lacks the
    # topic words, so it is also searched together with the previous question; each
    # passage keeps its best score, which is never worse than the question alone.
    top_k = request.top_k or settings.chat_top_k or settings.retrieval_top_k
    queries = [request.question]
    previous = _previous_question(history)
    if previous:
        queries.append(f"{previous}\n{request.question}")
    started = time.perf_counter()
    try:
        results = [
            await retrieval.search(db, embedder, q, scope, top_k=top_k) for q in queries
        ]
    except Exception as exc:
        raise RetrievalFailedError(f"Retrieval failed: {exc}") from exc
    retrieval_ms = round((time.perf_counter() - started) * 1000)
    result = results[0]
    chunks = retrieval.merge_results([r.chunks for r in results], top_k)

    # 4. Relevance gate: don't ask the model to answer from irrelevant text.
    relevant = [c for c in chunks if c.score >= settings.chat_min_relevance]
    passages = [
        _Passage(number=i, chunk=c.chunk, document=c.document, score=c.score)
        for i, c in enumerate(relevant, start=1)
    ]

    metadata: dict[str, object] = {
        "prompt_version": PROMPT_VERSION,
        "temperature": params.temperature,
        "max_tokens": params.max_tokens,
        "top_k": top_k,
        "min_relevance": settings.chat_min_relevance,
        "embedding_model": {"name": result.model.name, "version": result.model.version},
        "retrieval_queries": queries,
        "retrieved_count": len(chunks),
        "passage_count": len(passages),
        "latency_ms": {"retrieval": retrieval_ms},
    }

    model_info: ModelInfo | None = None
    usage: TokenUsage | None = None
    reason: InsufficientReason | None = None

    if not passages:
        answer = NO_EVIDENCE_ANSWER
        insufficient = True
        reason = "no_relevant_passages"
    else:
        # 5. Generation
        messages = build_messages(
            request.question,
            [PromptPassage(p.number, p.document.title, p.chunk.text) for p in passages],
            history,
        )
        started = time.perf_counter()
        response = await provider.generate(model_name, messages, params)
        generation_ms = round((time.perf_counter() - started) * 1000)

        # 6. Citations
        parsed = parse_answer(response.text, len(passages))
        insufficient = parsed.insufficient_evidence
        if insufficient:
            reason = "model_reported"
            answer = parsed.text or MODEL_INSUFFICIENT_ANSWER
        elif not parsed.text:
            raise LLMResponseError(f"{provider.name} returned an empty answer")
        else:
            answer = parsed.text
        cited = set(parsed.cited_numbers)
        for p in passages:
            p.cited = p.number in cited

        model_info = ModelInfo(provider=provider.name, name=model_name, is_local=provider.is_local)
        usage = TokenUsage(
            prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens
        )
        metadata.update(
            {
                "provider_is_local": provider.is_local,
                "response_model": response.model,
                "finish_reason": response.finish_reason,
                "usage": usage.model_dump(),
                "cited_numbers": parsed.cited_numbers,
                "invalid_citation_numbers": parsed.invalid_numbers,
                "history_messages": len(history),
            }
        )
        metadata["latency_ms"] = {"retrieval": retrieval_ms, "generation": generation_ms}

    # 7. Persist
    if conversation is None:
        conversation = Conversation(title=request.question.strip()[:200])
        db.add(conversation)
        await db.flush()
        await _set_conversation_scope(db, conversation.id, scope)
    elif request.collection_ids:
        await _set_conversation_scope(db, conversation.id, scope)

    db.add(Message(conversation_id=conversation.id, role="user", content=request.question))
    assistant = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        model_provider=model_info.provider if model_info else None,
        model_name=model_info.name if model_info else None,
        insufficient_evidence=insufficient,
        generation_metadata=metadata,
    )
    db.add(assistant)
    await db.flush()
    db.add_all(
        MessageSource(
            message_id=assistant.id,
            chunk_id=p.chunk.id,
            relevance_score=p.score,
            citation_number=p.number,
            cited=p.cited,
        )
        for p in passages
    )
    await db.commit()

    printed = await _printed_pages(db, [p.chunk for p in passages])
    return ChatResponse(
        conversation_id=conversation.id,
        message_id=assistant.id,
        answer=answer,
        insufficient_evidence=insufficient,
        insufficient_evidence_reason=reason,
        sources=[_source(p, printed) for p in passages],
        model=model_info,
        cloud_processing=model_info is not None and not model_info.is_local,
        usage=usage,
        collection_ids=scope,
        unindexed_document_count=result.unindexed_document_count,
    )


# ---------------------------------------------------------------------------
# Conversation queries
# ---------------------------------------------------------------------------


async def list_conversations(db: AsyncSession) -> list[ConversationSummary]:
    counts = (
        select(Message.conversation_id, func.count().label("n"))
        .group_by(Message.conversation_id)
        .subquery()
    )
    rows = (
        await db.execute(
            select(Conversation, func.coalesce(counts.c.n, 0))
            .outerjoin(counts, counts.c.conversation_id == Conversation.id)
            .order_by(Conversation.created_at.desc())
            .execution_options(populate_existing=True)
        )
    ).all()
    scopes = await _scopes(db, [c.id for c, _ in rows])
    return [
        ConversationSummary(
            id=c.id,
            title=c.title,
            created_at=c.created_at,
            collection_ids=scopes.get(c.id, []),
            message_count=int(n),
        )
        for c, n in rows
    ]


async def _scopes(
    db: AsyncSession, conversation_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, list[uuid.UUID]]:
    if not conversation_ids:
        return {}
    rows = await db.execute(
        select(ConversationCollection.conversation_id, ConversationCollection.collection_id).where(
            ConversationCollection.conversation_id.in_(conversation_ids)
        )
    )
    scopes: dict[uuid.UUID, list[uuid.UUID]] = {}
    for conv_id, col_id in rows.all():
        scopes.setdefault(conv_id, []).append(col_id)
    return scopes


async def get_conversation(db: AsyncSession, conversation_id: uuid.UUID) -> ConversationDetail:
    conversation = await db.get(Conversation, conversation_id, populate_existing=True)
    if conversation is None:
        raise ConversationNotFoundError(f"Conversation {conversation_id} not found")

    messages = list(
        (
            await db.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at, _ROLE_ORDER)
                .execution_options(populate_existing=True)
            )
        )
        .scalars()
        .all()
    )
    source_rows = (
        await db.execute(
            select(MessageSource, DocumentChunk, Document)
            .join(DocumentChunk, DocumentChunk.id == MessageSource.chunk_id)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(MessageSource.message_id.in_([m.id for m in messages]))
            .order_by(MessageSource.citation_number)
        )
    ).all()
    printed = await _printed_pages(db, [chunk for _, chunk, _ in source_rows])
    sources: dict[uuid.UUID, list[SourceReference]] = {}
    for ms, chunk, document in source_rows:
        passage = _Passage(
            number=ms.citation_number or 0,
            chunk=chunk,
            document=document,
            score=ms.relevance_score,
            cited=ms.cited,
        )
        sources.setdefault(ms.message_id, []).append(_source(passage, printed))

    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        collection_ids=await _conversation_scope(db, conversation.id),
        messages=[
            MessageRead(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
                model_provider=m.model_provider,
                model_name=m.model_name,
                insufficient_evidence=m.insufficient_evidence,
                generation_metadata=m.generation_metadata,
                sources=sources.get(m.id, []),
            )
            for m in messages
        ],
    )


async def delete_conversation(db: AsyncSession, conversation_id: uuid.UUID) -> None:
    conversation = await db.get(Conversation, conversation_id)
    if conversation is None:
        raise ConversationNotFoundError(f"Conversation {conversation_id} not found")
    await db.delete(conversation)
    await db.commit()
