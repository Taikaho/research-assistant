import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat import service
from app.chat.schemas import ChatRequest, ChatResponse, ConversationDetail, ConversationSummary
from app.database.connection import get_db
from app.embeddings.base import EmbeddingProvider
from app.embeddings.service import get_embedding_provider
from app.llm.base import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.llm.registry import LLMRegistry, UnknownProviderError, get_llm_registry

router = APIRouter(tags=["chat"])


def llm_error_to_http(exc: LLMError) -> HTTPException:
    """Translate provider-neutral LLM errors into HTTP responses."""
    if isinstance(exc, LLMModelNotFoundError):
        return HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc))
    if isinstance(exc, LLMRateLimitError):
        headers = (
            {"Retry-After": str(int(exc.retry_after))} if exc.retry_after is not None else None
        )
        return HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc), headers=headers)
    if isinstance(exc, LLMTimeoutError):
        return HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc))
    if isinstance(exc, LLMConnectionError):
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    if isinstance(exc, LLMAuthenticationError):
        return HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    return HTTPException(status.HTTP_502_BAD_GATEWAY, detail=f"LLM provider error: {exc}")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    embedder: EmbeddingProvider = Depends(get_embedding_provider),
    registry: LLMRegistry = Depends(get_llm_registry),
) -> ChatResponse:
    """Answer a question from passages in the selected collections, with sources."""
    try:
        return await service.answer_question(db, embedder, registry, request)
    except service.ConversationNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except service.CollectionsNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except service.MissingScopeError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except UnknownProviderError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except service.RetrievalFailedError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except LLMError as exc:
        raise llm_error_to_http(exc) from exc


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(db: AsyncSession = Depends(get_db)) -> list[ConversationSummary]:
    return await service.list_conversations(db)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> ConversationDetail:
    try:
        return await service.get_conversation(db, conversation_id)
    except service.ConversationNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> Response:
    try:
        await service.delete_conversation(db, conversation_id)
    except service.ConversationNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
