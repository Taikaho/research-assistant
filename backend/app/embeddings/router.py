from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.connection import get_db
from app.embeddings import service
from app.embeddings.base import EmbeddingProvider
from app.embeddings.schemas import (
    EmbeddingModelResponse,
    IndexStatusResponse,
    ReindexResponse,
)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.get("/status", response_model=IndexStatusResponse)
async def get_index_status(db: AsyncSession = Depends(get_db)) -> IndexStatusResponse:
    """Report the active embedding model and how many chunks it has indexed.

    Does not load the model. If the configured model differs from the active one,
    run POST /embeddings/reindex.
    """
    status = await service.index_status(db)
    model = status.model
    reindex_required = (
        model is None
        or model.name != settings.embedding_model_name
        or status.indexed_chunks < status.total_chunks
    )
    return IndexStatusResponse(
        configured_model_name=settings.embedding_model_name,
        active_model=EmbeddingModelResponse.model_validate(model) if model else None,
        total_chunks=status.total_chunks,
        indexed_chunks=status.indexed_chunks,
        reindex_required=reindex_required,
    )


@router.post("/reindex", response_model=ReindexResponse)
async def reindex(
    db: AsyncSession = Depends(get_db),
    provider: EmbeddingProvider = Depends(service.get_embedding_provider),
) -> ReindexResponse:
    """Embed every chunk that lacks an embedding from the configured model.

    Idempotent: chunks already embedded with the configured model are skipped.
    Runs synchronously; large libraries may take several minutes.
    """
    result = await service.reindex(db, provider)
    return ReindexResponse(
        model=EmbeddingModelResponse.model_validate(result.model),
        documents_reindexed=result.documents,
        chunks_reindexed=result.chunks,
    )
