from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.collections.service import find_missing_collection_ids
from app.database.connection import get_db
from app.embeddings.base import EmbeddingProvider
from app.embeddings.service import get_embedding_provider
from app.retrieval import service
from app.retrieval.schemas import (
    SearchModelInfo,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

router = APIRouter(prefix="/search", tags=["retrieval"])


@router.post("", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> SearchResponse:
    """Semantic search for passages within the selected collections."""
    missing = await find_missing_collection_ids(db, request.collection_ids)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collections not found: {', '.join(str(m) for m in missing)}",
        )

    try:
        result = await service.search(
            db,
            provider,
            request.query,
            request.collection_ids,
            top_k=request.top_k,
            min_score=request.min_score,
        )
    except Exception as exc:
        # Typically the embedding model failing to load or run.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Search failed: {exc}",
        ) from exc

    return SearchResponse(
        query=request.query,
        embedding_model=SearchModelInfo(
            name=result.model.name,
            version=result.model.version,
            dimensions=result.model.dimensions,
        ),
        results=[
            SearchResult(
                chunk_id=r.chunk.id,
                document_id=r.document.id,
                document_title=r.document.title,
                filename=r.document.filename,
                authors=r.document.authors,
                publication_year=r.document.publication_year,
                chunk_index=r.chunk.chunk_index,
                pdf_page_numbers=list(r.chunk.page_numbers or []),
                text=r.chunk.text,
                score=r.score,
            )
            for r in result.chunks
        ],
        unindexed_document_count=result.unindexed_document_count,
    )
