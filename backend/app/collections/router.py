import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.collections import service
from app.collections.schemas import CollectionCreate, CollectionResponse, CollectionUpdate
from app.database.connection import get_db
from app.database.models import Collection
from app.documents import service as documents_service
from app.documents.router import document_responses
from app.documents.schemas import DocumentResponse

router = APIRouter(prefix="/collections", tags=["collections"])


async def _with_count(db: AsyncSession, collection: Collection) -> CollectionResponse:
    counts = await service.document_counts(db, [collection.id])
    response = CollectionResponse.model_validate(collection)
    response.document_count = counts.get(collection.id, 0)
    return response


@router.post("", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    data: CollectionCreate, db: AsyncSession = Depends(get_db)
) -> CollectionResponse:
    try:
        collection = await service.create_collection(db, data)
    except IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A collection named {data.name!r} already exists.",
        )
    return CollectionResponse.model_validate(collection)


@router.get("", response_model=list[CollectionResponse])
async def list_collections(db: AsyncSession = Depends(get_db)) -> list[CollectionResponse]:
    collections = await service.list_collections(db)
    counts = await service.document_counts(db, [c.id for c in collections])
    responses = []
    for c in collections:
        response = CollectionResponse.model_validate(c)
        response.document_count = counts.get(c.id, 0)
        responses.append(response)
    return responses


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(
    collection_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> CollectionResponse:
    collection = await service.get_collection(db, collection_id)
    if collection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found.")
    return await _with_count(db, collection)


@router.patch("/{collection_id}", response_model=CollectionResponse)
async def update_collection(
    collection_id: uuid.UUID,
    data: CollectionUpdate,
    db: AsyncSession = Depends(get_db),
) -> CollectionResponse:
    collection = await service.get_collection(db, collection_id)
    if collection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found.")
    try:
        collection = await service.update_collection(db, collection, data)
    except IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A collection named {data.name!r} already exists.",
        )
    return await _with_count(db, collection)


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> None:
    collection = await service.get_collection(db, collection_id)
    if collection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found.")
    await service.delete_collection(db, collection)


# ---------------------------------------------------------------------------
# Collection membership
# ---------------------------------------------------------------------------


@router.get("/{collection_id}/documents", response_model=list[DocumentResponse])
async def list_collection_documents(
    collection_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> list[DocumentResponse]:
    if await service.get_collection(db, collection_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found.")
    documents = await service.list_documents(db, collection_id)
    return await document_responses(db, documents)


@router.put(
    "/{collection_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def add_document_to_collection(
    collection_id: uuid.UUID, document_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> None:
    """Add an existing document to a collection. Idempotent."""
    if await service.get_collection(db, collection_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found.")
    if await documents_service.get_document(db, document_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    await service.add_document(db, collection_id, document_id)


@router.delete(
    "/{collection_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def remove_document_from_collection(
    collection_id: uuid.UUID, document_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> None:
    """Remove a document from a collection. The document itself is kept."""
    if await service.get_collection(db, collection_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found.")
    if not await service.remove_document(db, collection_id, document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document is not in this collection."
        )
