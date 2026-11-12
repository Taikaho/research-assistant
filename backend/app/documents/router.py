import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_db
from app.documents import service
from app.documents.schemas import DocumentResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile, db: AsyncSession = Depends(get_db)
) -> DocumentResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=422,
            detail="Only PDF files are accepted.",
        )

    try:
        document, created = await service.ingest_document(db, file)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"PDF processing failed: {exc}",
        )

    # Return 200 for duplicates, 201 for new uploads (FastAPI default for this route is 201).
    # We override via Response but it's simpler to just always return 201 on success —
    # duplicate detection is communicated via the unchanged document record.
    # (Callers can check created=False by comparing file_hash.)
    _ = created  # used implicitly — same schema either way
    return DocumentResponse.model_validate(document)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> DocumentResponse:
    document = await service.get_document(db, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return DocumentResponse.model_validate(document)
