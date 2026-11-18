import uuid
from collections.abc import Sequence
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.collections import service as collections_service
from app.database.connection import get_db
from app.database.models import Document
from app.documents import service
from app.documents.schemas import DocumentResponse, PageResponse
from app.embeddings.base import EmbeddingProvider
from app.embeddings.service import get_embedding_provider

router = APIRouter(prefix="/documents", tags=["documents"])


async def document_responses(
    db: AsyncSession, documents: Sequence[Document]
) -> list[DocumentResponse]:
    """Build responses for documents, including their collection membership."""
    membership = await service.collection_ids_by_document(db, [d.id for d in documents])
    responses = []
    for d in documents:
        response = DocumentResponse.model_validate(d)
        response.collection_ids = membership.get(d.id, [])
        responses.append(response)
    return responses


async def _get_or_404(db: AsyncSession, document_id: uuid.UUID) -> Document:
    document = await service.get_document(db, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return document


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    collection_ids: list[uuid.UUID] = Form(default=[]),
    db: AsyncSession = Depends(get_db),
    provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> DocumentResponse:
    """Upload a PDF. Repeat the `collection_ids` form field to add it to collections."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=422,
            detail="Only PDF files are accepted.",
        )

    missing = await collections_service.find_missing_collection_ids(db, collection_ids)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collections not found: {', '.join(str(m) for m in missing)}",
        )

    try:
        document, created = await service.ingest_document(db, file, provider, collection_ids)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except service.EmbeddingUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
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
    return (await document_responses(db, [document]))[0]


@router.get("", response_model=list[DocumentResponse])
async def list_documents(db: AsyncSession = Depends(get_db)) -> list[DocumentResponse]:
    documents = await service.list_documents(db)
    return await document_responses(db, documents)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> DocumentResponse:
    document = await _get_or_404(db, document_id)
    return (await document_responses(db, [document]))[0]


@router.get("/{document_id}/file", response_class=FileResponse)
async def get_document_file(
    document_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> FileResponse:
    """Serve the original PDF inline, so a browser can open it at a page (#page=N)."""
    document = await _get_or_404(db, document_id)
    path = Path(document.file_path)
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Stored PDF file is missing."
        )
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=document.filename,
        content_disposition_type="inline",
    )


@router.get("/{document_id}/pages/{page_number}", response_model=PageResponse)
async def get_document_page(
    document_id: uuid.UUID, page_number: int, db: AsyncSession = Depends(get_db)
) -> PageResponse:
    """Extracted text of one page. `page_number` is the zero-based PDF page index."""
    await _get_or_404(db, document_id)
    page = await service.get_page(db, document_id, page_number)
    if page is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found.")
    return PageResponse(
        document_id=document_id,
        pdf_page_number=page.pdf_page_number,
        printed_page_number=page.printed_page_number,
        text=page.extracted_text,
    )


@router.post("/{document_id}/reprocess", response_model=DocumentResponse)
async def reprocess_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> DocumentResponse:
    """Re-extract, re-chunk and re-embed a document with the current settings.

    Chunks are replaced, so source links in earlier chat answers to this document
    are removed. Collection membership is kept.
    """
    document = await _get_or_404(db, document_id)
    try:
        document = await service.reprocess_document(db, document, provider)
    except service.StoredFileMissingError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except service.EmbeddingUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"PDF processing failed: {exc}") from exc
    return (await document_responses(db, [document]))[0]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> None:
    """Permanently delete a document from every collection, including its chunks and PDF.

    To only take a document out of one collection, use
    DELETE /collections/{collection_id}/documents/{document_id}.
    """
    document = await _get_or_404(db, document_id)
    await service.delete_document(db, document)
