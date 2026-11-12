"""Document ingestion service.

Handles file storage, deduplication, PDF extraction, and chunking.
All heavy work (PDF extraction, chunking, DB writes) is done synchronously
within the request for Phase 2. Background task processing is Phase 4.
"""

import hashlib
import uuid
from pathlib import Path

import aiofiles
from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.models import Document, DocumentChunk, DocumentPage
from app.documents.chunking import chunk_page_texts
from app.documents.processing import extract_pdf


async def _compute_hash(file: UploadFile) -> tuple[bytes, str]:
    """Read the entire upload and return (raw_bytes, sha256_hex)."""
    data = await file.read()
    file_hash = hashlib.sha256(data).hexdigest()
    return data, file_hash


async def get_document_by_hash(db: AsyncSession, file_hash: str) -> Document | None:
    result = await db.execute(select(Document).where(Document.file_hash == file_hash))
    return result.scalar_one_or_none()


async def get_document(db: AsyncSession, document_id: uuid.UUID) -> Document | None:
    result = await db.execute(select(Document).where(Document.id == document_id))
    return result.scalar_one_or_none()


async def ingest_document(db: AsyncSession, file: UploadFile) -> tuple[Document, bool]:
    """Store, extract, and chunk a PDF upload.

    Returns:
        (document, created) — created=False means a duplicate was detected
        and the existing document record is returned unchanged.

    Raises:
        ValueError: If upload exceeds the size limit.
        Exception: Propagated from PyMuPDF on corrupt PDFs (status set to 'failed').
    """
    raw_bytes, file_hash = await _compute_hash(file)

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise ValueError(
            f"Upload exceeds maximum allowed size of {settings.max_upload_size_mb} MB."
        )

    # Deduplication: return existing document if same file was already uploaded.
    existing = await get_document_by_hash(db, file_hash)
    if existing is not None:
        return existing, False

    document_id = uuid.uuid4()
    storage = Path(settings.storage_path)
    storage.mkdir(parents=True, exist_ok=True)
    dest_path = storage / f"{document_id}.pdf"

    async with aiofiles.open(dest_path, "wb") as f:
        await f.write(raw_bytes)

    filename = file.filename or "unknown.pdf"
    document = Document(
        id=document_id,
        filename=filename,
        title=filename,  # overwritten after extraction
        file_path=str(dest_path),
        file_hash=file_hash,
        processing_status="processing",
    )
    db.add(document)
    await db.flush()  # get the row into the DB within the transaction

    try:
        meta = extract_pdf(dest_path)
    except Exception:
        document.processing_status = "failed"
        await db.commit()
        raise

    # Persist page text
    for page in meta.pages:
        db.add(
            DocumentPage(
                document_id=document_id,
                pdf_page_number=page.pdf_page_number,
                extracted_text=page.text,
            )
        )

    # Chunk across all pages
    page_tuples = [(p.pdf_page_number, p.text) for p in meta.pages]
    chunks = chunk_page_texts(page_tuples, settings.chunk_size, settings.chunk_overlap)
    for chunk in chunks:
        db.add(
            DocumentChunk(
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                page_numbers=chunk.page_numbers,
                chunk_metadata={
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                },
            )
        )

    # Update document with extracted metadata
    document.title = meta.title
    document.authors = meta.authors
    document.publication_year = meta.publication_year
    document.page_count = meta.page_count
    document.processing_status = "completed"
    await db.commit()
    await db.refresh(document)
    return document, True
