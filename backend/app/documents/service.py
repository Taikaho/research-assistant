"""Document ingestion service.

Handles file storage, deduplication, PDF extraction, chunking and embedding.
All work is done within the request; model inference runs in a worker thread.
"""

import hashlib
import logging
import uuid
from collections.abc import Sequence
from pathlib import Path

import aiofiles
from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.collections.service import link_document
from app.core.config import settings
from app.database.models import Document, DocumentChunk, DocumentPage
from app.documents.chunking import chunk_page_texts
from app.documents.processing import extract_pdf
from app.embeddings.base import EmbeddingProvider
from app.embeddings.service import embed_chunks, register_model

logger = logging.getLogger(__name__)


class EmbeddingUnavailableError(RuntimeError):
    """The embedding model could not be loaded or failed to embed the document."""


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


async def list_documents(db: AsyncSession) -> Sequence[Document]:
    result = await db.execute(select(Document).order_by(Document.created_at))
    return result.scalars().all()


async def delete_document(db: AsyncSession, document: Document) -> None:
    """Permanently delete a document, its pages, chunks, collection links and PDF file.

    Pages, chunks and collection links are removed by ON DELETE CASCADE.
    The file is removed after the commit so a failed transaction never leaves
    a database row pointing at a missing file.
    """
    file_path = Path(document.file_path)
    await db.delete(document)
    await db.commit()
    try:
        file_path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not delete stored PDF %s", file_path, exc_info=True)


async def ingest_document(
    db: AsyncSession,
    file: UploadFile,
    provider: EmbeddingProvider,
    collection_ids: Sequence[uuid.UUID] = (),
) -> tuple[Document, bool]:
    """Store, extract, chunk and embed a PDF upload, and add it to the given collections.

    The caller must verify that collection_ids exist. For a duplicate upload the
    existing document is added to the collections instead.

    Returns:
        (document, created) — created=False means a duplicate was detected
        and the existing document record is returned unchanged.

    Raises:
        ValueError: If upload exceeds the size limit.
        EmbeddingUnavailableError: If the embedding model cannot be loaded (nothing is
            stored) or embedding fails (status set to 'failed').
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
        if collection_ids:
            await link_document(db, collection_ids, existing.id)
            await db.commit()
        return existing, False

    # Resolve the embedding model before storing anything, so an unavailable
    # model does not leave a half-processed document behind.
    try:
        embedding_model = await register_model(db, provider)
    except Exception as exc:
        await db.rollback()
        raise EmbeddingUnavailableError(f"Embedding model unavailable: {exc}") from exc

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
    chunk_rows = [
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
        for chunk in chunks
    ]
    db.add_all(chunk_rows)

    try:
        await embed_chunks(provider, embedding_model, chunk_rows)
    except Exception as exc:
        logger.exception("Embedding failed for document %s", document_id)
        document.processing_status = "failed"
        await db.commit()
        raise EmbeddingUnavailableError(f"Embedding failed: {exc}") from exc

    # Update document with extracted metadata
    document.title = meta.title
    document.authors = meta.authors
    document.publication_year = meta.publication_year
    document.page_count = meta.page_count
    document.processing_status = "completed"
    await link_document(db, collection_ids, document_id)
    await db.commit()
    await db.refresh(document)
    return document, True
