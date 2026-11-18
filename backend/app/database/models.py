"""
SQLAlchemy ORM models matching the schema defined in SPEC.md Section 6.

Design notes:
- EmbeddingModel records the name, version, and actual output dimensions of each
  embedding model. Dimensions are determined programmatically at embedding time,
  never hardcoded. This prevents silently mixing embeddings from incompatible
  vector spaces.
- DocumentChunk.embedding is a dimensionless pgvector column (migration 0003).
  The dimension is a property of the embedding model, not of the schema, so
  switching models only requires re-indexing — no migration.
- Each chunk references embedding_model_id. Retrieval always filters on the
  active model's id, so vectors from incompatible spaces are never compared.
"""

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    # updated_at is managed by a database trigger (see migration 0001)
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (UniqueConstraint("name", name="uq_collections_name"),)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    authors: Mapped[str | None] = mapped_column(Text, nullable=True)
    publication_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    # SHA-256 hash of the PDF file; used to detect duplicate uploads
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # pending | processing | completed | failed
    processing_status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


class CollectionDocument(Base):
    """Many-to-many between Collection and Document.

    A document may belong to multiple collections without storing duplicate
    PDF files or regenerating embeddings.
    Deleting a collection does NOT cascade to documents.
    """

    __tablename__ = "collection_documents"

    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        primary_key=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    )
    added_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


class DocumentPage(Base):
    __tablename__ = "document_pages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Zero-based PDF page index (authoritative reference)
    pdf_page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    # Printed page number as it appears in the document (may differ from pdf_page_number)
    printed_page_number: Mapped[str | None] = mapped_column(String(50), nullable=True)
    extracted_text: Mapped[str] = mapped_column(Text, nullable=False, default="")


class EmbeddingModel(Base):
    """Registry of embedding models and their resolved output dimensions.

    When an embedding model is first used, its actual vector dimension is read
    from the loaded model and recorded here together with its name and version
    (Hugging Face commit hash when available).

    If the configured model changes (embedding_model_name in settings), a new
    row is inserted and marked active. Chunks embedded with the previous model
    are excluded from retrieval until they are re-indexed.
    """

    __tablename__ = "embedding_models"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Model version string (e.g. commit hash or semantic version); nullable if unknown
    version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # Output vector dimension — determined programmatically, never hardcoded
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    # The model currently used for indexing and retrieval; at most one is active
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_embedding_model_name_version"),
    )


class DocumentChunk(Base):
    """Text chunk extracted from a document page range.

    `embedding` has no fixed dimension. Its vector space is identified by
    `embedding_model_id`; both are set together and must only be compared with
    vectors from the same model.

    page_numbers stores a JSON list of PDF page numbers covered by this chunk
    (a chunk may span a page boundary).
    """

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Null until the chunk has been embedded
    embedding: Mapped[Any | None] = mapped_column(Vector(), nullable=True)
    embedding_model_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("embedding_models.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # List of PDF page numbers this chunk covers, e.g. [5, 6]
    page_numbers: Mapped[Any | None] = mapped_column(JSONB, nullable=True)
    # Arbitrary chunk-level metadata (chunking config, overlap, etc.)
    chunk_metadata: Mapped[Any | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


class ConversationCollection(Base):
    """Records which collections are in scope for a conversation."""

    __tablename__ = "conversation_collections"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        primary_key=True,
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # user | assistant | system
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model_provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


class MessageSource(Base):
    """Links a message to the document chunks that supported it.

    The backend assembles these records from retrieval metadata.
    The LLM is never responsible for generating or validating source references.
    """

    __tablename__ = "message_sources"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=False,
    )
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
