"""Initial schema

Creates all tables defined in SPEC.md Section 6, plus the embedding_models
registry table.

NOTE: The document_chunks table does NOT include a vector column.
The pgvector column type requires a fixed dimension known at migration time.
That dimension is determined programmatically by loading the configured
embedding model (see app/core/config.py: embedding_model_name).
A dedicated Phase 3 migration will add the vector column once the model
has been selected and its output dimension confirmed.

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-11-11
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Extensions
    # ------------------------------------------------------------------
    # pgvector: required for the vector column added in Phase 3.
    # Enabling it now ensures the extension is available when needed.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ------------------------------------------------------------------
    # Trigger function for updated_at columns
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE OR REPLACE FUNCTION set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    # ------------------------------------------------------------------
    # collections
    # ------------------------------------------------------------------
    op.create_table(
        "collections",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.execute(
        """
        CREATE TRIGGER trg_collections_updated_at
        BEFORE UPDATE ON collections
        FOR EACH ROW EXECUTE FUNCTION set_updated_at()
        """
    )

    # ------------------------------------------------------------------
    # documents
    # ------------------------------------------------------------------
    op.create_table(
        "documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("authors", sa.Text, nullable=True),
        sa.Column("publication_year", sa.Integer, nullable=True),
        sa.Column("page_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("file_path", sa.String(1024), nullable=False),
        sa.Column("file_hash", sa.String(64), nullable=False),
        sa.Column(
            "processing_status",
            sa.String(50),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_documents_file_hash", "documents", ["file_hash"])
    op.create_index("ix_documents_processing_status", "documents", ["processing_status"])

    # ------------------------------------------------------------------
    # collection_documents  (many-to-many)
    # Composite PK prevents duplicate relationships.
    # Deleting a collection cascades here but does NOT delete documents.
    # ------------------------------------------------------------------
    op.create_table(
        "collection_documents",
        sa.Column(
            "collection_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="RESTRICT"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "added_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # ------------------------------------------------------------------
    # document_pages
    # ------------------------------------------------------------------
    op.create_table(
        "document_pages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("pdf_page_number", sa.Integer, nullable=False),
        sa.Column("printed_page_number", sa.String(50), nullable=True),
        sa.Column("extracted_text", sa.Text, nullable=False, server_default=""),
    )
    op.create_index(
        "ix_document_pages_document_id", "document_pages", ["document_id"]
    )

    # ------------------------------------------------------------------
    # embedding_models
    # Tracks the name, version, and actual output dimensions of every
    # embedding model used. Dimensions are populated programmatically
    # when the model is first loaded — never hardcoded.
    # ------------------------------------------------------------------
    op.create_table(
        "embedding_models",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("version", sa.String(100), nullable=True),
        sa.Column("dimensions", sa.Integer, nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("name", "version", name="uq_embedding_model_name_version"),
    )

    # ------------------------------------------------------------------
    # document_chunks
    # NOTE: No vector column here. See module docstring.
    # ------------------------------------------------------------------
    op.create_table(
        "document_chunks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column(
            "embedding_model_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("embedding_models.id", ondelete="SET NULL"),
            nullable=True,
        ),
        # List of PDF page numbers this chunk covers, e.g. [5, 6]
        sa.Column("page_numbers", postgresql.JSONB, nullable=True),
        # Chunking config and other arbitrary metadata
        sa.Column("chunk_metadata", postgresql.JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_document_chunks_document_id", "document_chunks", ["document_id"]
    )
    op.create_index(
        "ix_document_chunks_embedding_model_id",
        "document_chunks",
        ["embedding_model_id"],
    )

    # ------------------------------------------------------------------
    # conversations
    # ------------------------------------------------------------------
    op.create_table(
        "conversations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("title", sa.String(512), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # ------------------------------------------------------------------
    # conversation_collections
    # Records which collections are active for a conversation.
    # ------------------------------------------------------------------
    op.create_table(
        "conversation_collections",
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "collection_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        ),
    )

    # ------------------------------------------------------------------
    # messages
    # ------------------------------------------------------------------
    op.create_table(
        "messages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(50), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("model_provider", sa.String(100), nullable=True),
        sa.Column("model_name", sa.String(255), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_messages_conversation_id", "messages", ["conversation_id"]
    )

    # ------------------------------------------------------------------
    # message_sources
    # Links each assistant message to the chunks that supported it.
    # The backend populates this; the LLM never generates source references.
    # ------------------------------------------------------------------
    op.create_table(
        "message_sources",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "message_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "chunk_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_chunks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("relevance_score", sa.Float, nullable=False),
    )
    op.create_index(
        "ix_message_sources_message_id", "message_sources", ["message_id"]
    )


def downgrade() -> None:
    op.drop_table("message_sources")
    op.drop_table("messages")
    op.drop_table("conversation_collections")
    op.drop_table("conversations")
    op.drop_table("document_chunks")
    op.drop_table("embedding_models")
    op.drop_table("document_pages")
    op.drop_table("collection_documents")
    op.drop_table("documents")
    op.drop_table("collections")
    op.execute("DROP FUNCTION IF EXISTS set_updated_at CASCADE")
    op.execute("DROP EXTENSION IF EXISTS vector")
