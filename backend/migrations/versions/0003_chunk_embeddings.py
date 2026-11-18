"""Add the embedding vector column to document_chunks.

The column is a dimensionless pgvector `vector`. Each chunk's vector space is
identified by document_chunks.embedding_model_id, and retrieval filters on it,
so changing the embedding model requires re-indexing but no schema change.

No ANN index is created: exact search over a filtered set of chunks is fast at
the expected corpus size and, unlike HNSW with filters, never loses recall.
An HNSW index needs a fixed dimension and can be added later as a partial
expression index per embedding model if the corpus grows.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-16
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("document_chunks", sa.Column("embedding", Vector(), nullable=True))


def downgrade() -> None:
    op.drop_column("document_chunks", "embedding")
