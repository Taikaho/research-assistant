"""Add unique constraint on collections.name and change collection_documents FK to CASCADE.

Revision ID: 0002
Revises: a1b2c3d4e5f6
Create Date: 2026-11-12
"""

from alembic import op

revision = "0002"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Unique constraint on collection name (business rule: no duplicate collection names)
    op.create_unique_constraint("uq_collections_name", "collections", ["name"])

    # Change document_id FK in collection_documents from RESTRICT to CASCADE so that
    # deleting a document automatically removes its collection memberships.
    op.drop_constraint(
        "collection_documents_document_id_fkey", "collection_documents", type_="foreignkey"
    )
    op.create_foreign_key(
        "collection_documents_document_id_fkey",
        "collection_documents",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint(
        "collection_documents_document_id_fkey", "collection_documents", type_="foreignkey"
    )
    op.create_foreign_key(
        "collection_documents_document_id_fkey",
        "collection_documents",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.drop_constraint("uq_collections_name", "collections", type_="unique")
