"""Record how each answer was produced and which passages it cited.

messages.insufficient_evidence / generation_metadata: model configuration,
prompt version, retrieval settings and token usage per answer (FR-12).
message_sources.citation_number / cited: map the answer's [n] markers back to
the stored source chunks.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-18
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column(
            "insufficient_evidence", sa.Boolean(), nullable=False, server_default="false"
        ),
    )
    op.add_column("messages", sa.Column("generation_metadata", JSONB(), nullable=True))
    op.add_column("message_sources", sa.Column("citation_number", sa.Integer(), nullable=True))
    op.add_column(
        "message_sources",
        sa.Column("cited", sa.Boolean(), nullable=False, server_default="false"),
    )


def downgrade() -> None:
    op.drop_column("message_sources", "cited")
    op.drop_column("message_sources", "citation_number")
    op.drop_column("messages", "generation_metadata")
    op.drop_column("messages", "insufficient_evidence")
