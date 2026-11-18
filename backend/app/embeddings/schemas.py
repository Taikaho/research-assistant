import uuid
from datetime import datetime

from pydantic import BaseModel


class EmbeddingModelResponse(BaseModel):
    id: uuid.UUID
    name: str
    version: str | None
    dimensions: int
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class IndexStatusResponse(BaseModel):
    configured_model_name: str
    active_model: EmbeddingModelResponse | None
    total_chunks: int
    indexed_chunks: int
    # True when some completed documents are not searchable with the configured model.
    reindex_required: bool


class ReindexResponse(BaseModel):
    model: EmbeddingModelResponse
    documents_reindexed: int
    chunks_reindexed: int
