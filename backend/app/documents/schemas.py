import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    title: str
    authors: str | None
    publication_year: int | None
    page_count: int
    file_hash: str
    processing_status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ChunkResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    text: str
    page_numbers: Any | None
    chunk_metadata: Any | None

    model_config = {"from_attributes": True}
