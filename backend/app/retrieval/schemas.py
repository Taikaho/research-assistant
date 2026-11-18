import uuid

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    # The search scope. Required: documents outside these collections never appear.
    collection_ids: list[uuid.UUID] = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("top_k")
    @classmethod
    def _cap_top_k(cls, value: int | None) -> int | None:
        if value is not None and value > settings.retrieval_max_top_k:
            raise ValueError(f"top_k must be at most {settings.retrieval_max_top_k}")
        return value


class SearchResult(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    filename: str
    authors: str | None
    publication_year: int | None
    chunk_index: int
    # Zero-based PDF page indexes covered by the passage (from PDF processing).
    pdf_page_numbers: list[int]
    text: str
    # Cosine similarity between query and passage, in [-1, 1].
    score: float


class SearchModelInfo(BaseModel):
    name: str
    version: str | None
    dimensions: int


class SearchResponse(BaseModel):
    query: str
    embedding_model: SearchModelInfo
    results: list[SearchResult]
    # Completed in-scope documents with chunks not yet embedded by the active
    # model; they were not searched. Run POST /embeddings/reindex.
    unindexed_document_count: int
