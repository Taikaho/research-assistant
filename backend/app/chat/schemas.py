import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    # Search scope. Required for a new conversation; for an existing one it
    # replaces the conversation's scope when given, otherwise the stored scope is used.
    collection_ids: list[uuid.UUID] | None = Field(default=None, min_length=1)
    conversation_id: uuid.UUID | None = None
    # None = configured defaults.
    model_provider: str | None = None
    model_name: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=32_768)
    top_k: int | None = Field(default=None, ge=1)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @field_validator("top_k")
    @classmethod
    def _cap_top_k(cls, value: int | None) -> int | None:
        if value is not None and value > settings.retrieval_max_top_k:
            raise ValueError(f"top_k must be at most {settings.retrieval_max_top_k}")
        return value


class SourceReference(BaseModel):
    """A passage given to the model. Assembled by the backend, never by the LLM."""

    # The [n] marker used in the answer text for this passage.
    citation_number: int
    # True when the answer text cites this passage.
    cited: bool
    document_id: uuid.UUID
    document_title: str
    filename: str
    chunk_id: uuid.UUID
    # Zero-based PDF page indexes covered by the passage (from PDF processing).
    pdf_page_numbers: list[int]
    # Printed page labels for the same pages, where detected (None otherwise).
    printed_page_numbers: list[str | None]
    relevance_score: float
    excerpt: str


class ModelInfo(BaseModel):
    provider: str
    name: str
    is_local: bool


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


InsufficientReason = Literal["no_relevant_passages", "model_reported"]


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    insufficient_evidence: bool
    insufficient_evidence_reason: InsufficientReason | None = None
    sources: list[SourceReference]
    # None when the LLM was not called (no relevant passages).
    model: ModelInfo | None
    # True when document passages were sent to a cloud provider for this answer.
    cloud_processing: bool
    usage: TokenUsage | None = None
    collection_ids: list[uuid.UUID]
    # Completed in-scope documents not yet indexed by the active embedding model.
    unindexed_document_count: int


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


class ConversationSummary(BaseModel):
    id: uuid.UUID
    title: str | None
    created_at: datetime
    collection_ids: list[uuid.UUID]
    message_count: int


class MessageRead(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    created_at: datetime
    model_provider: str | None
    model_name: str | None
    insufficient_evidence: bool
    generation_metadata: dict[str, object] | None
    sources: list[SourceReference]


class ConversationDetail(BaseModel):
    id: uuid.UUID
    title: str | None
    created_at: datetime
    collection_ids: list[uuid.UUID]
    messages: list[MessageRead]
