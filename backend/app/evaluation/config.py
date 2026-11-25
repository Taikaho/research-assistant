"""Experiment configuration: which models and settings an evaluation run uses.

A config is a JSON file (see evaluation/configs/). Fields left out fall back to
the application settings, and the resolved values are written into every run's
results, so a run can be reproduced from its own output.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.config import Settings


class EmbeddingConfig(BaseModel):
    model_name: str | None = None
    revision: str | None = None
    # None = application setting. Must match the model ("" for bge-m3).
    query_prefix: str | None = None


class ChunkingConfig(BaseModel):
    chunk_size: int | None = Field(default=None, gt=0)
    chunk_overlap: int | None = Field(default=None, ge=0)


class RetrievalConfig(BaseModel):
    # Cut-offs for Recall@k / Precision@k / hit rate; the largest is the search depth.
    k_values: list[int] = Field(default=[1, 3, 5, 10], min_length=1)
    # Share of a dataset passage a chunk must contain to count as relevant.
    passage_match_threshold: float = Field(default=0.5, gt=0, le=1)
    # The recommended relevance gate is the highest threshold that still passes this
    # share of answerable questions (see metrics.recall_preserving_threshold).
    gate_min_pass_rate: float = Field(default=0.95, gt=0, le=1)


class LLMConfig(BaseModel):
    provider: str | None = None
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)


class GenerationConfig(LLMConfig):
    enabled: bool = True
    # Passages given to the model; None = application setting.
    top_k: int | None = Field(default=None, gt=0)
    # Relevance gate (CHAT_MIN_RELEVANCE); None = application setting.
    min_relevance: float | None = None


class JudgeConfig(LLMConfig):
    enabled: bool = False


class ExperimentConfig(BaseModel):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    description: str | None = None
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    judge: JudgeConfig = JudgeConfig()

    @property
    def search_depth(self) -> int:
        return max(self.retrieval.k_values)

    def apply(self, settings: Settings) -> None:
        """Override the process settings used by ingestion, retrieval and chat."""
        if self.embedding.model_name is not None:
            settings.embedding_model_name = self.embedding.model_name
            settings.embedding_model_revision = self.embedding.revision
        if self.embedding.query_prefix is not None:
            settings.embedding_query_prefix = self.embedding.query_prefix
        if self.chunking.chunk_size is not None:
            settings.chunk_size = self.chunking.chunk_size
        if self.chunking.chunk_overlap is not None:
            settings.chunk_overlap = self.chunking.chunk_overlap
        if self.generation.min_relevance is not None:
            settings.chat_min_relevance = self.generation.min_relevance


def load_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def settings_snapshot(settings: Settings) -> dict[str, object]:
    """Settings that influence results. Never includes credentials."""
    return {
        "embedding_provider": settings.embedding_provider,
        "embedding_model_name": settings.embedding_model_name,
        "embedding_model_revision": settings.embedding_model_revision,
        "embedding_query_prefix": settings.embedding_query_prefix,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "retrieval_top_k": settings.retrieval_top_k,
        "retrieval_min_score": settings.retrieval_min_score,
        "retrieval_candidate_multiplier": settings.retrieval_candidate_multiplier,
        "retrieval_dedup_threshold": settings.retrieval_dedup_threshold,
        "chat_top_k": settings.chat_top_k,
        "chat_min_relevance": settings.chat_min_relevance,
        "llm_temperature": settings.llm_temperature,
        "llm_max_tokens": settings.llm_max_tokens,
        "ollama_num_ctx": settings.ollama_num_ctx,
    }
