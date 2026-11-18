from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    app_name: str = "Research Assistant"
    app_version: str = "0.1.0"
    debug: bool = False

    # Database (development / Docker Compose default)
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/research_assistant"
    )

    # Database (integration tests only — never the same as database_url)
    # Runs on port 5433 via docker-compose.test.yml to prevent accidental dev-DB writes.
    test_database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5433/research_assistant_test"
    )

    # Local document storage path (mounted as a Docker volume)
    storage_path: str = "./storage"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Document upload limits
    max_upload_size_mb: int = 50

    # Text chunking (Phase 2). Character-based approximation: ~2000 chars ≈ 500 tokens.
    # Cross-page chunking and token-aware splitting are Phase 3 refinements.
    chunk_size: int = 2000
    chunk_overlap: int = 400

    # Embedding model (Phase 3).
    # The vector dimension is resolved programmatically by loading the model — it is
    # never hardcoded. Changing the model requires re-indexing (POST /embeddings/reindex);
    # retrieval only ever compares vectors produced by the currently configured model.
    #
    # Default: BAAI/bge-small-en-v1.5 (384 dims, 512-token input). Chunks are ~500
    # tokens, so models with a 256-token limit (e.g. all-MiniLM-L6-v2) would silently
    # ignore the second half of every chunk.
    embedding_provider: str = "sentence_transformers"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    # Optional Hugging Face revision (commit hash or tag) to pin the model weights.
    embedding_model_revision: str | None = None
    # Directory for downloaded model weights; None uses the Hugging Face default cache.
    embedding_cache_dir: str | None = None
    # Instruction prepended to queries only (BGE retrieval convention). Must match the
    # model: set to "" for models that do not use a query instruction.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_batch_size: int = 32
    embedding_device: str | None = None  # e.g. "cpu", "cuda"; None = auto-detect

    # Retrieval (Phase 3)
    retrieval_top_k: int = 5
    retrieval_max_top_k: int = 50
    # Cosine similarity floor; results below it are dropped.
    retrieval_min_score: float = 0.0
    # Fetch top_k * multiplier candidates so near-duplicates can be removed.
    retrieval_candidate_multiplier: int = 3
    # Word-trigram Jaccard similarity at or above which two passages count as duplicates.
    retrieval_dedup_threshold: float = 0.9

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
