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

    # Embedding model (used starting Phase 3).
    # The actual vector dimension is resolved programmatically by loading this model —
    # it is never hardcoded here. See EmbeddingModel table and Phase 3 migration.
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
