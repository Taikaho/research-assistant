from pydantic import SecretStr
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
    # Default: BAAI/bge-m3 (multilingual, 1024 dims, 8192-token input), chosen in
    # Phase 6 so that questions in Finnish (and other languages) retrieve English
    # passages; the English-only bge-small-en-v1.5 scored Finnish questions like
    # unrelated text. Models with a 256-token limit (e.g. all-MiniLM-L6-v2) would
    # silently ignore the second half of every ~500-token chunk.
    embedding_provider: str = "sentence_transformers"
    embedding_model_name: str = "BAAI/bge-m3"
    # Optional Hugging Face revision (commit hash or tag) to pin the model weights.
    embedding_model_revision: str | None = None
    # Directory for downloaded model weights; None uses the Hugging Face default cache.
    embedding_cache_dir: str | None = None
    # Instruction prepended to queries only. Must match the model: "" for bge-m3;
    # "Represent this sentence for searching relevant passages: " for the English
    # bge-*-en-v1.5 models.
    embedding_query_prefix: str = ""
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

    # LLM providers (Phase 4).
    # Default provider used when a chat request does not name one: "ollama" or "openai".
    llm_default_provider: str = "ollama"
    # Hard switch: when true, cloud providers are never registered, so document
    # contents can never leave the machine regardless of what a request asks for.
    llm_local_only: bool = False
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # Ollama (local)
    ollama_base_url: str = "http://localhost:11434"
    # Model used when a request names none; None = first installed model.
    ollama_default_model: str | None = None
    ollama_timeout_seconds: float = 180.0
    # Context window passed to Ollama. Its built-in default (2048-4096 tokens in many
    # versions) silently truncates prompts holding several ~500-token passages.
    ollama_num_ctx: int = 8192

    # OpenAI-compatible cloud API (OpenAI, Azure OpenAI proxies, OpenRouter, Groq, ...).
    # The provider is only enabled when an API key is set.
    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_default_model: str | None = None
    # Display/identifier name of the cloud provider, e.g. "openai" or "openrouter".
    openai_provider_name: str = "openai"
    openai_timeout_seconds: float = 60.0
    # Retries for rate limits (429) and transient server errors (5xx).
    openai_max_retries: int = 2
    # Request field for the output token limit. Most compatible APIs use "max_tokens";
    # OpenAI's reasoning models require "max_completion_tokens".
    openai_token_param: str = "max_tokens"

    # Chat / RAG (Phase 4)
    # Passages given to the LLM; None = retrieval_top_k.
    chat_top_k: int | None = None
    # Best passage must reach this cosine similarity, otherwise the question is answered
    # with an insufficient-evidence response without calling the LLM. A coarse first gate
    # that only removes clearly unrelated questions; the model's own insufficient-evidence
    # signal is the second. The value depends on the embedding model: 0.45 is calibrated
    # for bge-m3 with `python -m app.evaluation` (bge-small-en-v1.5 needed ~0.6).
    # Provisional until recalibrated on the manually verified dataset.
    chat_min_relevance: float = 0.45
    # Previous messages (user + assistant) included in the prompt for follow-ups.
    chat_history_messages: int = 6

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
