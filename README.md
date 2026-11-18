# Research Assistant

An AI-powered scientific document assistant. Upload PDF articles, organise them
into collections, and ask questions with full source traceability — including
document title, PDF page number, and the original supporting passage.

> **Status:** Phase 3 complete — local embeddings and semantic retrieval.
> See `SPEC.md` for the full specification and development phases.

---

## Prerequisites

| Tool | Minimum version |
|---|---|
| Docker Desktop | 4.x |
| Docker Compose | v2 (bundled with Docker Desktop) |
| Python | 3.12+ (local dev / testing only) |

---

## Quick start (Docker Compose)

```bash
# 1. Copy and review environment variables
cp .env.example .env

# 2. Start the application
docker compose up --build

# 3. Verify
curl http://localhost:8000/health
```

The backend is available at `http://localhost:8000`.
Interactive API docs: `http://localhost:8000/docs`

---

## Local development setup

```bash
cd backend

# Install dependencies (use a virtual environment)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
# CPU-only torch first (the default wheel bundles multi-GB CUDA libraries)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# Copy env file
cp ../.env.example ../.env

# Start the development database
docker compose -f ../docker-compose.yml up db -d

# Run migrations
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/research_assistant \
  alembic upgrade head

# Start the backend with auto-reload
uvicorn app.main:app --reload
```

---

## Running tests

### Unit tests (default — no database required)

```bash
cd backend
pytest
# or explicitly:
pytest -m unit -v
```

### Integration tests (require a running test database)

```bash
# 1. Start the test database (port 5433 — separate from development DB)
docker compose -f docker-compose.test.yml up -d

# 2. Run migrations against the test database
cd backend
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/research_assistant_test \
  alembic upgrade head

# 3. Run integration tests
pytest -m integration -v

# 4. Stop test database when done
docker compose -f docker-compose.test.yml down
```

Integration tests use a deterministic fake embedding model; no download needed.

### Real embedding model test (downloads the model on first run)

```bash
pytest -m model -v
```

### Run all tests

```bash
pytest -m ''
```

### Lint and type check

```bash
ruff check app/ tests/
mypy app/
```

---

## Environment variables

See `.env.example` for all variables with descriptions.

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://...@localhost:5432/research_assistant` | Development database |
| `TEST_DATABASE_URL` | `postgresql+asyncpg://...@localhost:5433/research_assistant_test` | Test database (integration tests) |
| `STORAGE_PATH` | `./storage` | Local PDF storage directory |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Sentence Transformers model; changing it requires re-indexing |
| `EMBEDDING_CACHE_DIR` | Hugging Face default | Where model weights are downloaded |
| `RETRIEVAL_TOP_K` | `5` | Default number of passages returned by search |
| `LLM_DEFAULT_PROVIDER` | `ollama` | Provider used when a chat request names none |
| `LLM_LOCAL_ONLY` | `false` | `true` disables cloud providers entirely |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama service |
| `OLLAMA_DEFAULT_MODEL` | first installed | Default Ollama model |
| `OPENAI_API_KEY` | unset | Enables the OpenAI-compatible cloud provider |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint |
| `CHAT_MIN_RELEVANCE` | `0.6` | Below this, answer "insufficient evidence" without calling the LLM |
| `DEBUG` | `false` | Enable SQLAlchemy query logging |

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Application and database health |
| GET | `/docs` | Interactive API documentation |
| POST | `/api/v1/collections` | Create a collection |
| GET | `/api/v1/collections` | List collections |
| GET | `/api/v1/collections/{id}` | Get a collection |
| PATCH | `/api/v1/collections/{id}` | Rename or update a collection |
| DELETE | `/api/v1/collections/{id}` | Delete a collection (its documents are kept) |
| GET | `/api/v1/collections/{id}/documents` | List the documents in a collection |
| PUT | `/api/v1/collections/{id}/documents/{document_id}` | Add an existing document to a collection (idempotent) |
| DELETE | `/api/v1/collections/{id}/documents/{document_id}` | Remove a document from a collection (the document is kept) |
| POST | `/api/v1/documents` | Upload a PDF (multipart `file`; repeat `collection_ids` to add it to collections) |
| GET | `/api/v1/documents` | List documents |
| GET | `/api/v1/documents/{id}` | Get document metadata and processing status |
| DELETE | `/api/v1/documents/{id}` | Permanently delete a document, its pages, chunks and stored PDF |
| POST | `/api/v1/search` | Semantic search within selected collections |
| POST | `/api/v1/chat` | Answer a question from the selected collections, with sources |
| GET | `/api/v1/conversations` | List conversations |
| GET | `/api/v1/conversations/{id}` | Conversation history with sources |
| DELETE | `/api/v1/conversations/{id}` | Delete a conversation (documents are kept) |
| GET | `/api/v1/models` | Configured providers and their available models |
| GET | `/api/v1/models/providers` | Configured providers (no network calls) |
| GET | `/api/v1/models/health` | Provider availability |
| GET | `/api/v1/embeddings/status` | Active embedding model and indexed chunk count |
| POST | `/api/v1/embeddings/reindex` | Embed all chunks missing an embedding from the configured model |

Uploading a file that already exists (same SHA-256) returns the existing document
and adds it to any `collection_ids` given, instead of storing a second copy.

Example upload into two collections:

```bash
curl -F "file=@paper.pdf"      -F "collection_ids=<collection-uuid-1>"      -F "collection_ids=<collection-uuid-2>"      http://localhost:8000/api/v1/documents
```

Example search:

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How does LDA differ from KNN?", "collection_ids": ["<collection-uuid>"], "top_k": 5}'
```

The first upload or search downloads the embedding model (~130 MB).

Example question (needs an Ollama model, e.g. `ollama pull llama3.2:3b`):

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How does LDA differ from KNN?", "collection_ids": ["<collection-uuid>"], "model_provider": "ollama", "model_name": "llama3.2:3b"}'
```

The answer cites passages as `[1]`, `[2]`, ...; each entry in `sources` carries
its `citation_number`, document, PDF page numbers and the original passage.
Pass the returned `conversation_id` to ask a follow-up question.

---

## Project structure

```
research-assistant/
├── backend/
│   ├── app/
│   │   ├── api/          # HTTP routing
│   │   ├── chat/         # RAG pipeline, prompts, citations, conversations
│   │   ├── collections/  # Collections and collection membership
│   │   ├── core/         # Settings (pydantic-settings)
│   │   ├── database/     # SQLAlchemy models and session
│   │   ├── documents/    # PDF upload, extraction and chunking
│   │   ├── embeddings/   # EmbeddingProvider, Sentence Transformers, re-indexing
│   │   ├── llm/          # LLMProvider, Ollama and OpenAI-compatible adapters
│   │   └── retrieval/    # Semantic search with collection scoping
│   ├── migrations/       # Alembic migrations
│   ├── tests/            # Unit and integration tests
│   ├── alembic.ini
│   ├── pyproject.toml
│   └── Dockerfile
├── docs/                 # Architecture and schema documentation
├── evaluation/           # RAG evaluation framework (Phase 6)
├── frontend/             # React frontend (Phase 5)
├── storage/              # Local PDF storage (excluded from Git)
├── docker-compose.yml    # Development environment
└── docker-compose.test.yml  # Test database only
```

---

## Development phases

| Phase | Description | Status |
|---|---|---|
| 1 | Project foundation | Complete |
| 2 | Collections and document ingestion | Complete |
| 3 | Embeddings and retrieval | Complete |
| 4 | LLM integration | Complete |
| 5 | Frontend | Planned |
| 6 | Evaluation | Planned |
| 7 | Portfolio release | Planned |
