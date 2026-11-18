# Architecture

## Overview

Research Assistant is a local-first RAG application structured as a modular monolith.

```
Browser
  └─► FastAPI backend (localhost:8000)
        ├─► PostgreSQL + pgvector  (localhost:5432)
        ├─► Local filesystem storage  (./storage/)
        └─► Ollama / Cloud LLM  (Phase 4)
```

All services run via Docker Compose. The application never requires a cloud-hosted
backend or database.

## Module structure

| Module | Responsibility |
|---|---|
| `app/api/` | HTTP routing and request/response schemas |
| `app/core/` | Application settings (pydantic-settings, env vars) |
| `app/database/` | SQLAlchemy models, engine, session factory |
| `app/collections/` | Collection CRUD (Phase 2) |
| `app/documents/` | PDF upload, processing, chunking (Phase 2) |
| `app/embeddings/` | Embedding model interface, vector generation (Phase 3) |
| `app/retrieval/` | Semantic search, collection filtering (Phase 3) |
| `app/llm/` | LLMProvider interface, Ollama and cloud adapters (Phase 4) |
| `app/chat/` | Conversation management, RAG pipeline (Phase 4) |
| `app/evaluation/` | Retrieval and generation metrics (Phase 6) |

## Embedding model design

Embeddings are generated locally with Sentence Transformers behind the
`EmbeddingProvider` interface (`app/embeddings/base.py`). The default model is
`BAAI/bge-small-en-v1.5` (384 dims, 512-token input, which covers ~500-token chunks).
Model inference runs in a worker thread so it never blocks the event loop.

To prevent silently mixing incompatible vector spaces:

1. The `embedding_models` table records each model's name, version (Hugging Face
   commit hash of the loaded weights) and **actual** output dimension, read from
   the model. The configured model is registered on first use and marked active.
2. `document_chunks.embedding` is a dimensionless pgvector `vector` column, so a
   model change needs no migration.
3. Each chunk stores `embedding_model_id`. Retrieval only compares the query
   with chunks embedded by the active model.
4. After changing the model, `POST /api/v1/embeddings/reindex` (or
   `python -m app.embeddings.reindex`) re-embeds stale chunks. Until then, search
   responses report `unindexed_document_count` instead of mixing models.

No ANN index exists yet: exact cosine search over the collection-filtered chunks
is fast at the expected corpus size and, unlike HNSW with filters, never loses
recall. HNSW can be added later as a partial expression index per model.

## Retrieval

`POST /api/v1/search` embeds the query and ranks chunks by cosine similarity.

- **Scope** is enforced in SQL: a chunk is a candidate only if its document is
  `completed` and belongs to at least one selected collection (`EXISTS`, so a
  document in several selected collections is returned once).
- `top_k * RETRIEVAL_CANDIDATE_MULTIPLIER` candidates are fetched, those below
  `min_score` dropped, and near-duplicates (word-trigram Jaccard ≥
  `RETRIEVAL_DEDUP_THRESHOLD`) removed before the top-k cut.
- Each result carries its chunk id, document id/title/filename, authors, year,
  zero-based PDF page numbers, passage text and score.

## LLM provider design

Business logic is isolated from provider-specific code via the `LLMProvider`
interface (Phase 4). The RAG pipeline does not know whether the selected model
is local (Ollama) or cloud-based.

API credentials are stored only in environment variables and never persisted to
the database or transmitted to the frontend.
