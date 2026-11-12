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

pgvector requires a fixed vector dimension per column. This dimension must match
the output of the configured embedding model. To prevent silently mixing
incompatible vector spaces:

1. The `embedding_models` table records each model's name, version, and **actual**
   output dimension (determined programmatically, not hardcoded).
2. The vector column in `document_chunks` is added in a **Phase 3 migration**
   after the model is configured and its dimension confirmed.
3. Each chunk stores an `embedding_model_id` FK so queries can be scoped to a
   single vector space.
4. Changing the embedding model requires re-indexing all documents.

## LLM provider design

Business logic is isolated from provider-specific code via the `LLMProvider`
interface (Phase 4). The RAG pipeline does not know whether the selected model
is local (Ollama) or cloud-based.

API credentials are stored only in environment variables and never persisted to
the database or transmitted to the frontend.
