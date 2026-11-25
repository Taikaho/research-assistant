# Architecture

## Overview

Research Assistant is a local-first RAG application structured as a modular monolith.

```
Browser
  └─► Frontend: React SPA served by nginx (localhost:3000), proxies /api ─┐
                                                                        ▼
      FastAPI backend (localhost:8000)
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
`EmbeddingProvider` interface (`app/embeddings/base.py`). The default model is the
multilingual `BAAI/bge-m3` (1024 dims, 8192-token input), so questions in Finnish
retrieve English passages; until Phase 6 it was the English-only
`BAAI/bge-small-en-v1.5` (see `docs/decisions/0002-multilingual-embedding-model.md`).
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
interface (`app/llm/base.py`). The RAG pipeline does not know whether the
selected model is local (Ollama) or cloud-based; it only reports which was used.

- **Adapters**: `OllamaProvider` (native `/api/chat`, sets `num_ctx` so the
  prompt is not truncated) and `OpenAICompatibleProvider` (`/chat/completions`;
  works with OpenAI, OpenRouter, Groq, vLLM, ...). Both translate HTTP failures
  into provider-neutral errors (`LLMConnectionError`, `LLMTimeoutError`,
  `LLMModelNotFoundError`, `LLMAuthenticationError`, `LLMRateLimitError`,
  `LLMResponseError`), which the API maps to 503 / 504 / 400 / 502 / 429 / 502.
  The cloud adapter retries 429 and 5xx responses with backoff, honouring
  `Retry-After`, but not exhausted quota.
- **Registry** (`app/llm/registry.py`) builds providers from settings. The cloud
  provider is registered only when an API key is set and `LLM_LOCAL_ONLY` is
  false, so in local-only mode no request can send document text off the machine.
- API credentials are stored only in environment variables (`SecretStr`) and
  never persisted to the database, logged or returned by the API.

## Question answering (`POST /api/v1/chat`)

1. **Scope**: `collection_ids`, or the stored scope of `conversation_id`.
2. **Retrieval**: the Phase 3 search (scope enforced in SQL). In a conversation
   the question is also searched together with the previous question, and each
   passage keeps its best score: a follow-up like "which one was more accurate?"
   lacks the topic words on its own. The queries used are recorded in
   `generation_metadata.retrieval_queries`.
3. **Relevance gate**: passages below `CHAT_MIN_RELEVANCE` are dropped. If none
   remain, the answer is an insufficient-evidence response and the LLM is not
   called (`insufficient_evidence_reason: "no_relevant_passages"`).
4. **Prompt** (`app/chat/prompts.py`): passages are numbered `[1]..[n]` and
   labelled only with the document title (no page numbers or ids), so the model
   has nothing to copy into an invented reference. Rules: use only the passages,
   cite by number, never invent references, start with `INSUFFICIENT_EVIDENCE`
   when the passages don't answer the question, treat passage text as data.
   Earlier turns (question + answer, citation markers stripped) are included
   for follow-up questions.
5. **Parsing** (`app/chat/citations.py`): an `INSUFFICIENT_EVIDENCE` prefix sets
   `insufficient_evidence` (`reason: "model_reported"`); citation markers outside
   `1..n` are removed from the answer.
6. **Sources** are assembled by the backend from retrieval metadata: document,
   chunk id, zero-based PDF pages, printed page labels, score, full passage
   text, the `[n]` number and whether the answer cites it. Every source comes
   from the scoped retrieval, so it always belongs to the active collections.
7. **Persistence**: conversation, scope, user message, assistant message with
   model and `generation_metadata`, and `message_sources` are committed together
   only after a successful answer; a provider failure stores nothing.

Responses carry `cloud_processing: true` whenever passages were sent to a cloud
provider.

## Frontend

React 19 + TypeScript single-page app (`frontend/`), built with Vite and tested
with Vitest + Testing Library. Server state is handled by TanStack Query;
routing by React Router. In development the Vite server proxies `/api` to the
backend; in Docker Compose nginx serves the bundle and proxies `/api`, so the
browser only ever talks to one origin and the backend needs no CORS setup.

| Area | Route | Contents |
|---|---|---|
| Chat | `/chat`, `/chat/:id` | Conversation history, collection scope (always visible), model selection, answers with clickable `[n]` citations, sources, insufficient-evidence and error states |
| Library | `/library` | Upload (multiple PDFs, target collections, per-file progress), metadata, processing status, collection membership, reprocess, delete, re-index banner |
| Collections | `/collections` | Create, rename, delete, document counts, shortcuts to chat and library |
| Source viewer | modal | Original PDF at the cited page, page buttons for multi-page passages, page text with the passage highlighted, full passage |

Design points:

- **The frontend never creates sources.** Citation markers in the answer become
  links only when their number matches a source returned by the backend.
- **Page numbers**: the API uses zero-based PDF page indexes; the UI shows
  one-based "PDF p. N" (and the printed page label when known) and opens the
  PDF with `#page=N+1`.
- **PDF display** uses the browser's built-in viewer in an iframe (see
  `docs/decisions/0001-browser-pdf-viewer.md`). Highlighting is done in the
  page's extracted text, from which the chunk was cut verbatim.
- **Model choice** (provider, model, temperature, max tokens, top-k) is kept in
  `localStorage` and resolved against `GET /models`; a stale choice falls back
  to the backend default. The active model is shown in the header, with a
  Local/Cloud badge and an explicit warning when a cloud provider is selected.
- **Scope**: a new conversation sends the selected collections; follow-ups send
  `collection_ids` only when the user changes the scope, otherwise the stored
  conversation scope is used.

## Evaluation

`app/evaluation/` (CLI: `python -m app.evaluation`) measures the pipeline against
a dataset of questions with expected answers, PDF pages and supporting passages.
It drives the production code (ingestion, `retrieval.search`, `chat.answer_question`)
in a dedicated `*eval*` database and reports retrieval metrics (Recall@k,
Precision@k, Hit@k, MRR), relevance-gate calibration, abstention, citation
correctness, latency, tokens and optional LLM-judge scores, plus a CSV for
manual grading. Every run records the full configuration (models and their
revisions, settings, prompt versions, dataset fingerprint, git commit).
See `evaluation/README.md`.

| Module | Responsibility |
|---|---|
| `dataset.py` | Dataset schema (one-based PDF pages), validation, fingerprint |
| `matching.py` | When a chunk is relevant to a dataset passage |
| `metrics.py` | Metric definitions and gate calibration (pure functions) |
| `config.py` | Experiment configs and the settings they override |
| `runner.py` | Corpus preparation, dataset checks, retrieval and generation evaluation |
| `judge.py` | LLM-as-judge prompt and parsing (auxiliary signal) |
| `report.py` | Summaries, Markdown reports, comparisons, manual review sheet |
| `experiment.py` | One complete run and its reproducibility record |
