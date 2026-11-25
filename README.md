# Research Assistant

[![CI](https://github.com/Taikaho/research-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Taikaho/research-assistant/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A local-first assistant for asking questions about your own scientific PDFs.
Upload articles, group them into collections, and get answers that cite the exact
passages they came from: document, PDF page and the original text, one click
away from the page in the PDF.

- **Source-traceable answers.** Every claim cites numbered passages; the sources
  are assembled by the backend from retrieval metadata, never written by the model.
- **Says when it doesn't know.** A relevance gate answers *insufficient evidence*
  without calling the model when no passage is similar enough, and the model is
  instructed to do the same when the passages don't answer the question.
- **Local by default.** Embeddings run in-process and answers come from a local
  Ollama model. A cloud model (any OpenAI-compatible API) is opt-in and can be
  switched off entirely.
- **Multilingual retrieval.** Ask in Finnish (or 100+ other languages) about
  English papers, using the `BAAI/bge-m3` embedding model.
- **Measured, not assumed.** An evaluation framework reports retrieval,
  abstention and citation quality, and calibrates the relevance gate.

Stack: Python 3.12, FastAPI, PostgreSQL + pgvector, Sentence Transformers,
Ollama, React 19 + TypeScript, Docker Compose.

---

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start-docker-compose)
- [Try it with sample papers](#try-it-with-sample-papers)
- [Configuration](#configuration)
- [Privacy](#privacy)
- [How it works](#how-it-works)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Development](#development)
- [Documentation](#documentation)

---

## Requirements

| Software | Version | Used for |
|---|---|---|
| Docker with Compose | Compose v2.24+ (Docker Desktop 4.27+) | Frontend, backend and database |
| [Ollama](https://ollama.com) | recent | Local language model, runs on the host |

### Model requirements

| Model | Role | Download | Memory | Notes |
|---|---|---|---|---|
| `BAAI/bge-m3` | Embeddings (default) | ~2.3 GB, on first use | ~2.5 GB peak while embedding | CPU is enough; roughly 0.6 s per chunk on a laptop CPU, so a 20-page paper takes about 30 s |
| `qwen2.5:7b` via Ollama | Answers (recommended) | 4.7 GB | ~6 GB | Follows the citation and abstention rules best in our evaluation |
| `llama3.2:3b` via Ollama | Answers (low memory) | 2.0 GB | ~3 GB | Faster, but leaves out citations more often (see [Limitations](#limitations)) |

Figures are approximate and measured on CPU. A GPU speeds up Ollama
considerably. The Docker image embeds on the CPU; a local install with a CUDA
build of PyTorch uses the GPU automatically (`EMBEDDING_DEVICE`).
Plan for about 8 GB of free RAM with `qwen2.5:7b` and about 10 GB of disk
for the images, models and database.

---

## Quick start (Docker Compose)

```bash
# 1. Install Ollama on the host and pull a model
ollama pull qwen2.5:7b

# 2. Optional: settings such as the default model or a cloud API key
cp .env.example .env

# 3. Build and start the frontend, backend and database
docker compose up --build -d

# 4. Check that the backend is up
curl http://localhost:8000/health
```

Open **http://localhost:3000**. The API and its interactive documentation are at
`http://localhost:8000/docs`; [docs/api.md](docs/api.md) has an endpoint overview.
The same steps as the UI, from the command line:

```bash
# Upload a PDF into a collection (create one first with POST /api/v1/collections)
curl -F "file=@paper.pdf" -F "collection_ids=<collection-uuid>" \
     http://localhost:8000/api/v1/documents

# Semantic search within collections
curl -X POST http://localhost:8000/api/v1/search -H "Content-Type: application/json" \
     -d '{"query": "How does ColBERTv2 compress its index?", "collection_ids": ["<collection-uuid>"]}'

# Ask a question; the answer cites [n] and lists its sources
curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" \
     -d '{"question": "Which retrieval functionalities does M3-Embedding support?", "collection_ids": ["<collection-uuid>"]}'
```

The first upload or search downloads the embedding model (~2.3 GB) into the
`model_cache` volume, so it takes a few minutes; later starts reuse it.
Uploaded PDFs are stored in `./storage/`, everything else in the `postgres_data`
volume. Stop with `docker compose down` (add `-v` to delete the data volumes).

---

## Try it with sample papers

`samples/download_samples.py` downloads five openly licensed (CC BY 4.0) arXiv
papers on dense retrieval and retrieval-augmented generation into
`samples/pdfs/`. It needs only Python 3.10+:

```bash
python samples/download_samples.py
```

| File | Paper |
|---|---|
| `bge-m3.pdf` | M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation |
| `self-rag.pdf` | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection |
| `colbertv2.pdf` | ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction |
| `contriever.pdf` | Unsupervised Dense Information Retrieval with Contrastive Learning |
| `e5.pdf` | Text Embeddings by Weakly-Supervised Contrastive Pre-training |

In the app, create a collection under **Collections**, upload the five PDFs into
it under **Library**, then ask in **Chat**:

- Which three retrieval functionalities does M3-Embedding support?
- What are reflection tokens in Self-RAG, and which types are there?
- How does ColBERTv2 reduce the storage footprint of late interaction?
- How was the CCPairs dataset filtered before training E5?
- *Miten Contriever koulutetaan ilman merkittyä opetusdataa?* (Finnish: How is
  Contriever trained without labelled data?)
- What was the population of Helsinki in 2020? (not in the papers: expect
  *insufficient evidence*)

Click a citation such as `[2]` to open the PDF at the cited page with the passage
highlighted in the page text.

---

## Configuration

Settings are environment variables; `.env.example` documents every one. With
Docker Compose, put them in `.env` at the repository root: Compose passes it to
the backend, while the database URL, storage path, model cache and Ollama
address are set by `docker-compose.yml`. A backend run from `backend/` outside
Docker reads `backend/.env` instead.

| Variable | Default | Description |
|---|---|---|
| `LLM_DEFAULT_PROVIDER` | `ollama` | Provider used when a request names none |
| `OLLAMA_DEFAULT_MODEL` | first installed | Default Ollama model, e.g. `qwen2.5:7b` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama address (Docker Compose uses the host's) |
| `OPENAI_API_KEY` | unset | Enables the OpenAI-compatible cloud provider |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint (OpenRouter, Groq, vLLM, ...) |
| `LLM_LOCAL_ONLY` | `false` | `true` disables cloud providers entirely |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Sentence Transformers model; changing it requires re-indexing and a matching `EMBEDDING_QUERY_PREFIX` and `CHAT_MIN_RELEVANCE` |
| `EMBEDDING_DEVICE` | auto-detect | `cpu` or `cuda`; the Docker image has CPU-only PyTorch |
| `CHAT_MIN_RELEVANCE` | `0.45` | The relevance gate: below this similarity the answer is *insufficient evidence* without calling the model |
| `RETRIEVAL_TOP_K` | `5` | Passages returned by search and given to the model |
| `STORAGE_PATH` | `./storage` | Where uploaded PDFs are stored |
| `DATABASE_URL` | `postgresql+asyncpg://...@localhost:5432/research_assistant` | Database for a backend run outside Docker |

### Changing the embedding model

Each chunk records the model that embedded it, and search only compares vectors
from the active model. After changing `EMBEDDING_MODEL_NAME`, re-index once with
the banner on the Library page, `POST /api/v1/embeddings/reindex`, or
`python -m app.embeddings.reindex`. To use the English-only `BAAI/bge-small-en-v1.5`
instead, also set
`EMBEDDING_QUERY_PREFIX="Represent this sentence for searching relevant passages: "`
and `CHAT_MIN_RELEVANCE=0.6`.

---

## Privacy

- **Local by default**: PDFs, extracted text, embeddings and conversations are
  stored only in your database and `./storage/`; embeddings and Ollama answers
  are computed on your machine.
- **Cloud only when enabled**: a cloud model is available only when
  `OPENAI_API_KEY` is set, and is used only when selected. It then receives the
  question, the retrieved passages and recent conversation turns, never whole
  documents. The UI marks cloud models and every response reports
  `cloud_processing`.
- **Kill switch**: `LLM_LOCAL_ONLY=true` disables cloud providers even when a key
  is present.

Details, including exactly what is sent and how to delete data:
[docs/privacy.md](docs/privacy.md).

---

## How it works

1. **Ingestion**: an uploaded PDF is stored, its text extracted page by page,
   split into overlapping chunks that remember their PDF pages, and embedded
   locally.
2. **Retrieval**: a question is embedded and compared with the chunks of the
   selected collections only (scope is enforced in SQL).
3. **Relevance gate**: if the best passage is below `CHAT_MIN_RELEVANCE`, the
   answer is *insufficient evidence* and no model is called.
4. **Generation**: the model sees numbered passages `[1]..[n]` and must cite them;
   citation numbers outside that range are removed.
5. **Sources**: the backend attaches each passage's document, PDF pages and text,
   marking which ones the answer cites.

Diagrams and design details: [docs/architecture.md](docs/architecture.md).
Decisions with their trade-offs: [docs/decisions/](docs/decisions/).

---

## Evaluation

`python -m app.evaluation` runs the production ingestion, retrieval and chat code
against a question dataset in a separate database, and reports Recall@k, MRR,
relevance-gate calibration, abstention, citation hit rate and precision, latency
and tokens. Every run records its full configuration for reproducibility.

Results so far are on a **synthetic smoke dataset** (three short fictional
articles, 23 questions in English and Finnish). They show the pipeline works
end to end and how models differ; they are **not evidence of quality on real
articles**. An evaluation on real papers with manually verified questions is
planned.

| bge-m3 + | Answer rate | Correct abstention | Citation hit | Citation precision | Uncited answers |
|---|---|---|---|---|---|
| `llama3.2:3b` | 100 % | 60 % | 50.0 % | 75.0 % | 38.9 % |
| `qwen2.5:7b` | 94.4 % | 100 % | 76.5 % | 86.4 % | 23.5 % |

Method, metrics and how to write a dataset: [evaluation/README.md](evaluation/README.md).

---

## Limitations

- **Small local models cite inconsistently.** In the smoke evaluation,
  `llama3.2:3b` left 39 % of answers without citations and once stated a fact
  for an unanswerable question; `qwen2.5:7b` is better but still leaves about a
  quarter of answers uncited. Uncited answers are shown with all their sources.
- **Page numbers are PDF page indexes.** Printed page labels ("p. 1043") are not
  detected for most PDFs, so a source says "PDF p. 3" rather than the journal page.
- **The relevance gate is provisional.** `0.45` was calibrated on the synthetic
  dataset and will be recalibrated on real articles.
- **Text-based PDFs only.** There is no OCR; scanned PDFs produce no text.
  Tables and formulas are extracted as plain text.
- **Embedding runs on the CPU in Docker** and happens during the upload request,
  so large uploads take a while.
- **Single user, no authentication.** It is meant to run on your own machine;
  do not expose ports 3000 or 8000 to a network.

---

## Development

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
# CPU-only torch first (the default wheel bundles multi-GB CUDA libraries)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# Database only, published on localhost:5432
docker compose -f ../docker-compose.yml -f ../docker-compose.dev.yml up -d db

alembic upgrade head             # uses DATABASE_URL (default: localhost:5432)
uvicorn app.main:app --reload
```

To run the whole stack in Docker with the backend source mounted and
auto-reloading, add the development override:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### Frontend

Requires Node.js 24+.

```bash
cd frontend
npm ci
npm run dev        # http://localhost:5173, proxies /api to http://localhost:8000
```

Set `BACKEND_URL=http://host:port` to proxy elsewhere. No CORS configuration is
needed in either setup: the browser always talks to a single origin.

### Tests

```bash
cd backend
pytest                       # unit tests (default, no database)
pytest -m integration        # needs the test database below
pytest -m model              # downloads and runs the real embedding model
ruff check app/ tests/
mypy app/

cd ../frontend
npm run typecheck
npm test                     # Vitest, API calls mocked
npm run build
```

Integration tests use a separate database on port 5433:

```bash
docker compose -f docker-compose.test.yml up -d
cd backend
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/research_assistant_test \
  alembic upgrade head
pytest -m integration -v
```

CI runs the unit tests, lint and type checks, the integration tests against
pgvector, the frontend checks, and builds both Docker images.

### Release check

Before tagging a release, run the full stack from a clean state:
`bash scripts/release-check.sh` walks through starting Docker and Ollama, a
clean `docker compose up --build`, the first upload (which downloads the
embedding model), a cited answer through the API and a check in the browser.

### Project structure

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
│   │   ├── evaluation/   # Evaluation CLI and metrics
│   │   ├── llm/          # LLMProvider, Ollama and OpenAI-compatible adapters
│   │   └── retrieval/    # Semantic search with collection scoping
│   ├── migrations/       # Alembic migrations
│   └── tests/            # Unit and integration tests
├── frontend/             # React + TypeScript single-page app
├── docs/                 # Architecture, API, privacy, schema, decisions
├── evaluation/           # Evaluation configs and datasets
├── samples/              # Sample paper download script
├── docker-compose.yml       # Release stack
├── docker-compose.dev.yml   # Development override (source mount, reload, DB port)
└── docker-compose.test.yml  # Test database only
```

---

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | System, ingestion and question-answering diagrams; module design |
| [docs/api.md](docs/api.md) | Endpoint overview and curl examples |
| [docs/privacy.md](docs/privacy.md) | What is stored, what leaves the machine, deleting data |
| [docs/database-schema.md](docs/database-schema.md) | Tables and relationships |
| [docs/decisions/](docs/decisions/) | Architecture decision records |
| [evaluation/README.md](evaluation/README.md) | Evaluation method, metrics and results |
| [GLOSSARY.md](GLOSSARY.md) | Domain terms |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [SPEC.md](SPEC.md) | Original specification and development phases |

## License

[MIT](LICENSE). The sample papers are not included in this repository; each is
licensed CC BY 4.0 by its authors on arXiv.
