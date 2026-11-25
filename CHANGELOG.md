# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - Unreleased

First stable release: a complete, local-first research assistant for PDF
articles with source-traceable answers.

### Added

- **Collections and documents**: create, rename and delete collections; upload
  PDFs into one or more collections; duplicate detection by SHA-256; per-page
  text extraction with Unicode NFKC normalisation; title, author and year from
  PDF metadata; reprocess and delete documents.
- **Retrieval**: local embeddings with Sentence Transformers (default
  multilingual `BAAI/bge-m3`) stored in PostgreSQL + pgvector; semantic search
  scoped to the selected collections in SQL; near-duplicate filtering; each
  chunk records its embedding model, so a model change never mixes vector
  spaces and re-indexing needs no migration.
- **Question answering**: provider-agnostic RAG pipeline with Ollama (local)
  and any OpenAI-compatible API (cloud, opt-in); numbered-passage citations
  validated by the backend; a relevance gate and model-reported *insufficient
  evidence*; follow-up questions within conversations; `cloud_processing` flag
  on every answer; `LLM_LOCAL_ONLY` switch.
- **Web interface**: React + TypeScript app for chat with clickable citations,
  the document library, collections, model selection with a Local/Cloud badge,
  and a source viewer that opens the PDF at the cited page and highlights the
  passage.
- **Evaluation**: `python -m app.evaluation` measures retrieval (Recall@k,
  Precision@k, Hit@k, MRR), calibrates the relevance gate, and scores
  abstention, citation correctness, latency and tokens, with optional LLM-judge
  scores and a manual review sheet; every run records its full configuration.
  A synthetic smoke dataset in English and Finnish.
- **Release packaging**: Docker Compose release stack with a development
  override, Docker image builds in CI, sample-paper download script,
  architecture diagrams, API overview, privacy documentation, MIT license.

### Known limitations

- Small local models (`llama3.2:3b`) often answer without citations;
  `qwen2.5:7b` is recommended.
- Sources show PDF page indexes; printed page labels are rarely detected.
- The relevance gate (0.45 for bge-m3) is calibrated on synthetic data only.
- No OCR for scanned PDFs; no authentication (single-user, local use).
