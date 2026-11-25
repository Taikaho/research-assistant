# API reference

The backend serves interactive, always-current documentation generated from the
code at `http://localhost:8000/docs` (OpenAPI schema at `/openapi.json`). This
page is an overview. In Docker Compose the same API is reachable through the
frontend at `http://localhost:3000/api/v1/...`.

## Endpoints

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
| GET | `/api/v1/documents/{id}/file` | The original PDF, served inline (open at a page with `#page=N`) |
| GET | `/api/v1/documents/{id}/pages/{page}` | Extracted text of one page (zero-based PDF page index) |
| POST | `/api/v1/documents/{id}/reprocess` | Re-extract, re-chunk and re-embed a document (collection membership is kept) |
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

## Conventions

- **Page numbers** in the API are zero-based PDF page indexes. The UI shows them
  one-based ("PDF p. 3") and opens the PDF with `#page=N+1`. Printed page labels
  are returned when the PDF defines them.
- **Duplicate uploads**: uploading a file that already exists (same SHA-256)
  returns the existing document and adds it to any `collection_ids` given,
  instead of storing a second copy.
- **Scope**: search and chat only ever use documents in the selected
  collections. A follow-up question in a conversation reuses the conversation's
  stored scope unless `collection_ids` is sent again.

## Examples

Create a collection and upload a PDF into it (repeat `collection_ids` for several
collections):

```bash
curl -X POST http://localhost:8000/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "Retrieval papers"}'

curl -F "file=@samples/pdfs/bge-m3.pdf" \
     -F "collection_ids=<collection-uuid>" \
     http://localhost:8000/api/v1/documents
```

Semantic search:

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How does ColBERTv2 compress its index?", "collection_ids": ["<collection-uuid>"], "top_k": 5}'
```

Ask a question:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Which retrieval functionalities does M3-Embedding support?", "collection_ids": ["<collection-uuid>"], "model_provider": "ollama", "model_name": "qwen2.5:7b"}'
```

The answer cites passages as `[1]`, `[2]`, ... Each entry in `sources` carries
its `citation_number`, whether the answer `cited` it, the document, PDF page
numbers, score and the original passage. When the selected collections do not
support an answer, `insufficient_evidence` is `true` and
`insufficient_evidence_reason` says whether the relevance gate stopped the
question (`no_relevant_passages`) or the model declined (`model_reported`).
`cloud_processing` is `true` whenever passages were sent to a cloud provider.
Pass the returned `conversation_id` to ask a follow-up question.
