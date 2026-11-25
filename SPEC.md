# Technical Project Specification
## Research Assistant – AI-powered Scientific Document Assistant

**Version:** 1.1  
**Project type:** Personal portfolio project  
**Deployment:** Local-first application  
**Primary focus:** RAG, AI application development, document processing, source traceability and evaluation.

---

# 1. Project Overview

## 1.1 Purpose

Research Assistant is a locally running AI-powered application that allows users to organize scientific articles into document collections and ask questions about their contents.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant passages from uploaded documents and generate answers using a selected language model.

A central requirement is source traceability. Every factual answer must include references to the original document and the corresponding PDF page number.

The application must also display the original source passages supporting its answers.

The system must not fabricate citations, page numbers or information that cannot be supported by the retrieved documents.

## 1.2 Primary Goals

- Run the complete application locally.
- Upload and manage scientific PDF documents.
- Organize documents into multiple collections.
- Extract text while preserving document and page metadata.
- Divide documents into searchable text chunks.
- Generate and store embeddings.
- Retrieve relevant passages using semantic search.
- Generate answers using local or cloud-based LLMs.
- Display document names, page numbers and original source passages.
- Indicate when available documents do not contain sufficient information.
- Support comparisons between retrieval configurations and language models.
- Provide a foundation for future RAG evaluation and experiment tracking.

## 1.3 Target Users

Initially, the application is intended for a single user.

The architecture should support future expansion to multiple users without requiring a complete redesign.

---

# 2. Deployment and Operating Model

## 2.1 Local-first Deployment

The application must run on the user's own computer.

The user accesses the application through a web browser.

Initial deployment:

- Frontend: localhost
- Backend: localhost
- PostgreSQL: local Docker container
- Ollama: locally installed service or container
- Document storage: local filesystem

Docker Compose should manage the application services.

The application must not require a cloud-hosted backend or database.

## 2.2 Local and Cloud LLM Support

The application must support two LLM execution modes.

### Local models

Initial provider: Ollama.

Requirements:

- Connect to a locally running Ollama instance.
- Retrieve available models where supported.
- Allow the user to select an installed model.
- Support model-specific configuration.
- Handle unavailable models and connection errors.

### Cloud models

The architecture must support external LLM APIs.

Requirements:

- Implement a provider abstraction.
- Store API credentials securely in environment variables.
- Allow the user to select a configured cloud provider and model.
- Handle API errors, rate limits and timeouts.
- Keep provider-specific logic outside the RAG pipeline.

The initial cloud provider can be selected during implementation. OpenAI-compatible APIs are a useful starting point because they allow integration with multiple compatible services.

The application itself remains local even when a cloud model is selected. Only the information required for the model request is transmitted to the external provider.

The user must be informed when cloud processing is enabled.

The application must never transmit document contents to a cloud provider when local-only mode is selected.

## 2.3 Embedding Models

Embedding generation should initially run locally using Sentence Transformers.

Requirements:

- Use a configurable embedding model.
- Record the model name and version.
- Ensure query embeddings and document embeddings use compatible models.
- Support re-indexing when the embedding model changes.

Embedding generation should be independent of the selected LLM provider.

---

# 3. Functional Requirements

## FR-01: Document Collections

The user must be able to create and manage multiple document collections.

Examples:

- Hyperspectral Imaging
- Machine Learning
- IoT and Embedded Systems
- Research Methods

Requirements:

- Create collections.
- Rename collections.
- Delete collections.
- View collection contents.
- Add existing documents to collections.
- Remove documents from collections.
- Use one or more collections as the search scope for a question.

A document should be reusable across multiple collections without storing duplicate PDF files or generating unnecessary duplicate embeddings.

Deleting a collection must not automatically delete the original documents.

The application must distinguish between removing a document from a collection and permanently deleting the document.

## FR-02: PDF Document Upload

The user must be able to upload PDF documents through the user interface.

Requirements:

- Support multiple PDF files.
- Validate file type and file size.
- Assign a unique identifier to each document.
- Allow documents to be assigned to one or more collections.
- Store the original PDF locally.
- Extract available document metadata:
  - Title
  - Authors
  - Publication year
  - Original filename
  - Number of pages
- Show processing status.

Initially supported:
- Text-based PDF documents.

OCR for scanned documents is outside the initial MVP scope.

## FR-03: Document Processing

Use PyMuPDF to extract text page by page.

Requirements:

- Preserve the original PDF page index.
- Attempt to extract printed page numbers where possible.
- Store document metadata alongside extracted text.
- Handle empty pages and extraction errors.
- Avoid unnecessary duplicate processing.

The PDF page number is the authoritative fallback reference.

Printed page numbers should be stored separately because they may differ from the PDF page index.

## FR-04: Text Chunking

Divide extracted text into chunks suitable for semantic retrieval.

Requirements:

- Configurable chunk size and overlap.
- Preserve document ID and page metadata.
- Preserve the relationship between each chunk and its original page or pages.
- Avoid mixing text from unrelated documents.
- Allow chunking configuration to be changed.

Initial configuration:

- Chunk size: approximately 500 tokens.
- Chunk overlap: approximately 100 tokens.

A chunk spanning multiple pages must retain references to all pages it covers.

## FR-05: Embedding Generation

Use Sentence Transformers for local embedding generation.

Requirements:

- Make embedding models configurable.
- Record model name and version.
- Store embeddings in PostgreSQL with pgvector.
- Support re-indexing when the embedding model changes.
- Separate embedding logic from retrieval and application logic.

Embedding model changes must not silently mix incompatible vector spaces.

## FR-06: Vector Database

Use PostgreSQL with pgvector.

The database must store:

- Document metadata.
- Document collections.
- Collection-document relationships.
- Extracted page information.
- Text chunks.
- Embeddings.
- Processing status.
- Conversations and messages.
- Source references.

The schema must support multiple collections and document reuse.

## FR-07: Semantic Retrieval

When the user submits a question, the application retrieves relevant chunks from the selected collections.

Requirements:

- Convert the question into an embedding.
- Retrieve relevant chunks using vector similarity.
- Support configurable top-k retrieval.
- Filter results by selected collections.
- Prevent documents outside the selected collections from appearing in search results.
- Return source metadata with every retrieved chunk.
- Avoid duplicate or nearly identical results when possible.

Initial retrieval method:
- Vector similarity search.

Future retrieval methods:
- PostgreSQL full-text search.
- Hybrid retrieval.
- Reranking models.

## FR-08: LLM Answer Generation

The application must generate answers using retrieved document passages as context.

Requirements:

- Support local Ollama models.
- Support cloud-based LLM providers through separate adapters.
- Allow the user to select the model.
- Keep provider-specific logic outside the RAG pipeline.
- Provide retrieved passages as context.
- Require answers to be grounded in supplied source material.
- Require the model to indicate when evidence is insufficient.
- Instruct the model not to invent references or page numbers.
- Separate answer content from source metadata.

The backend must assemble authoritative source references using retrieved chunk metadata.

The LLM must never be responsible for inventing authoritative citations.

## FR-09: Source References

Every document-supported factual answer must include source references.

Each reference must contain:

- Document title or filename.
- PDF page number.
- Printed page number, if available.
- Original source passage.
- Relevant chunk identifier.

Requirements:

- Sources must refer to uploaded documents.
- Page numbers must originate from PDF processing metadata.
- The user must be able to inspect the original source passage.
- The user must be able to open the PDF at the referenced page.
- Multiple sources must be supported.

The backend must validate that every referenced chunk belongs to an uploaded document included in the active search scope.

## FR-10: Chat Interface

The application must provide a conversational interface.

Requirements:

- Submit questions about uploaded documents.
- Display generated answers.
- Display source references.
- Display loading and error states.
- Preserve conversation history.
- Allow selection of one or more document collections.
- Allow follow-up questions.
- Allow the user to inspect retrieved source passages.

The active collection selection must be clearly visible in the interface.

## FR-11: Document Management

The user must be able to:

- View uploaded documents.
- View document metadata.
- View processing status.
- Delete documents.
- Reprocess documents.
- Add documents to collections.
- Remove documents from collections.

Deleting a document must also remove its associated chunks and embeddings.

Removing a document from a collection must not delete the original document.

## FR-12: Model Configuration

The application must provide a configuration interface for model selection.

Requirements:

- Select local or cloud execution.
- Select an available model.
- Display configured providers.
- Configure generation parameters where supported.
- Display the active model in the chat interface.
- Record the model and configuration used for each answer.

API credentials must not be stored in the database as plain text or exposed to the frontend.

---

# 4. Non-Functional Requirements

## NFR-01: Modularity

Implement the application as a modular monolith.

Separate responsibilities into:

- API layer.
- Collection management.
- Document processing.
- Embedding generation.
- Retrieval.
- LLM providers.
- Chat.
- Database access.
- Evaluation.

Business logic must not depend directly on a specific LLM or embedding provider.

## NFR-02: Reliability

- Document processing errors must not crash the application.
- Failed jobs must have identifiable status.
- Database operations must preserve data consistency.
- Invalid files must be rejected.
- Source metadata must never be silently discarded.
- Unavailable LLM providers must produce understandable errors.

## NFR-03: Security

- Validate uploaded files.
- Set reasonable upload limits.
- Prevent path traversal.
- Keep documents private.
- Store credentials outside source control.
- Treat document content as untrusted input.
- Protect against prompt injection in uploaded documents.

## NFR-04: Maintainability

- Use Python type hints.
- Follow consistent formatting and linting.
- Include automated tests.
- Document architectural decisions.
- Use clear interfaces between modules.

## NFR-05: Reproducibility

Record:

- Embedding model and version.
- LLM provider and model.
- Chunking configuration.
- Retrieval configuration.
- Prompt version.
- Collection and document identifiers.
- Evaluation results where applicable.

## NFR-06: Resource Efficiency

The application should be usable on a typical personal computer.

Requirements:

- Avoid loading unnecessary models into memory.
- Allow the user to select smaller local models.
- Handle models that do not fit available memory.
- Keep embedding generation configurable.
- Avoid unnecessary re-indexing.
- Allow local LLM execution to be disabled when using cloud models.

The system must not assume that the computer has a dedicated GPU.

---

# 5. Technical Architecture

## 5.1 Technology Stack

| Component | Technology |
|---|---|
| Backend | Python 3.12+ / FastAPI |
| Frontend | React / TypeScript |
| PDF processing | PyMuPDF |
| Embeddings | Sentence Transformers |
| Vector database | PostgreSQL / pgvector |
| Local LLM | Ollama |
| Cloud LLM | Configurable API provider |
| ORM | SQLAlchemy |
| Migrations | Alembic |
| Validation | Pydantic |
| Testing | pytest |
| Frontend testing | Vitest |
| Containers | Docker / Docker Compose |
| CI/CD | GitHub Actions |

Versions must be pinned and updated deliberately.

## 5.2 Provider Architecture

Implement separate interfaces for language models and embedding models.

Suggested interfaces:

### LLMProvider

Responsibilities:

- Generate responses.
- Accept system and user prompts.
- Accept retrieved context.
- Return generated text.
- Expose model identification.
- Handle provider-specific errors.

Implementations:

- OllamaProvider
- CloudLLMProvider

### EmbeddingProvider

Responsibilities:

- Generate document embeddings.
- Generate query embeddings.
- Expose model identification and vector dimensions.

Initial implementation:

- SentenceTransformerProvider

Provider selection must be handled through application configuration.

The RAG pipeline must not need to know whether the selected LLM is local or cloud-based.

## 5.3 Local System Architecture

The application should use the following local services:

1. React frontend.
2. FastAPI backend.
3. PostgreSQL with pgvector.
4. Ollama service or locally installed Ollama.
5. Local document storage.

Docker Compose should orchestrate the application services.

Ollama may run directly on the host operating system to simplify access to GPU resources.

## 5.4 Document Ingestion Workflow

1. User uploads a PDF.
2. Backend validates the file.
3. Original PDF is stored locally.
4. PyMuPDF extracts text page by page.
5. Metadata is extracted.
6. Text is divided into chunks.
7. Embeddings are generated locally.
8. Chunks, metadata and embeddings are stored in PostgreSQL.
9. Document is associated with selected collections.
10. Processing status changes to completed.

## 5.5 Question Answering Workflow

1. User selects one or more document collections.
2. User submits a question.
3. Backend identifies documents belonging to the selected collections.
4. Question is converted into an embedding.
5. Retrieval module searches the selected documents.
6. Retrieved chunks are ranked and filtered.
7. Backend constructs the LLM prompt.
8. Selected local or cloud LLM generates an answer.
9. Backend attaches source metadata.
10. Frontend displays the answer and references.

If retrieval does not find sufficiently relevant evidence, the system must return an insufficient-evidence response rather than forcing an answer.

---

# 6. Database Structure

## collections

- id: UUID
- name: VARCHAR
- description: TEXT, nullable
- created_at: TIMESTAMP
- updated_at: TIMESTAMP

## documents

- id: UUID
- filename: VARCHAR
- title: VARCHAR
- authors: TEXT
- publication_year: INTEGER, nullable
- page_count: INTEGER
- file_path: VARCHAR
- file_hash: VARCHAR
- processing_status: VARCHAR
- created_at: TIMESTAMP

## collection_documents

- collection_id: UUID
- document_id: UUID
- added_at: TIMESTAMP

Use a composite primary key or equivalent uniqueness constraint to prevent duplicate relationships.

## document_pages

- id: UUID
- document_id: UUID
- pdf_page_number: INTEGER
- printed_page_number: VARCHAR, nullable
- extracted_text: TEXT

## document_chunks

- id: UUID
- document_id: UUID
- chunk_index: INTEGER
- text: TEXT
- embedding: VECTOR
- embedding_model: VARCHAR
- metadata: JSONB
- created_at: TIMESTAMP

## conversations

- id: UUID
- title: VARCHAR, nullable
- created_at: TIMESTAMP

## conversation_collections

- conversation_id: UUID
- collection_id: UUID

This records which collections are associated with a conversation.

## messages

- id: UUID
- conversation_id: UUID
- role: VARCHAR
- content: TEXT
- model_provider: VARCHAR, nullable
- model_name: VARCHAR, nullable
- created_at: TIMESTAMP

## message_sources

- id: UUID
- message_id: UUID
- chunk_id: UUID
- relevance_score: FLOAT

Database migrations must be managed using Alembic.

The schema must be refined during implementation.

---

# 7. API Design

## Collections

- POST /api/collections
  - Create a collection.
- GET /api/collections
  - List collections.
- GET /api/collections/{collection_id}
  - Retrieve collection details.
- PATCH /api/collections/{collection_id}
  - Update collection metadata.
- DELETE /api/collections/{collection_id}
  - Delete a collection.
- GET /api/collections/{collection_id}/documents
  - List documents in a collection.
- POST /api/collections/{collection_id}/documents/{document_id}
  - Add a document to a collection.
- DELETE /api/collections/{collection_id}/documents/{document_id}
  - Remove a document from a collection.

## Documents

- POST /api/documents
  - Upload a PDF.
- GET /api/documents
  - List documents.
- GET /api/documents/{document_id}
  - Retrieve document metadata.
- DELETE /api/documents/{document_id}
  - Permanently delete a document.
- POST /api/documents/{document_id}/reprocess
  - Reprocess a document.
- GET /api/documents/{document_id}/pages/{page_number}
  - Retrieve extracted page text.

## Chat

- POST /api/chat
  - Submit a question and receive an answer with sources.
- GET /api/conversations
  - List conversations.
- GET /api/conversations/{conversation_id}
  - Retrieve conversation history.

## Models

- GET /api/models
  - List configured and available models.
- GET /api/models/providers
  - List configured providers.
- GET /api/models/health
  - Check provider availability.

## Health

- GET /health
  - Check application health.

Example chat request:

```json
{
  "question": "How does LDA differ from KNN?",
  "collection_ids": ["collection-uuid"],
  "conversation_id": null,
  "model_provider": "ollama",
  "model_name": "configured-model"
}
```

Example chat response:

```json
{
  "answer": "LDA and KNN use different classification approaches...",
  "insufficient_evidence": false,
  "sources": [
    {
      "document_id": "document-uuid",
      "document_title": "Hyperspectral Imaging",
      "pdf_page_number": 7,
      "printed_page_number": "123",
      "chunk_id": "chunk-uuid",
      "relevance_score": 0.91,
      "excerpt": "LDA is a supervised classification method..."
    }
  ],
  "model": {
    "provider": "ollama",
    "name": "configured-model"
  }
}
```

These values are illustrative.

---

# 8. User Interface Requirements

The initial interface should contain four main areas.

## 8.1 Collection Management

- List collections.
- Create and rename collections.
- Delete collections.
- Display the number of documents in each collection.

## 8.2 Document Library

- Display uploaded documents.
- Display metadata.
- Display processing status.
- Upload and delete documents.
- Assign documents to collections.
- Display collection membership.

## 8.3 Chat

- Question input.
- Conversation history.
- Collection selection.
- Model selection.
- Answer display.
- Source references.
- Loading and error states.

## 8.4 Source Viewer

- Display the original PDF.
- Navigate to the referenced page.
- Display the supporting text passage.
- Highlight relevant text where technically feasible.

The active collection and LLM provider must be visible to the user.

The UI should remain focused on document research rather than decorative functionality.

---

# 9. Repository Structure

```text
research-assistant/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── database/
│   │   ├── collections/
│   │   ├── documents/
│   │   ├── embeddings/
│   │   ├── retrieval/
│   │   ├── llm/
│   │   ├── chat/
│   │   └── evaluation/
│   │
│   ├── tests/
│   ├── pyproject.toml
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── evaluation/
│   ├── datasets/
│   ├── results/
│   └── README.md
│
├── storage/
│   └── .gitkeep
│
├── docs/
│   ├── architecture.md
│   ├── database-schema.md
│   └── decisions/
│
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

The storage directory must be excluded from Git.

Actual documents, API credentials and generated model data must never be committed to the repository.

---

# 10. Development Phases

## Phase 1: Project Foundation

- Initialize Git repository.
- Configure Python environment.
- Create FastAPI application.
- Configure PostgreSQL and pgvector.
- Create Docker Compose environment.
- Configure local storage.
- Implement health endpoint.
- Configure pytest and GitHub Actions.
- Create initial database migrations.

Deliverable: Local application infrastructure running.

## Phase 2: Collections and Document Ingestion

- Implement collection management.
- Implement PDF upload.
- Extract page-level text.
- Extract metadata.
- Implement chunking.
- Store documents, pages and chunks.
- Implement collection-document relationships.
- Add unit tests.

Deliverable: Users can organize and process articles in multiple collections.

## Phase 3: Embeddings and Retrieval

- Integrate Sentence Transformers.
- Generate embeddings locally.
- Store vectors in pgvector.
- Implement semantic search.
- Implement collection filtering.
- Test retrieval results.

Deliverable: Relevant passages can be retrieved from selected collections.

## Phase 4: LLM Integration

- Implement LLMProvider interface.
- Integrate Ollama.
- Implement cloud LLM adapter.
- Add model selection.
- Design grounding-focused prompts.
- Generate answers from retrieved context.
- Attach source metadata in the backend.
- Handle insufficient evidence.
- Add provider error handling.

Deliverable: Working RAG question-answering backend with local and cloud model support.

## Phase 5: Frontend

- Implement collection management.
- Implement document library.
- Implement upload interface.
- Implement chat interface.
- Implement model selection.
- Display answers and references.
- Implement PDF page navigation.
- Add loading and error states.

Deliverable: Usable end-to-end application.

## Phase 6: Evaluation

- Create manually verified evaluation dataset.
- Implement retrieval metrics.
- Implement answer evaluation.
- Measure citation correctness.
- Compare local and cloud models.
- Record experiment configurations.
- Document results and limitations.

Deliverable: Reproducible RAG evaluation framework.

## Phase 7: Portfolio Release

- Improve documentation.
- Add architecture diagrams.
- ~~Add screenshots or demonstration video.~~ Removed from scope (2026-11-25):
  the release ships without demo media; the README's sample papers and example
  questions let a developer see the system working within minutes instead.
- Document installation and configuration.
- Document model requirements.
- Document privacy and cloud processing behavior.
- Publish a stable release.

Deliverable: Complete and demonstrable GitHub portfolio project.

---

# 11. Evaluation and Testing

## 11.1 Unit Tests

Test at least:

- PDF text extraction.
- Page number preservation.
- Chunking.
- Metadata handling.
- Embedding interfaces.
- Collection filtering.
- Retrieval.
- Citation construction.
- Document deletion.
- Collection deletion.
- LLM provider interfaces.
- Error handling.

## 11.2 Integration Tests

Test complete workflows:

- Create collection → upload PDF → process document.
- Upload one document → add it to multiple collections.
- Submit question → retrieve chunks → generate answer → display sources.
- Select multiple collections → verify search scope.
- Delete collection → verify documents remain available.
- Delete document → verify chunks and embeddings are removed.
- Switch LLM provider → verify that the RAG pipeline remains functional.

Use test databases and mocked LLM responses where appropriate.

## 11.3 RAG Evaluation Dataset

Create approximately 30–50 manually verified questions.

Each test case should contain:

- Question.
- Expected answer.
- Relevant document.
- Relevant PDF page.
- Relevant source passage.

## 11.4 Evaluation Metrics

Retrieval:

- Recall@k
- Precision@k
- Mean Reciprocal Rank (MRR)

Generation:

- Answer correctness.
- Faithfulness to retrieved context.
- Answer relevance.
- Citation correctness.

Operational measurements:

- Retrieval latency.
- Total response latency.
- Token usage where available.
- Model provider and model identification.

Automated LLM-based evaluation must not be treated as ground truth. Important results should be checked manually.

---

# 12. MVP Acceptance Criteria

The MVP is complete when:

1. The application runs locally using Docker Compose.
2. A user can create multiple document collections.
3. A user can upload a scientific PDF.
4. The system extracts and stores text with page metadata.
5. A document can belong to multiple collections.
6. The system generates embeddings and indexes documents.
7. A user can select one or more collections for a question.
8. The system retrieves relevant passages.
9. The user can choose between configured local and cloud LLM providers.
10. The LLM generates answers grounded in retrieved passages.
11. Answers contain valid document and PDF page references.
12. The user can inspect original supporting text.
13. The system indicates when sufficient evidence cannot be found.
14. Automated tests cover critical processing and retrieval workflows.
15. The project contains installation instructions and technical documentation.

---

# 13. Future Enhancements

These features are outside the initial MVP:

- OCR for scanned documents.
- Scientific table extraction.
- Figure and image understanding.
- Hybrid retrieval.
- Reranking.
- Advanced metadata extraction.
- Multiple embedding model comparisons.
- Multiple LLM comparisons.
- Experiment tracking with MLflow.
- RAG evaluation dashboard.
- Exporting answers with academic references.
- Multi-user authentication.
- Additional document formats.

---

# 14. Development Principles

1. Build a working vertical slice before adding advanced features.
2. Prefer simple and maintainable solutions.
3. Keep LLM and embedding providers replaceable.
4. Preserve page-level metadata throughout the pipeline.
5. Never allow the LLM to invent source references.
6. Keep local processing as the default.
7. Make cloud processing explicit and configurable.
8. Write tests for critical functionality.
9. Keep retrieval and generation independently testable.
10. Document important architectural decisions.
11. Keep the application runnable locally through Docker Compose.

**Primary success criterion:** Research Assistant must provide useful answers that can be independently verified against the original scientific articles and their exact PDF pages, while allowing the user to control which document collections and language models are used.
