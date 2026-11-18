# Database Schema

Implemented in `backend/migrations/versions/0001_initial_schema.py`.

## Tables

### collections
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | gen_random_uuid() |
| name | VARCHAR(255) | |
| description | TEXT | nullable |
| created_at | TIMESTAMPTZ | server default now() |
| updated_at | TIMESTAMPTZ | maintained by trigger |

### documents
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| filename | VARCHAR(512) | original upload filename |
| title | VARCHAR(512) | extracted or inferred |
| authors | TEXT | nullable |
| publication_year | INTEGER | nullable |
| page_count | INTEGER | |
| file_path | VARCHAR(1024) | path under storage/ |
| file_hash | VARCHAR(64) | SHA-256 for dedup detection |
| processing_status | VARCHAR(50) | pending/processing/completed/failed |
| created_at | TIMESTAMPTZ | |

### collection_documents  (many-to-many)
| Column | Type | Notes |
|---|---|---|
| collection_id | UUID FK → collections | CASCADE delete |
| document_id | UUID FK → documents | RESTRICT delete |
| added_at | TIMESTAMPTZ | |

Composite PK on (collection_id, document_id). Deleting a collection does not
delete the documents, only the relationship rows.

### document_pages
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| document_id | UUID FK → documents | CASCADE |
| pdf_page_number | INTEGER | zero-based PDF index (authoritative) |
| printed_page_number | VARCHAR(50) | nullable; as printed in document |
| extracted_text | TEXT | |

### embedding_models
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| name | VARCHAR(255) | model identifier |
| version | VARCHAR(100) | nullable; commit hash or semver |
| dimensions | INTEGER | actual output dimension (programmatic) |
| is_active | BOOLEAN | |
| created_at | TIMESTAMPTZ | |

Unique constraint on (name, version).

### document_chunks
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| document_id | UUID FK → documents | CASCADE |
| chunk_index | INTEGER | position within document |
| text | TEXT | chunk content |
| embedding | VECTOR | nullable; dimensionless, dimension given by the model |
| embedding_model_id | UUID FK → embedding_models | nullable; SET NULL; vector space of `embedding` |
| page_numbers | JSONB | list of PDF page numbers covered |
| chunk_metadata | JSONB | chunking config, overlap, etc. |
| created_at | TIMESTAMPTZ | |

**Note:** `embedding` (migration 0003) has no fixed dimension, so changing the
embedding model only requires re-indexing. Vectors are only ever compared within
one `embedding_model_id`. No ANN index yet; exact search is used.

### conversations
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| title | VARCHAR(512) | nullable; auto-generated or user-set |
| created_at | TIMESTAMPTZ | |

### conversation_collections  (many-to-many)
| Column | Type | Notes |
|---|---|---|
| conversation_id | UUID FK → conversations | CASCADE |
| collection_id | UUID FK → collections | CASCADE |

### messages
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| conversation_id | UUID FK → conversations | CASCADE |
| role | VARCHAR(50) | user / assistant / system |
| content | TEXT | |
| model_provider | VARCHAR(100) | nullable |
| model_name | VARCHAR(255) | nullable |
| insufficient_evidence | BOOLEAN | assistant answers that report missing evidence (0004) |
| generation_metadata | JSONB | nullable; prompt version, temperature, max_tokens, top_k, relevance gate, embedding model, token usage, latencies (0004) |
| created_at | TIMESTAMPTZ | |

### message_sources
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| message_id | UUID FK → messages | CASCADE |
| chunk_id | UUID FK → document_chunks | CASCADE |
| relevance_score | FLOAT | |
| citation_number | INTEGER | nullable; the `[n]` marker used for this passage in the answer (0004) |
| cited | BOOLEAN | true when the answer text cites the passage (0004) |

Sources are assembled by the backend from retrieval metadata.
The LLM never generates or validates source references.
