// Types mirroring the backend's Pydantic response schemas (backend/app/*/schemas.py).
// UUIDs and datetimes arrive as strings.

export interface Collection {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
  document_count: number;
}

export type ProcessingStatus = "pending" | "processing" | "completed" | "failed";

export interface DocumentInfo {
  id: string;
  filename: string;
  title: string;
  authors: string | null;
  publication_year: number | null;
  page_count: number;
  file_hash: string;
  processing_status: ProcessingStatus | string;
  created_at: string;
  collection_ids: string[];
}

export interface PageText {
  document_id: string;
  /** Zero-based PDF page index. */
  pdf_page_number: number;
  printed_page_number: string | null;
  text: string;
}

export interface SourceReference {
  citation_number: number;
  cited: boolean;
  document_id: string;
  document_title: string;
  filename: string;
  chunk_id: string;
  /** Zero-based PDF page indexes covered by the passage. */
  pdf_page_numbers: number[];
  printed_page_numbers: (string | null)[];
  relevance_score: number;
  excerpt: string;
}

export interface ModelInfo {
  provider: string;
  name: string;
  is_local: boolean;
}

export interface ChatRequest {
  question: string;
  collection_ids?: string[];
  conversation_id?: string | null;
  model_provider?: string | null;
  model_name?: string | null;
  temperature?: number | null;
  max_tokens?: number | null;
  top_k?: number | null;
}

export interface ChatResponse {
  conversation_id: string;
  message_id: string;
  answer: string;
  insufficient_evidence: boolean;
  insufficient_evidence_reason: "no_relevant_passages" | "model_reported" | null;
  sources: SourceReference[];
  model: ModelInfo | null;
  cloud_processing: boolean;
  usage: { prompt_tokens: number | null; completion_tokens: number | null } | null;
  collection_ids: string[];
  unindexed_document_count: number;
}

export interface ConversationSummary {
  id: string;
  title: string | null;
  created_at: string;
  collection_ids: string[];
  message_count: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "system" | string;
  content: string;
  created_at: string;
  model_provider: string | null;
  model_name: string | null;
  insufficient_evidence: boolean;
  generation_metadata: Record<string, unknown> | null;
  sources: SourceReference[];
}

export interface ConversationDetail {
  id: string;
  title: string | null;
  created_at: string;
  collection_ids: string[];
  messages: Message[];
}

export interface ProviderModels {
  name: string;
  is_local: boolean;
  default_model: string | null;
  is_default: boolean;
  available: boolean;
  models: string[];
  error: string | null;
}

export interface ModelsResponse {
  default_provider: string;
  local_only: boolean;
  providers: ProviderModels[];
}

export interface IndexStatus {
  configured_model_name: string;
  active_model: { name: string; version: string | null; dimensions: number } | null;
  total_chunks: number;
  indexed_chunks: number;
  reindex_required: boolean;
}
