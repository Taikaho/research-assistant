import type {
  ChatRequest,
  ChatResponse,
  Collection,
  ConversationDetail,
  ConversationSummary,
  DocumentInfo,
  IndexStatus,
  ModelsResponse,
  PageText,
} from "./types";

export const API_BASE = "/api/v1";

/** An HTTP error from the backend, with FastAPI's `detail` turned into readable text. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

interface ValidationIssue {
  loc?: (string | number)[];
  msg?: string;
}

/** Turn FastAPI's `detail` (a string, or a list of validation issues) into one message. */
export function formatDetail(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const parts = (detail as ValidationIssue[])
      .map((issue) => {
        const field = issue.loc?.filter((p) => p !== "body").join(".");
        return field ? `${field}: ${issue.msg ?? "invalid"}` : issue.msg;
      })
      .filter(Boolean);
    return parts.length ? parts.join("; ") : null;
  }
  return null;
}

const STATUS_HINTS: Record<number, string> = {
  502: "The language model provider returned an error.",
  503: "A required service is unavailable.",
  504: "The request timed out.",
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(0, "Cannot reach the backend. Is it running?");
  }

  if (!response.ok) {
    let message: string | null = null;
    try {
      message = formatDetail(((await response.json()) as { detail?: unknown }).detail);
    } catch {
      // Not JSON (e.g. a proxy error page).
    }
    throw new ApiError(
      response.status,
      message ?? STATUS_HINTS[response.status] ?? `Request failed (HTTP ${response.status}).`,
    );
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function json(method: string, body?: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  };
}

export const api = {
  // Collections
  listCollections: () => request<Collection[]>("/collections"),
  createCollection: (data: { name: string; description?: string | null }) =>
    request<Collection>("/collections", json("POST", data)),
  updateCollection: (id: string, data: { name?: string; description?: string | null }) =>
    request<Collection>(`/collections/${id}`, json("PATCH", data)),
  deleteCollection: (id: string) => request<void>(`/collections/${id}`, { method: "DELETE" }),
  addToCollection: (collectionId: string, documentId: string) =>
    request<void>(`/collections/${collectionId}/documents/${documentId}`, { method: "PUT" }),
  removeFromCollection: (collectionId: string, documentId: string) =>
    request<void>(`/collections/${collectionId}/documents/${documentId}`, { method: "DELETE" }),

  // Documents
  listDocuments: () => request<DocumentInfo[]>("/documents"),
  uploadDocument: (file: File, collectionIds: string[]) => {
    const form = new FormData();
    form.append("file", file);
    for (const id of collectionIds) form.append("collection_ids", id);
    return request<DocumentInfo>("/documents", { method: "POST", body: form });
  },
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" }),
  reprocessDocument: (id: string) =>
    request<DocumentInfo>(`/documents/${id}/reprocess`, { method: "POST" }),
  getPage: (documentId: string, pdfPage: number) =>
    request<PageText>(`/documents/${documentId}/pages/${pdfPage}`),

  // Embeddings
  indexStatus: () => request<IndexStatus>("/embeddings/status"),
  reindex: () => request<unknown>("/embeddings/reindex", { method: "POST" }),

  // Chat
  chat: (body: ChatRequest) => request<ChatResponse>("/chat", json("POST", body)),
  listConversations: () => request<ConversationSummary[]>("/conversations"),
  getConversation: (id: string) => request<ConversationDetail>(`/conversations/${id}`),
  deleteConversation: (id: string) =>
    request<void>(`/conversations/${id}`, { method: "DELETE" }),

  // Models
  listModels: () => request<ModelsResponse>("/models"),
};

/** URL of the original PDF opened at a zero-based page index (browser PDF viewers are 1-based). */
export function pdfUrl(documentId: string, pdfPage?: number): string {
  const base = `${API_BASE}/documents/${documentId}/file`;
  return pdfPage === undefined ? base : `${base}#page=${pdfPage + 1}`;
}

export function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}
