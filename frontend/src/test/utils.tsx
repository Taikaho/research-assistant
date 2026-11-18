import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router";
import { vi } from "vitest";

import { AppRoutes } from "../App";
import type {
  Collection,
  ConversationDetail,
  DocumentInfo,
  Message,
  ModelsResponse,
  SourceReference,
} from "../api/types";
import { ModelSettingsProvider } from "../lib/modelSettings";

export function renderWithProviders(ui: ReactElement, { route = "/" }: { route?: string } = {}) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ModelSettingsProvider>
        <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
      </ModelSettingsProvider>
    </QueryClientProvider>,
  );
}

export function renderApp(route: string) {
  return renderWithProviders(<AppRoutes />, { route });
}

type Handler = (init: RequestInit | undefined, url: URL) => unknown;

export interface MockRoute {
  method?: string;
  path: string | RegExp;
  status?: number;
  body?: unknown;
  handler?: Handler;
}

/**
 * Replace global fetch with a routing table. Unmatched requests fail the test
 * loudly. Returns the mock so tests can inspect the calls.
 */
export function mockFetch(routes: MockRoute[]) {
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), "http://localhost");
    const method = (init?.method ?? "GET").toUpperCase();
    const route = routes.find(
      (r) =>
        (r.method ?? "GET").toUpperCase() === method &&
        (typeof r.path === "string" ? url.pathname === r.path : r.path.test(url.pathname)),
    );
    if (!route) throw new Error(`Unmocked request: ${method} ${url.pathname}`);
    const body = route.handler ? await route.handler(init, url) : route.body;
    const status = route.status ?? (body === undefined ? 204 : 200);
    return new Response(status === 204 ? null : JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

export const collection = (overrides: Partial<Collection> = {}): Collection => ({
  id: "col-1",
  name: "Spectroscopy",
  description: null,
  created_at: "2026-11-01T10:00:00Z",
  updated_at: "2026-11-01T10:00:00Z",
  document_count: 1,
  ...overrides,
});

export const documentInfo = (overrides: Partial<DocumentInfo> = {}): DocumentInfo => ({
  id: "doc-1",
  filename: "paper.pdf",
  title: "Hyperspectral Imaging",
  authors: "A. Author",
  publication_year: 2021,
  page_count: 12,
  file_hash: "abc",
  processing_status: "completed",
  created_at: "2026-11-01T10:00:00Z",
  collection_ids: ["col-1"],
  ...overrides,
});

export const source = (overrides: Partial<SourceReference> = {}): SourceReference => ({
  citation_number: 1,
  cited: true,
  document_id: "doc-1",
  document_title: "Hyperspectral Imaging",
  filename: "paper.pdf",
  chunk_id: "chunk-1",
  pdf_page_numbers: [6],
  printed_page_numbers: [null],
  relevance_score: 0.71,
  excerpt: "LDA is a supervised classification method.",
  ...overrides,
});

export const message = (overrides: Partial<Message> = {}): Message => ({
  id: "msg-2",
  role: "assistant",
  content: "LDA is supervised [1].",
  created_at: "2026-11-01T10:00:00Z",
  model_provider: "ollama",
  model_name: "llama3.2:3b",
  insufficient_evidence: false,
  generation_metadata: { provider_is_local: true },
  sources: [source()],
  ...overrides,
});

export const conversation = (overrides: Partial<ConversationDetail> = {}): ConversationDetail => ({
  id: "conv-1",
  title: "How does LDA work?",
  created_at: "2026-11-01T10:00:00Z",
  collection_ids: ["col-1"],
  messages: [
    { ...message({ id: "msg-1", role: "user", content: "How does LDA work?", sources: [] }) },
    message(),
  ],
  ...overrides,
});

export const models = (overrides: Partial<ModelsResponse> = {}): ModelsResponse => ({
  default_provider: "ollama",
  local_only: false,
  providers: [
    {
      name: "ollama",
      is_local: true,
      default_model: null,
      is_default: true,
      available: true,
      models: ["llama3.2:3b", "qwen2.5:7b"],
      error: null,
    },
    {
      name: "openai",
      is_local: false,
      default_model: "gpt-test",
      is_default: false,
      available: true,
      models: ["gpt-test"],
      error: null,
    },
  ],
  ...overrides,
});
