import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ChatRequest, ChatResponse } from "../api/types";
import {
  collection,
  conversation,
  message,
  mockFetch,
  models,
  renderApp,
  source,
  type MockRoute,
} from "../test/utils";

afterEach(() => vi.unstubAllGlobals());

const baseRoutes = (): MockRoute[] => [
  { path: "/api/v1/collections", body: [collection(), collection({ id: "col-2", name: "Botany", document_count: 0 })] },
  { path: "/api/v1/conversations", body: [] },
  { path: "/api/v1/models", body: models() },
];

const chatResponse = (overrides: Partial<ChatResponse> = {}): ChatResponse => ({
  conversation_id: "conv-1",
  message_id: "msg-2",
  answer: "LDA is supervised [1].",
  insufficient_evidence: false,
  insufficient_evidence_reason: null,
  sources: [source()],
  model: { provider: "ollama", name: "llama3.2:3b", is_local: true },
  cloud_processing: false,
  usage: null,
  collection_ids: ["col-1"],
  unindexed_document_count: 0,
  ...overrides,
});

describe("ChatPage", () => {
  it("requires a collection before a question can be sent", async () => {
    mockFetch(baseRoutes());
    renderApp("/chat");
    expect(await screen.findByText("Select at least one collection to search.")).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText("Question"), "Anything?");
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
  });

  it("sends a new question with scope and model, then shows the answer with sources", async () => {
    let sent: ChatRequest | undefined;
    let resolveChat: (value: ChatResponse) => void = () => {};
    mockFetch([
      ...baseRoutes(),
      {
        method: "POST",
        path: "/api/v1/chat",
        handler: (init) => {
          sent = JSON.parse(String(init?.body)) as ChatRequest;
          return new Promise<ChatResponse>((resolve) => (resolveChat = resolve));
        },
      },
      { path: "/api/v1/conversations/conv-1", body: conversation() },
    ]);
    renderApp("/chat?collections=col-1");

    // The active scope and the active model are visible.
    const scope = await screen.findByLabelText("Search scope");
    expect(await within(scope).findByText("Spectroscopy")).toBeInTheDocument();
    expect(await screen.findByTestId("active-model")).toHaveTextContent("ollama · llama3.2:3b");

    await userEvent.type(screen.getByLabelText("Question"), "How does LDA work?{Enter}");

    // Loading state while the backend answers.
    expect(await screen.findByText("Searching passages and generating an answer…")).toBeInTheDocument();
    expect(sent).toMatchObject({
      question: "How does LDA work?",
      conversation_id: null,
      collection_ids: ["col-1"],
      model_provider: "ollama",
      model_name: "llama3.2:3b",
    });

    resolveChat(chatResponse());
    const citation = await screen.findByRole("button", { name: /Source 1: Hyperspectral Imaging, PDF p\. 7/ });
    expect(citation).toBeInTheDocument();
    expect(screen.getByText("1 cited · 1 passages retrieved")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "How does LDA work?" })).toBeInTheDocument();
  });

  it("opens the source viewer at the cited PDF page", async () => {
    mockFetch([
      ...baseRoutes(),
      { path: "/api/v1/conversations/conv-1", body: conversation() },
      {
        path: "/api/v1/documents/doc-1/pages/6",
        body: {
          document_id: "doc-1",
          pdf_page_number: 6,
          printed_page_number: null,
          text: "Before. LDA is a supervised classification method. After.",
        },
      },
    ]);
    renderApp("/chat/conv-1");

    await userEvent.click(await screen.findByRole("button", { name: /Source 1:/ }));
    const dialog = await screen.findByRole("dialog");
    const frame = within(dialog).getByTitle(/PDF p\. 7/);
    expect(frame).toHaveAttribute("src", "/api/v1/documents/doc-1/file#page=7");

    const mark = await within(dialog).findByText("LDA is a supervised classification method.");
    expect(mark.tagName).toBe("MARK");

    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("marks insufficient-evidence answers and model-free responses", async () => {
    mockFetch([
      ...baseRoutes(),
      {
        path: "/api/v1/conversations/conv-1",
        body: conversation({
          messages: [
            message({
              content: "The selected documents do not contain enough information.",
              insufficient_evidence: true,
              model_name: null,
              model_provider: null,
              generation_metadata: {},
              sources: [],
            }),
          ],
        }),
      },
    ]);
    renderApp("/chat/conv-1");
    expect(
      await screen.findByText("Insufficient evidence — no relevant passages in the selected collections"),
    ).toBeInTheDocument();
    expect(screen.getByText("Model not called")).toBeInTheDocument();
  });

  it("shows backend errors and restores the question for a retry", async () => {
    mockFetch([
      ...baseRoutes(),
      { path: "/api/v1/conversations/conv-1", body: conversation() },
      {
        method: "POST",
        path: "/api/v1/chat",
        status: 503,
        body: { detail: "Cannot connect to Ollama at http://localhost:11434." },
      },
    ]);
    renderApp("/chat/conv-1");
    await screen.findByText("LDA is supervised");
    await userEvent.type(screen.getByLabelText("Question"), "Follow-up?{Enter}");

    expect(await screen.findByText(/Cannot connect to Ollama/)).toBeInTheDocument();
    expect(screen.getByLabelText("Question")).toHaveValue("Follow-up?");
  });

  it("keeps the stored scope for follow-ups unless it is changed", async () => {
    const bodies: ChatRequest[] = [];
    mockFetch([
      ...baseRoutes(),
      { path: "/api/v1/conversations/conv-1", body: conversation() },
      {
        method: "POST",
        path: "/api/v1/chat",
        handler: (init) => {
          bodies.push(JSON.parse(String(init?.body)) as ChatRequest);
          return chatResponse();
        },
      },
    ]);
    renderApp("/chat/conv-1");
    await screen.findByText("LDA is supervised");

    await userEvent.type(screen.getByLabelText("Question"), "First?{Enter}");
    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]!.collection_ids).toBeUndefined();
    expect(bodies[0]!.conversation_id).toBe("conv-1");

    await userEvent.click(screen.getByRole("button", { name: "Change" }));
    await userEvent.click(screen.getByRole("checkbox", { name: /Botany/ }));
    await userEvent.type(screen.getByLabelText("Question"), "Second?{Enter}");
    await waitFor(() => expect(bodies).toHaveLength(2));
    expect(bodies[1]!.collection_ids).toEqual(["col-1", "col-2"]);
  });

  it("warns when a cloud provider is selected", async () => {
    mockFetch(baseRoutes());
    renderApp("/chat");
    const provider = await screen.findByRole("combobox", { name: "Provider" });
    await userEvent.selectOptions(provider, "openai");
    expect(await screen.findByText(/Cloud processing\./)).toBeInTheDocument();
    expect(screen.getByTestId("active-model")).toHaveTextContent("openai · gpt-test");
  });
});
