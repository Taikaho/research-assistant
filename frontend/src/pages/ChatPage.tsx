import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router";

import { api } from "../api/client";
import { keys, useCollections, useConversation, useModels } from "../api/hooks";
import type { ChatRequest, ChatResponse, SourceReference } from "../api/types";
import { ConversationList } from "../components/chat/ConversationList";
import { MessageView } from "../components/chat/MessageView";
import { ModelSelector } from "../components/chat/ModelSelector";
import { ScopeSelector } from "../components/chat/ScopeSelector";
import { ErrorBox, Notice, Spinner } from "../components/common";
import { SourceViewer } from "../components/SourceViewer";
import { resolveModel, useModelSettings } from "../lib/modelSettings";

const LAST_SCOPE_KEY = "research-assistant.last-scope";

function loadLastScope(): string[] {
  try {
    const raw = localStorage.getItem(LAST_SCOPE_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((v): v is string => typeof v === "string") : [];
  } catch {
    return [];
  }
}

function saveLastScope(ids: string[]): void {
  try {
    localStorage.setItem(LAST_SCOPE_KEY, JSON.stringify(ids));
  } catch {
    // Not fatal.
  }
}

function Composer({
  disabledReason,
  pending,
  onSend,
  question,
  setQuestion,
}: {
  disabledReason: string | null;
  pending: boolean;
  onSend: (question: string) => void;
  question: string;
  setQuestion: (value: string) => void;
}) {
  const canSend = !disabledReason && !pending && question.trim().length > 0;
  const submit = (event?: FormEvent) => {
    event?.preventDefault();
    if (canSend) onSend(question.trim());
  };
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <form className="composer" onSubmit={submit}>
      <textarea
        aria-label="Question"
        placeholder="Ask a question about the selected documents… (Enter to send, Shift+Enter for a new line)"
        value={question}
        maxLength={4000}
        rows={3}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={onKeyDown}
      />
      <div className="composer-actions">
        {disabledReason && <span className="text-warning small">{disabledReason}</span>}
        <button type="submit" className="btn btn-primary" disabled={!canSend}>
          {pending ? "Answering…" : "Ask"}
        </button>
      </div>
    </form>
  );
}

export function ChatPage() {
  const { conversationId } = useParams();
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const collections = useCollections();
  const conversation = useConversation(conversationId);
  const models = useModels();
  const { settings } = useModelSettings();

  const isNew = !conversationId;
  const [newScope, setNewScope] = useState<string[]>(() => {
    const fromUrl = params.get("collections");
    return fromUrl ? fromUrl.split(",").filter(Boolean) : loadLastScope();
  });
  const [scopeOverride, setScopeOverride] = useState<string[] | null>(null);
  const [question, setQuestion] = useState("");
  const [viewerSource, setViewerSource] = useState<SourceReference | null>(null);
  const [lastResponse, setLastResponse] = useState<ChatResponse | null>(null);

  const currentId = useRef(conversationId);
  currentId.current = conversationId;
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => setScopeOverride(null), [conversationId]);

  const scope = isNew ? newScope : (scopeOverride ?? conversation.data?.collection_ids ?? []);
  const validScope = scope.filter((id) => collections.data?.some((c) => c.id === id));
  const setScope = (ids: string[]) => {
    if (isNew) {
      setNewScope(ids);
      saveLastScope(ids);
    } else {
      setScopeOverride(ids);
    }
  };

  const send = useMutation({
    mutationFn: (request: ChatRequest) => api.chat(request),
    onSuccess: async (response, request) => {
      setLastResponse(response);
      setScopeOverride(null);
      if (!request.conversation_id) {
        await queryClient.prefetchQuery({
          queryKey: keys.conversation(response.conversation_id),
          queryFn: () => api.getConversation(response.conversation_id),
        });
      }
      // Prefix match: refreshes the conversation list and the open conversation.
      await queryClient.invalidateQueries({ queryKey: keys.conversations });
      if (!request.conversation_id && currentId.current === undefined) {
        await navigate(`/chat/${response.conversation_id}`);
      }
    },
    onError: (_error, request) => {
      setQuestion((current) => current || request.question);
    },
  });

  const onSend = (text: string) => {
    const { provider, model } = resolveModel(settings, models.data);
    setQuestion("");
    send.mutate({
      question: text,
      conversation_id: conversationId ?? null,
      // For an existing conversation the stored scope is used unless it was changed.
      collection_ids: isNew || scopeOverride ? validScope : undefined,
      model_provider: provider?.name ?? null,
      model_name: model,
      temperature: settings.temperature,
      max_tokens: settings.maxTokens,
      top_k: settings.topK,
    });
  };

  const messages = conversation.data?.messages ?? [];
  const sendTarget = send.variables?.conversation_id ?? undefined;
  const sendIsHere = send.variables !== undefined && sendTarget === conversationId;

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [messages.length, send.isPending]);

  const closeViewer = useCallback(() => setViewerSource(null), []);

  const disabledReason = collections.isPending
    ? "Loading collections…"
    : validScope.length === 0
      ? "Select at least one collection to search."
      : null;

  const responseHere = lastResponse && lastResponse.conversation_id === conversationId ? lastResponse : null;

  return (
    <div className="chat-layout">
      <aside className="chat-sidebar">
        <ConversationList activeId={conversationId} />
        <section className="sidebar-section">
          <h2>Model</h2>
          <ModelSelector />
        </section>
      </aside>

      <section className="chat-main">
        <div className="chat-header">
          <h1>{isNew ? "New conversation" : (conversation.data?.title ?? "Conversation")}</h1>
          {collections.isError ? (
            <ErrorBox error={collections.error} title="Could not load collections." />
          ) : (
            <ScopeSelector
              key={conversationId ?? "new"}
              collections={collections.data ?? []}
              selected={scope}
              onChange={setScope}
              startOpen={isNew && validScope.length === 0}
              disabled={send.isPending}
            />
          )}
          {scopeOverride && <p className="muted small">The new scope applies from the next question.</p>}
        </div>

        <div className="messages" aria-live="polite">
          {isNew && !send.isPending && (
            <div className="intro">
              <p>
                Ask a question about the documents in the selected collections. Answers are
                generated only from retrieved passages, and every citation links to the exact PDF
                page it came from.
              </p>
            </div>
          )}
          {conversation.isPending && !isNew && <Spinner label="Loading conversation…" />}
          {conversation.isError && (
            <ErrorBox
              error={conversation.error}
              title="Could not load this conversation."
              onRetry={() => conversation.refetch()}
            />
          )}
          {messages.map((m) => (
            <MessageView key={m.id} message={m} onOpenSource={setViewerSource} />
          ))}
          {responseHere && responseHere.unindexed_document_count > 0 && (
            <Notice kind="warning">
              {responseHere.unindexed_document_count} document(s) in this scope are not indexed with
              the current embedding model and were not searched. Re-index them in the Library.
            </Notice>
          )}
          {sendIsHere && send.isPending && (
            <>
              <div className="message message-user">
                <div className="bubble">{send.variables.question}</div>
              </div>
              <div className="message message-assistant pending">
                <Spinner label="Searching passages and generating an answer…" />
              </div>
            </>
          )}
          {sendIsHere && send.isError && (
            <ErrorBox
              error={send.error}
              title="The question could not be answered."
              onRetry={() => {
                setQuestion("");
                send.mutate(send.variables);
              }}
            />
          )}
          <div ref={endRef} />
        </div>

        <Composer
          disabledReason={disabledReason}
          pending={send.isPending}
          onSend={onSend}
          question={question}
          setQuestion={setQuestion}
        />
      </section>

      {viewerSource && <SourceViewer source={viewerSource} onClose={closeViewer} />}
    </div>
  );
}
