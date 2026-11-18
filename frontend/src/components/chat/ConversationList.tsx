import { useMutation, useQueryClient } from "@tanstack/react-query";
import { NavLink, useNavigate } from "react-router";

import { api } from "../../api/client";
import { keys, useConversations } from "../../api/hooks";
import type { ConversationSummary } from "../../api/types";
import { formatDate } from "../../lib/format";
import { ErrorBox, Spinner } from "../common";

function ConversationItem({ conversation, active }: { conversation: ConversationSummary; active: boolean }) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const remove = useMutation({
    mutationFn: () => api.deleteConversation(conversation.id),
    onSuccess: async () => {
      if (active) await navigate("/chat");
      queryClient.removeQueries({ queryKey: keys.conversation(conversation.id) });
      await queryClient.invalidateQueries({ queryKey: keys.conversations, exact: true });
    },
  });

  return (
    <li className="conversation-item">
      <NavLink to={`/chat/${conversation.id}`} title={conversation.title ?? undefined}>
        <span className="conversation-title">{conversation.title ?? "Untitled conversation"}</span>
        <span className="muted small">{formatDate(conversation.created_at)}</span>
      </NavLink>
      <button
        type="button"
        className="icon-btn"
        aria-label={`Delete conversation ${conversation.title ?? ""}`}
        disabled={remove.isPending}
        onClick={() => {
          if (window.confirm("Delete this conversation and its history?")) remove.mutate();
        }}
      >
        ×
      </button>
    </li>
  );
}

export function ConversationList({ activeId }: { activeId?: string }) {
  const conversations = useConversations();
  return (
    <div className="conversation-list">
      <NavLink to="/chat" end className="btn btn-primary btn-block">
        + New conversation
      </NavLink>
      {conversations.isPending && <Spinner label="Loading…" />}
      {conversations.isError && (
        <ErrorBox error={conversations.error} onRetry={() => conversations.refetch()} />
      )}
      {conversations.data?.length === 0 && <p className="muted small">No conversations yet.</p>}
      <ul>
        {conversations.data?.map((c) => (
          <ConversationItem key={c.id} conversation={c} active={c.id === activeId} />
        ))}
      </ul>
    </div>
  );
}
