import { useMemo, useState } from "react";

import type { Message, SourceReference } from "../../api/types";
import { splitCitations } from "../../lib/citations";
import { formatScore, sourcePagesLabel, truncate } from "../../lib/format";
import { LocalityBadge } from "../common";

export type OpenSource = (source: SourceReference) => void;

export function AnswerText({
  text,
  sources,
  onOpenSource,
}: {
  text: string;
  sources: SourceReference[];
  onOpenSource: OpenSource;
}) {
  const byNumber = useMemo(() => new Map(sources.map((s) => [s.citation_number, s])), [sources]);
  const segments = useMemo(() => splitCitations(text, new Set(byNumber.keys())), [text, byNumber]);
  return (
    <div className="answer-text">
      {segments.map((segment, i) => {
        if (segment.kind === "text") return <span key={i}>{segment.text}</span>;
        const source = byNumber.get(segment.number)!;
        return (
          <button
            key={i}
            type="button"
            className="citation"
            title={`${source.document_title}, ${sourcePagesLabel(source)}`}
            aria-label={`Source ${segment.number}: ${source.document_title}, ${sourcePagesLabel(source)}`}
            onClick={() => onOpenSource(source)}
          >
            {segment.number}
          </button>
        );
      })}
    </div>
  );
}

function SourceItem({ source, onOpenSource }: { source: SourceReference; onOpenSource: OpenSource }) {
  return (
    <li className={`source-item${source.cited ? " cited" : ""}`}>
      <div className="source-head">
        <span className="source-number">[{source.citation_number}]</span>
        <span className="source-title">{source.document_title}</span>
        <span className="source-pages">{sourcePagesLabel(source)}</span>
        <span className="muted small" title="Cosine similarity between question and passage">
          relevance {formatScore(source.relevance_score)}
        </span>
        {!source.cited && <span className="badge badge-muted">not cited</span>}
      </div>
      <p className="source-excerpt">{truncate(source.excerpt, 260)}</p>
      <button type="button" className="btn btn-small" onClick={() => onOpenSource(source)}>
        View in PDF
      </button>
    </li>
  );
}

export function SourceList({
  sources,
  onOpenSource,
  expandedByDefault,
}: {
  sources: SourceReference[];
  onOpenSource: OpenSource;
  expandedByDefault: boolean;
}) {
  const cited = sources.filter((s) => s.cited);
  const [showAll, setShowAll] = useState(cited.length === 0 && expandedByDefault);
  if (sources.length === 0) return null;
  const visible = showAll ? sources : cited;

  return (
    <div className="sources">
      <div className="sources-head">
        <strong>Sources</strong>
        <span className="muted small">
          {cited.length} cited · {sources.length} passages retrieved
        </span>
        {cited.length < sources.length && (
          <button type="button" className="btn btn-small btn-link" onClick={() => setShowAll(!showAll)}>
            {showAll ? "Show cited only" : "Show all retrieved passages"}
          </button>
        )}
      </div>
      {visible.length > 0 && (
        <ol className="source-list">
          {visible.map((s) => (
            <SourceItem key={s.chunk_id} source={s} onOpenSource={onOpenSource} />
          ))}
        </ol>
      )}
    </div>
  );
}

function isLocal(message: Message): boolean | null {
  const value = message.generation_metadata?.["provider_is_local"];
  return typeof value === "boolean" ? value : null;
}

export function MessageView({ message, onOpenSource }: { message: Message; onOpenSource: OpenSource }) {
  if (message.role === "user") {
    return (
      <div className="message message-user">
        <div className="bubble">{message.content}</div>
      </div>
    );
  }

  const local = isLocal(message);
  const llmCalled = Boolean(message.model_name);
  const hasCitations = message.sources.some((s) => s.cited);
  return (
    <div className={`message message-assistant${message.insufficient_evidence ? " insufficient" : ""}`}>
      {message.insufficient_evidence && (
        <div className="insufficient-label">
          {llmCalled
            ? "Insufficient evidence — the model found no support in the passages"
            : "Insufficient evidence — no relevant passages in the selected collections"}
        </div>
      )}
      <AnswerText text={message.content} sources={message.sources} onOpenSource={onOpenSource} />
      {!message.insufficient_evidence && llmCalled && !hasCitations && (
        <p className="text-warning small">
          This answer cites no passage. Check it against the retrieved sources before relying on it.
        </p>
      )}
      <div className="message-meta small muted">
        {llmCalled ? (
          <>
            {local !== null && <LocalityBadge isLocal={local} />}
            <span>
              {message.model_provider} · {message.model_name}
            </span>
          </>
        ) : (
          <span>Model not called</span>
        )}
      </div>
      <SourceList
        sources={message.sources}
        onOpenSource={onOpenSource}
        expandedByDefault={!message.insufficient_evidence}
      />
    </div>
  );
}
