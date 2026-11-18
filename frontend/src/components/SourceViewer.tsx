import { useEffect, useRef, useState } from "react";

import { pdfUrl } from "../api/client";
import { usePage } from "../api/hooks";
import type { SourceReference } from "../api/types";
import { formatScore, pageLabel, sourcePagesLabel } from "../lib/format";
import { findPassageRange } from "../lib/highlight";
import { ErrorBox, Spinner } from "./common";

function PageText({ source, page }: { source: SourceReference; page: number }) {
  const pageQuery = usePage(source.document_id, page);
  const markRef = useRef<HTMLElement>(null);

  useEffect(() => {
    markRef.current?.scrollIntoView?.({ block: "center" });
  }, [pageQuery.data]);

  if (pageQuery.isPending) return <Spinner label="Loading page text…" />;
  if (pageQuery.isError) return <ErrorBox error={pageQuery.error} title="Could not load page text." />;

  const text = pageQuery.data.text;
  if (!text.trim()) return <p className="muted">No text was extracted from this page.</p>;
  const range = findPassageRange(text, source.excerpt);
  return (
    <>
      {!range && (
        <p className="muted small">
          The passage could not be located on this page (the document may have been reprocessed).
        </p>
      )}
      <pre className="page-text">
        {range ? (
          <>
            {text.slice(0, range.start)}
            <mark ref={markRef}>{text.slice(range.start, range.end)}</mark>
            {text.slice(range.end)}
          </>
        ) : (
          text
        )}
      </pre>
    </>
  );
}

/**
 * Shows the original PDF at the referenced page next to the supporting passage.
 * The PDF is rendered by the browser's built-in viewer (`#page=N`); the passage is
 * highlighted in the page's extracted text, which is where it was cut from.
 */
export function SourceViewer({ source, onClose }: { source: SourceReference; onClose: () => void }) {
  const pages = source.pdf_page_numbers;
  const [page, setPage] = useState(pages[0] ?? 0);
  const [tab, setTab] = useState<"passage" | "page">("page");
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    setPage(source.pdf_page_numbers[0] ?? 0);
  }, [source]);

  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const printed = source.printed_page_numbers[pages.indexOf(page)] ?? null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal source-viewer"
        role="dialog"
        aria-modal="true"
        aria-label={`Source ${source.citation_number}: ${source.document_title}`}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="modal-header">
          <div>
            <h2>
              [{source.citation_number}] {source.document_title}
            </h2>
            <p className="muted small">
              {source.filename} · {sourcePagesLabel(source)} · relevance {formatScore(source.relevance_score)} ·{" "}
              {source.cited ? "cited in the answer" : "retrieved, not cited"}
            </p>
          </div>
          <button ref={closeRef} type="button" className="btn" onClick={onClose}>
            Close
          </button>
        </header>

        <div className="viewer-toolbar">
          <span className="muted small">Passage pages:</span>
          {pages.map((p, i) => (
            <button
              key={p}
              type="button"
              className={`btn btn-small${p === page ? " btn-primary" : ""}`}
              aria-pressed={p === page}
              onClick={() => setPage(p)}
            >
              {pageLabel(p, source.printed_page_numbers[i])}
            </button>
          ))}
          <a
            className="btn btn-small btn-link"
            href={pdfUrl(source.document_id, page)}
            target="_blank"
            rel="noreferrer"
          >
            Open PDF in new tab
          </a>
        </div>

        <div className="viewer-body">
          <iframe
            // Remount on page change: not every built-in PDF viewer reacts to a hash change.
            key={page}
            className="pdf-frame"
            title={`${source.document_title}, ${pageLabel(page, printed)}`}
            src={pdfUrl(source.document_id, page)}
          />
          <div className="viewer-text">
            <div className="tabs" role="tablist">
              <button
                type="button"
                role="tab"
                aria-selected={tab === "page"}
                className={tab === "page" ? "active" : ""}
                onClick={() => setTab("page")}
              >
                Page text ({pageLabel(page, printed)})
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={tab === "passage"}
                className={tab === "passage" ? "active" : ""}
                onClick={() => setTab("passage")}
              >
                Full passage
              </button>
            </div>
            {tab === "page" ? (
              <PageText source={source} page={page} />
            ) : (
              <pre className="page-text">{source.excerpt}</pre>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
