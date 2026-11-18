import { useMutation } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router";

import { api, pdfUrl } from "../api/client";
import {
  useCollections,
  useDocuments,
  useIndexStatus,
  useLibraryInvalidation,
  useMembershipMutation,
} from "../api/hooks";
import type { Collection, DocumentInfo } from "../api/types";
import { ErrorBox, Notice, Spinner, StatusBadge } from "../components/common";
import { UploadPanel } from "../components/UploadPanel";
import { formatDate } from "../lib/format";

const UNASSIGNED = "__none__";

function IndexStatusBanner() {
  const status = useIndexStatus();
  const invalidate = useLibraryInvalidation();
  const reindex = useMutation({ mutationFn: api.reindex, onSuccess: invalidate });

  if (!status.data?.reindex_required || status.data.total_chunks === 0) return null;
  const { indexed_chunks, total_chunks, configured_model_name } = status.data;
  return (
    <Notice kind="warning">
      <div>
        {indexed_chunks} of {total_chunks} passages are indexed with the configured embedding
        model <code>{configured_model_name}</code>. Unindexed documents are not searched.
      </div>
      <button
        type="button"
        className="btn btn-small"
        onClick={() => reindex.mutate()}
        disabled={reindex.isPending}
      >
        {reindex.isPending ? "Re-indexing… (may take minutes)" : "Re-index now"}
      </button>
      {reindex.isError && <ErrorBox error={reindex.error} />}
    </Notice>
  );
}

function MembershipCell({
  document,
  collections,
}: {
  document: DocumentInfo;
  collections: Collection[];
}) {
  const membership = useMembershipMutation();
  const byId = new Map(collections.map((c) => [c.id, c]));
  const available = collections.filter((c) => !document.collection_ids.includes(c.id));

  return (
    <div className="membership">
      {document.collection_ids.length === 0 && <span className="muted small">No collection</span>}
      {document.collection_ids.map((id) => (
        <span key={id} className="chip">
          {byId.get(id)?.name ?? "Unknown"}
          <button
            type="button"
            className="chip-remove"
            aria-label={`Remove from ${byId.get(id)?.name ?? "collection"}`}
            title="Remove from this collection (the document is kept)"
            disabled={membership.isPending}
            onClick={() =>
              membership.mutate({ collectionId: id, documentId: document.id, member: false })
            }
          >
            ×
          </button>
        </span>
      ))}
      {available.length > 0 && (
        <select
          className="select-small"
          aria-label={`Add ${document.title} to collection`}
          value=""
          disabled={membership.isPending}
          onChange={(e) => {
            if (e.target.value) {
              membership.mutate({
                collectionId: e.target.value,
                documentId: document.id,
                member: true,
              });
            }
          }}
        >
          <option value="">+ Add to…</option>
          {available.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
      )}
      {membership.isError && <ErrorBox error={membership.error} />}
    </div>
  );
}

function DocumentRow({ document, collections }: { document: DocumentInfo; collections: Collection[] }) {
  const invalidate = useLibraryInvalidation();
  const remove = useMutation({ mutationFn: () => api.deleteDocument(document.id), onSuccess: invalidate });
  const reprocess = useMutation({
    mutationFn: () => api.reprocessDocument(document.id),
    onSettled: invalidate,
  });
  const busy = remove.isPending || reprocess.isPending;

  const confirmDelete = () => {
    if (
      window.confirm(
        `Permanently delete "${document.title}"?\n\nThe PDF, its passages and embeddings are removed from every collection. Chat answers citing it lose those sources.`,
      )
    ) {
      remove.mutate();
    }
  };
  const confirmReprocess = () => {
    if (
      window.confirm(
        `Reprocess "${document.title}"?\n\nText is extracted, chunked and embedded again with the current settings. Earlier chat answers lose their links to this document's passages.`,
      )
    ) {
      reprocess.mutate();
    }
  };

  return (
    <tr>
      <td>
        <div className="doc-title">{document.title}</div>
        <div className="muted small">
          {document.filename}
          {document.authors && <> · {document.authors}</>}
          {document.publication_year && <> · {document.publication_year}</>}
        </div>
        {(remove.isError || reprocess.isError) && (
          <ErrorBox error={remove.error ?? reprocess.error} />
        )}
      </td>
      <td className="num">{document.page_count}</td>
      <td>
        {reprocess.isPending ? <Spinner label="Reprocessing…" /> : <StatusBadge status={document.processing_status} />}
      </td>
      <td>
        <MembershipCell document={document} collections={collections} />
      </td>
      <td className="small muted">{formatDate(document.created_at)}</td>
      <td>
        <div className="row-actions">
          <a className="btn btn-small" href={pdfUrl(document.id)} target="_blank" rel="noreferrer">
            Open PDF
          </a>
          <button type="button" className="btn btn-small" onClick={confirmReprocess} disabled={busy}>
            Reprocess
          </button>
          <button type="button" className="btn btn-small btn-danger" onClick={confirmDelete} disabled={busy}>
            {remove.isPending ? "Deleting…" : "Delete"}
          </button>
        </div>
      </td>
    </tr>
  );
}

export function LibraryPage() {
  const documents = useDocuments();
  const collections = useCollections();
  const [params, setParams] = useSearchParams();
  const [query, setQuery] = useState("");
  const filter = params.get("collection") ?? "";

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (documents.data ?? []).filter((d) => {
      if (filter === UNASSIGNED && d.collection_ids.length > 0) return false;
      if (filter && filter !== UNASSIGNED && !d.collection_ids.includes(filter)) return false;
      if (!q) return true;
      return [d.title, d.filename, d.authors ?? ""].some((s) => s.toLowerCase().includes(q));
    });
  }, [documents.data, filter, query]);

  const setFilter = (value: string) => {
    setParams(value ? { collection: value } : {}, { replace: true });
  };

  return (
    <div className="page">
      <h1>Library</h1>
      <IndexStatusBanner />
      <UploadPanel
        collections={collections.data ?? []}
        knownDocumentIds={new Set((documents.data ?? []).map((d) => d.id))}
        defaultCollectionIds={filter && filter !== UNASSIGNED ? [filter] : []}
      />

      <section className="panel">
        <div className="toolbar">
          <h2>Documents</h2>
          <input
            type="search"
            placeholder="Filter by title, file or author"
            aria-label="Filter documents"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <select aria-label="Collection filter" value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="">All collections</option>
            <option value={UNASSIGNED}>Not in any collection</option>
            {(collections.data ?? []).map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
        </div>

        {documents.isPending && <Spinner label="Loading documents…" />}
        {documents.isError && (
          <ErrorBox error={documents.error} title="Could not load documents." onRetry={() => documents.refetch()} />
        )}
        {collections.isError && <ErrorBox error={collections.error} title="Could not load collections." />}
        {documents.data &&
          (visible.length === 0 ? (
            <p className="empty">
              {documents.data.length === 0 ? "No documents yet. Upload a PDF above." : "No documents match the filter."}
            </p>
          ) : (
            <div className="table-wrap">
              <table className="doc-table">
                <thead>
                  <tr>
                    <th>Document</th>
                    <th className="num">Pages</th>
                    <th>Status</th>
                    <th>Collections</th>
                    <th>Added</th>
                    <th>
                      <span className="sr-only">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {visible.map((d) => (
                    <DocumentRow key={d.id} document={d} collections={collections.data ?? []} />
                  ))}
                </tbody>
              </table>
            </div>
          ))}
      </section>
    </div>
  );
}
