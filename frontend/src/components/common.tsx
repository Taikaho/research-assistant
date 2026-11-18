import type { ReactNode } from "react";

import { errorMessage } from "../api/client";

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="spinner-wrap" role="status">
      <span className="spinner" aria-hidden="true" />
      {label && <span>{label}</span>}
    </span>
  );
}

export function ErrorBox({
  error,
  title,
  onRetry,
}: {
  error: unknown;
  title?: string;
  onRetry?: () => void;
}) {
  return (
    <div className="alert alert-error" role="alert">
      <div>
        {title && <strong>{title} </strong>}
        {errorMessage(error)}
      </div>
      {onRetry && (
        <button type="button" className="btn btn-small" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function Notice({ kind = "info", children }: { kind?: "info" | "warning"; children: ReactNode }) {
  return (
    <div className={`alert alert-${kind}`} role={kind === "warning" ? "alert" : "note"}>
      {children}
    </div>
  );
}

const STATUS_LABELS: Record<string, string> = {
  completed: "Ready",
  processing: "Processing",
  pending: "Pending",
  failed: "Failed",
};

export function StatusBadge({ status }: { status: string }) {
  return <span className={`badge badge-status-${status}`}>{STATUS_LABELS[status] ?? status}</span>;
}

export function LocalityBadge({ isLocal }: { isLocal: boolean }) {
  return isLocal ? (
    <span className="badge badge-local" title="Runs on this machine">
      Local
    </span>
  ) : (
    <span className="badge badge-cloud" title="Question and passages are sent to an external provider">
      Cloud
    </span>
  );
}
