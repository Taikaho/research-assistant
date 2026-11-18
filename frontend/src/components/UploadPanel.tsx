import { useEffect, useRef, useState, type DragEvent } from "react";

import { api, errorMessage } from "../api/client";
import { useLibraryInvalidation } from "../api/hooks";
import type { Collection } from "../api/types";
import { Spinner } from "./common";

type UploadState = "queued" | "uploading" | "done" | "duplicate" | "failed";

interface UploadItem {
  key: string;
  file: File;
  state: UploadState;
  message?: string;
}

const STATE_LABELS: Record<UploadState, string> = {
  queued: "Waiting",
  uploading: "Uploading and processing…",
  done: "Added",
  duplicate: "Already in library — added to the selected collections",
  failed: "Failed",
};

function isPdf(file: File): boolean {
  return file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
}

export function UploadPanel({
  collections,
  knownDocumentIds,
  defaultCollectionIds,
}: {
  collections: Collection[];
  knownDocumentIds: ReadonlySet<string>;
  defaultCollectionIds: string[];
}) {
  const invalidate = useLibraryInvalidation();
  const inputRef = useRef<HTMLInputElement>(null);
  const [targets, setTargets] = useState<string[]>(defaultCollectionIds);
  const [items, setItems] = useState<UploadItem[]>([]);
  const [running, setRunning] = useState(false);
  const [dragging, setDragging] = useState(false);

  const defaultsKey = defaultCollectionIds.join(",");
  useEffect(() => {
    setTargets(defaultsKey ? defaultsKey.split(",") : []);
  }, [defaultsKey]);

  const patch = (key: string, change: Partial<UploadItem>) =>
    setItems((current) => current.map((item) => (item.key === key ? { ...item, ...change } : item)));

  // Uploads run one at a time: each request extracts and embeds the PDF on the
  // backend, which is CPU-heavy, and sequential progress is easier to follow.
  const upload = async (files: File[]) => {
    const queued: UploadItem[] = files.map((file, i) => ({
      key: `${Date.now()}-${i}-${file.name}`,
      file,
      state: isPdf(file) ? "queued" : "failed",
      message: isPdf(file) ? undefined : "Only PDF files are accepted.",
    }));
    setItems((current) => [...queued, ...current]);
    setRunning(true);
    const known = new Set(knownDocumentIds);
    for (const item of queued) {
      if (item.state !== "queued") continue;
      patch(item.key, { state: "uploading" });
      try {
        const document = await api.uploadDocument(item.file, targets);
        const duplicate = known.has(document.id);
        known.add(document.id);
        patch(item.key, {
          state: duplicate ? "duplicate" : "done",
          message: document.processing_status === "completed" ? document.title : `Status: ${document.processing_status}`,
        });
      } catch (error) {
        patch(item.key, { state: "failed", message: errorMessage(error) });
      }
      await invalidate();
    }
    setRunning(false);
  };

  const pick = (fileList: FileList | null) => {
    if (fileList && fileList.length > 0 && !running) void upload(Array.from(fileList));
  };

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    pick(event.dataTransfer.files);
  };

  const toggleTarget = (id: string) =>
    setTargets((current) => (current.includes(id) ? current.filter((t) => t !== id) : [...current, id]));

  return (
    <section className="panel" aria-label="Upload documents">
      <h2>Upload PDFs</h2>
      <div className="upload-grid">
        <div
          className={`dropzone${dragging ? " dragging" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <p>Drop PDF files here or</p>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => inputRef.current?.click()}
            disabled={running}
          >
            Choose files
          </button>
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,application/pdf"
            multiple
            hidden
            data-testid="file-input"
            onChange={(e) => {
              pick(e.target.files);
              e.target.value = "";
            }}
          />
          <p className="muted small">Text is extracted and embedded locally; large files can take a minute.</p>
        </div>
        <fieldset className="target-collections">
          <legend>Add uploads to collections</legend>
          {collections.length === 0 && <p className="muted small">No collections yet.</p>}
          {collections.map((c) => (
            <label key={c.id} className="checkbox">
              <input
                type="checkbox"
                checked={targets.includes(c.id)}
                onChange={() => toggleTarget(c.id)}
                disabled={running}
              />
              {c.name}
            </label>
          ))}
        </fieldset>
      </div>

      {items.length > 0 && (
        <ul className="upload-list" aria-label="Upload progress">
          {items.map((item) => (
            <li key={item.key} className={`upload-item upload-${item.state}`}>
              <span className="upload-name">{item.file.name}</span>
              <span className="upload-state">
                {item.state === "uploading" ? <Spinner label={STATE_LABELS.uploading} /> : STATE_LABELS[item.state]}
              </span>
              {item.message && <span className="upload-message small">{item.message}</span>}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
