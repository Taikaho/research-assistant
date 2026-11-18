import { useState } from "react";
import { Link } from "react-router";

import type { Collection } from "../../api/types";

/**
 * Shows which collections the next question searches, and lets the user change them.
 * The active selection is always visible as chips (FR-10).
 */
export function ScopeSelector({
  collections,
  selected,
  onChange,
  startOpen,
  disabled,
}: {
  collections: Collection[];
  selected: string[];
  onChange: (ids: string[]) => void;
  startOpen: boolean;
  disabled?: boolean;
}) {
  // Until the user toggles it, follow startOpen (which changes once collections load).
  const [userOpen, setOpen] = useState<boolean | null>(null);
  const open = userOpen ?? startOpen;
  const byId = new Map(collections.map((c) => [c.id, c]));
  const active = selected.filter((id) => byId.has(id));

  const toggle = (id: string) =>
    onChange(active.includes(id) ? active.filter((s) => s !== id) : [...active, id]);

  return (
    <div className="scope" aria-label="Search scope">
      <div className="scope-summary">
        <span className="scope-label">Searching in:</span>
        {active.length === 0 ? (
          <span className="text-warning">no collection selected</span>
        ) : (
          active.map((id) => (
            <span key={id} className="chip chip-scope">
              {byId.get(id)!.name}
            </span>
          ))
        )}
        <button
          type="button"
          className="btn btn-small btn-link"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
          disabled={disabled}
        >
          {open ? "Done" : "Change"}
        </button>
      </div>
      {open && (
        <div className="scope-options">
          {collections.length === 0 && (
            <p className="muted small">
              No collections yet. <Link to="/collections">Create one</Link> and add documents to it.
            </p>
          )}
          {collections.map((c) => (
            <label key={c.id} className="checkbox">
              <input
                type="checkbox"
                checked={active.includes(c.id)}
                onChange={() => toggle(c.id)}
                disabled={disabled}
              />
              {c.name}
              <span className="muted small">
                ({c.document_count} {c.document_count === 1 ? "doc" : "docs"})
              </span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
