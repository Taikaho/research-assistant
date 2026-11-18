import { useMutation } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { Link } from "react-router";

import { api } from "../api/client";
import { useCollections, useLibraryInvalidation } from "../api/hooks";
import type { Collection } from "../api/types";
import { ErrorBox, Spinner } from "../components/common";

function CreateCollectionForm() {
  const invalidate = useLibraryInvalidation();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const create = useMutation({
    mutationFn: api.createCollection,
    onSuccess: async () => {
      setName("");
      setDescription("");
      await invalidate();
    },
  });

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!name.trim()) return;
    create.mutate({ name: name.trim(), description: description.trim() || null });
  };

  return (
    <form className="panel form-inline" onSubmit={submit} aria-label="Create collection">
      <h2>New collection</h2>
      <label>
        Name
        <input value={name} onChange={(e) => setName(e.target.value)} maxLength={255} required />
      </label>
      <label className="grow">
        Description
        <input value={description} onChange={(e) => setDescription(e.target.value)} />
      </label>
      <button type="submit" className="btn btn-primary" disabled={create.isPending || !name.trim()}>
        {create.isPending ? "Creating…" : "Create"}
      </button>
      {create.isError && <ErrorBox error={create.error} />}
    </form>
  );
}

function CollectionRow({ collection }: { collection: Collection }) {
  const invalidate = useLibraryInvalidation();
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(collection.name);
  const [description, setDescription] = useState(collection.description ?? "");

  const update = useMutation({
    mutationFn: () =>
      api.updateCollection(collection.id, { name: name.trim(), description: description.trim() }),
    onSuccess: async () => {
      setEditing(false);
      await invalidate();
    },
  });
  const remove = useMutation({
    mutationFn: () => api.deleteCollection(collection.id),
    onSuccess: invalidate,
  });

  const confirmDelete = () => {
    if (
      window.confirm(
        `Delete the collection "${collection.name}"?\n\nIts documents stay in the library; only the grouping is removed.`,
      )
    ) {
      remove.mutate();
    }
  };

  if (editing) {
    return (
      <li className="collection-row">
        <form
          className="form-inline"
          aria-label={`Edit ${collection.name}`}
          onSubmit={(e) => {
            e.preventDefault();
            if (name.trim()) update.mutate();
          }}
        >
          <label>
            Name
            <input value={name} onChange={(e) => setName(e.target.value)} maxLength={255} required />
          </label>
          <label className="grow">
            Description
            <input value={description} onChange={(e) => setDescription(e.target.value)} />
          </label>
          <button type="submit" className="btn btn-primary" disabled={update.isPending || !name.trim()}>
            Save
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => {
              setEditing(false);
              setName(collection.name);
              setDescription(collection.description ?? "");
              update.reset();
            }}
          >
            Cancel
          </button>
        </form>
        {update.isError && <ErrorBox error={update.error} />}
      </li>
    );
  }

  const count = collection.document_count;
  return (
    <li className="collection-row">
      <div className="collection-info">
        <h3>{collection.name}</h3>
        {collection.description && <p className="muted">{collection.description}</p>}
        <p className="muted small">
          {count} {count === 1 ? "document" : "documents"}
        </p>
      </div>
      <div className="row-actions">
        <Link className="btn" to={`/chat?collections=${collection.id}`}>
          Ask
        </Link>
        <Link className="btn" to={`/library?collection=${collection.id}`}>
          Documents
        </Link>
        <button type="button" className="btn" onClick={() => setEditing(true)}>
          Rename
        </button>
        <button type="button" className="btn btn-danger" onClick={confirmDelete} disabled={remove.isPending}>
          {remove.isPending ? "Deleting…" : "Delete"}
        </button>
      </div>
      {remove.isError && <ErrorBox error={remove.error} />}
    </li>
  );
}

export function CollectionsPage() {
  const collections = useCollections();
  return (
    <div className="page">
      <h1>Collections</h1>
      <p className="muted">
        Collections group documents by topic. A document can belong to several collections, and
        chat questions are answered only from the collections you select.
      </p>
      <CreateCollectionForm />
      {collections.isPending && <Spinner label="Loading collections…" />}
      {collections.isError && (
        <ErrorBox error={collections.error} title="Could not load collections." onRetry={() => collections.refetch()} />
      )}
      {collections.data &&
        (collections.data.length === 0 ? (
          <p className="empty">No collections yet. Create one above, then add documents in the Library.</p>
        ) : (
          <ul className="collection-list">
            {collections.data.map((c) => (
              <CollectionRow key={c.id} collection={c} />
            ))}
          </ul>
        ))}
    </div>
  );
}
