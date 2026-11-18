import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { collection, mockFetch, models, renderApp } from "../test/utils";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("CollectionsPage", () => {
  it("lists collections with document counts", async () => {
    mockFetch([
      { path: "/api/v1/collections", body: [collection({ document_count: 3 })] },
      { path: "/api/v1/models", body: models() },
    ]);
    renderApp("/collections");
    expect(await screen.findByText("Spectroscopy")).toBeInTheDocument();
    expect(screen.getByText("3 documents")).toBeInTheDocument();
  });

  it("creates a collection and shows a conflict error", async () => {
    const posted: unknown[] = [];
    mockFetch([
      { path: "/api/v1/collections", body: [] },
      { path: "/api/v1/models", body: models() },
      {
        method: "POST",
        path: "/api/v1/collections",
        status: 409,
        handler: (init) => {
          posted.push(JSON.parse(String(init?.body)));
          return { detail: "A collection named 'Botany' already exists." };
        },
      },
    ]);
    renderApp("/collections");
    const form = await screen.findByRole("form", { name: "Create collection" });
    await userEvent.type(within(form).getByLabelText("Name"), "Botany");
    await userEvent.click(within(form).getByRole("button", { name: "Create" }));

    expect(await screen.findByText(/already exists/)).toBeInTheDocument();
    expect(posted).toEqual([{ name: "Botany", description: null }]);
  });

  it("renames a collection", async () => {
    const patched: unknown[] = [];
    mockFetch([
      { path: "/api/v1/collections", body: [collection()] },
      { path: "/api/v1/models", body: models() },
      {
        method: "PATCH",
        path: "/api/v1/collections/col-1",
        handler: (init) => {
          patched.push(JSON.parse(String(init?.body)));
          return collection({ name: "Spectra" });
        },
      },
    ]);
    renderApp("/collections");
    await userEvent.click(await screen.findByRole("button", { name: "Rename" }));
    const form = screen.getByRole("form", { name: "Edit Spectroscopy" });
    const name = within(form).getByLabelText("Name");
    await userEvent.clear(name);
    await userEvent.type(name, "Spectra");
    await userEvent.click(within(form).getByRole("button", { name: "Save" }));
    await waitFor(() => expect(patched).toEqual([{ name: "Spectra", description: "" }]));
  });

  it("deletes a collection only after confirmation", async () => {
    const fetchMock = mockFetch([
      { path: "/api/v1/collections", body: [collection()] },
      { path: "/api/v1/models", body: models() },
      { method: "DELETE", path: "/api/v1/collections/col-1" },
    ]);
    vi.spyOn(window, "confirm").mockReturnValue(true);
    renderApp("/collections");
    await userEvent.click(await screen.findByRole("button", { name: "Delete" }));
    await waitFor(() =>
      expect(fetchMock.mock.calls.some(([, init]) => init?.method === "DELETE")).toBe(true),
    );
  });
});
