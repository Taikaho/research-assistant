import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { collection, documentInfo, mockFetch, models, renderApp, type MockRoute } from "../test/utils";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const indexStatus = {
  configured_model_name: "BAAI/bge-small-en-v1.5",
  active_model: null,
  total_chunks: 10,
  indexed_chunks: 10,
  reindex_required: false,
};

const routes = (extra: MockRoute[] = []): MockRoute[] => [
  ...extra,
  { path: "/api/v1/collections", body: [collection(), collection({ id: "col-2", name: "Botany", document_count: 0 })] },
  {
    path: "/api/v1/documents",
    body: [documentInfo(), documentInfo({ id: "doc-2", title: "Failed scan", processing_status: "failed", collection_ids: [] })],
  },
  { path: "/api/v1/embeddings/status", body: indexStatus },
  { path: "/api/v1/models", body: models() },
];

describe("LibraryPage", () => {
  it("lists documents with metadata, status and membership", async () => {
    mockFetch(routes());
    renderApp("/library");
    const row = (await screen.findByText("Hyperspectral Imaging")).closest("tr")!;
    expect(within(row).getByText("Ready")).toBeInTheDocument();
    expect(within(row).getByText(/A\. Author/)).toBeInTheDocument();
    expect(within(row).getByText("Spectroscopy")).toBeInTheDocument();
    expect(within(row).getByRole("link", { name: "Open PDF" })).toHaveAttribute("href", "/api/v1/documents/doc-1/file");
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });

  it("filters by collection", async () => {
    mockFetch(routes());
    renderApp("/library?collection=col-1");
    await screen.findByText("Hyperspectral Imaging");
    expect(screen.queryByText("Failed scan")).not.toBeInTheDocument();
  });

  it("adds a document to a collection and removes it from one", async () => {
    const fetchMock = mockFetch(
      routes([
        { method: "PUT", path: "/api/v1/collections/col-2/documents/doc-1" },
        { method: "DELETE", path: "/api/v1/collections/col-1/documents/doc-1" },
      ]),
    );
    renderApp("/library");
    await screen.findByText("Hyperspectral Imaging");

    await userEvent.selectOptions(screen.getByLabelText("Add Hyperspectral Imaging to collection"), "col-2");
    await userEvent.click(screen.getByRole("button", { name: "Remove from Spectroscopy" }));

    await waitFor(() => {
      const calls = fetchMock.mock.calls.map(([url, init]) => `${init?.method ?? "GET"} ${String(url)}`);
      expect(calls).toContain("PUT /api/v1/collections/col-2/documents/doc-1");
      expect(calls).toContain("DELETE /api/v1/collections/col-1/documents/doc-1");
    });
  });

  it("asks for confirmation before deleting a document", async () => {
    const fetchMock = mockFetch(routes([{ method: "DELETE", path: "/api/v1/documents/doc-1" }]));
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);
    renderApp("/library");
    const row = (await screen.findByText("Hyperspectral Imaging")).closest("tr")!;

    await userEvent.click(within(row).getByRole("button", { name: "Delete" }));
    expect(fetchMock.mock.calls.some(([, init]) => init?.method === "DELETE")).toBe(false);

    await userEvent.click(within(row).getByRole("button", { name: "Delete" }));
    await waitFor(() =>
      expect(fetchMock.mock.calls.some(([url, init]) => init?.method === "DELETE" && String(url).endsWith("/doc-1"))).toBe(true),
    );
    expect(confirm).toHaveBeenCalledTimes(2);
  });

  it("uploads PDFs into the selected collections and reports progress", async () => {
    let form: FormData | undefined;
    mockFetch(
      routes([
        {
          method: "POST",
          path: "/api/v1/documents",
          handler: (init) => {
            form = init?.body as FormData;
            return documentInfo({ id: "doc-new", title: "New Paper" });
          },
        },
      ]),
    );
    renderApp("/library");
    await screen.findByText("Hyperspectral Imaging");

    const panel = screen.getByRole("region", { name: "Upload documents" });
    await userEvent.click(within(panel).getByRole("checkbox", { name: "Botany" }));
    const file = new File(["%PDF-1.4"], "new.pdf", { type: "application/pdf" });
    await userEvent.upload(screen.getByTestId("file-input"), file);

    const progress = await screen.findByRole("list", { name: "Upload progress" });
    expect(await within(progress).findByText("Added")).toBeInTheDocument();
    expect(within(progress).getByText("New Paper")).toBeInTheDocument();
    expect(form?.getAll("collection_ids")).toEqual(["col-2"]);
  });

  it("reports duplicates and per-file failures", async () => {
    let call = 0;
    mockFetch(
      routes([
        {
          method: "POST",
          path: "/api/v1/documents",
          handler: () => {
            call += 1;
            if (call === 1) return documentInfo();
            throw new Error("network down");
          },
        },
      ]),
    );
    renderApp("/library");
    await screen.findByText("Hyperspectral Imaging");
    await userEvent.upload(screen.getByTestId("file-input"), [
      new File(["%PDF"], "same.pdf", { type: "application/pdf" }),
      new File(["%PDF"], "broken.pdf", { type: "application/pdf" }),
    ]);

    const progress = await screen.findByRole("list", { name: "Upload progress" });
    expect(await within(progress).findByText(/Already in library/)).toBeInTheDocument();
    expect(await within(progress).findByText("Failed")).toBeInTheDocument();
  });

  it("offers re-indexing when the index is out of date", async () => {
    mockFetch(
      routes([
        {
          path: "/api/v1/embeddings/status",
          body: { ...indexStatus, indexed_chunks: 4, reindex_required: true },
        },
      ]),
    );
    renderApp("/library");
    expect(await screen.findByText(/4 of 10 passages are indexed/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Re-index now" })).toBeInTheDocument();
  });
});
