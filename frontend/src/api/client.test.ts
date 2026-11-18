import { afterEach, describe, expect, it, vi } from "vitest";

import { mockFetch } from "../test/utils";
import { api, ApiError, formatDetail, pdfUrl } from "./client";

afterEach(() => vi.unstubAllGlobals());

describe("formatDetail", () => {
  it("passes string details through", () => {
    expect(formatDetail("Collection not found.")).toBe("Collection not found.");
  });

  it("joins FastAPI validation issues with their field names", () => {
    expect(
      formatDetail([
        { loc: ["body", "question"], msg: "String should have at least 1 character" },
        { loc: ["body"], msg: "bad" },
      ]),
    ).toBe("question: String should have at least 1 character; bad");
  });

  it("returns null for unknown shapes", () => {
    expect(formatDetail({ foo: 1 })).toBeNull();
  });
});

describe("request errors", () => {
  it("raises ApiError with the backend detail", async () => {
    mockFetch([{ path: "/api/v1/collections", status: 409, body: { detail: "Name taken." } }]);
    await expect(api.listCollections()).rejects.toEqual(new ApiError(409, "Name taken."));
  });

  it("explains an unreachable backend", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(api.listDocuments()).rejects.toMatchObject({
      status: 0,
      message: expect.stringContaining("Cannot reach the backend"),
    });
  });

  it("falls back to a status hint when the body is not JSON", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("<html>", { status: 504 })));
    await expect(api.listDocuments()).rejects.toMatchObject({ message: "The request timed out." });
  });
});

describe("api calls", () => {
  it("uploads the file with each collection id as a form field", async () => {
    const fetchMock = mockFetch([{ method: "POST", path: "/api/v1/documents", body: { id: "d" } }]);
    const file = new File(["%PDF"], "a.pdf", { type: "application/pdf" });
    await api.uploadDocument(file, ["c1", "c2"]);
    const body = fetchMock.mock.calls[0]![1]!.body as FormData;
    expect(body.get("file")).toBeInstanceOf(File);
    expect(body.getAll("collection_ids")).toEqual(["c1", "c2"]);
  });

  it("returns undefined for 204 responses", async () => {
    mockFetch([{ method: "DELETE", path: "/api/v1/documents/d1" }]);
    await expect(api.deleteDocument("d1")).resolves.toBeUndefined();
  });

  it("builds one-based PDF page URLs from zero-based indexes", () => {
    expect(pdfUrl("d1", 0)).toBe("/api/v1/documents/d1/file#page=1");
    expect(pdfUrl("d1")).toBe("/api/v1/documents/d1/file");
  });
});
