import { describe, expect, it } from "vitest";

import { models, source } from "../test/utils";
import { splitCitations } from "./citations";
import { pageLabel, sourcePagesLabel, truncate } from "./format";
import { findPassageRange } from "./highlight";
import { DEFAULT_SETTINGS, resolveModel } from "./modelSettings";

describe("splitCitations", () => {
  it("turns known citation numbers into citation segments", () => {
    expect(splitCitations("A [1] and B [2].", new Set([1, 2]))).toEqual([
      { kind: "text", text: "A " },
      { kind: "citation", number: 1 },
      { kind: "text", text: " and B " },
      { kind: "citation", number: 2 },
      { kind: "text", text: "." },
    ]);
  });

  it("leaves unknown numbers and bracketed years as text", () => {
    expect(splitCitations("See [9] from [2019].", new Set([1]))).toEqual([
      { kind: "text", text: "See [9] from [2019]." },
    ]);
  });

  it("handles adjacent citations", () => {
    expect(splitCitations("X[1][2]", new Set([1, 2]))).toEqual([
      { kind: "text", text: "X" },
      { kind: "citation", number: 1 },
      { kind: "citation", number: 2 },
    ]);
  });
});

describe("findPassageRange", () => {
  const page = "Intro text. LDA is a supervised method. More text follows here.";

  it("finds a passage contained in the page", () => {
    expect(findPassageRange(page, "LDA is a supervised method.")).toEqual({ start: 12, end: 39 });
  });

  it("finds a passage that starts on the page and continues on the next", () => {
    const passage = "More text follows here.NEXT PAGE CONTENT";
    const range = findPassageRange(page, passage)!;
    expect(page.slice(range.start, range.end)).toBe("More text follows here.");
  });

  it("finds a passage that began on an earlier page", () => {
    const passage = "PREVIOUS PAGE TAIL Intro text. LDA";
    const range = findPassageRange(page, passage)!;
    expect(range.start).toBe(0);
    expect(page.slice(range.start, range.end)).toBe("Intro text. LDA");
  });

  it("highlights the whole page when the passage spans it", () => {
    expect(findPassageRange("middle page", "start. middle page end.")).toEqual({ start: 0, end: 11 });
  });

  it("handles long passages with a short part on the page", () => {
    const longPassage = "follows here." + "x".repeat(2000);
    const range = findPassageRange(page, longPassage)!;
    expect(page.slice(range.start)).toBe("follows here.");
  });

  it("returns null when the passage is not on the page", () => {
    expect(findPassageRange(page, "Completely different words")).toBeNull();
    expect(findPassageRange("", "x")).toBeNull();
  });

  it("ignores coincidental short overlaps", () => {
    expect(findPassageRange("ends with here.", "here. and then something else")).toBeNull();
  });
});

describe("page labels", () => {
  it("shows one-based PDF pages and printed labels", () => {
    expect(pageLabel(0)).toBe("PDF p. 1");
    expect(pageLabel(6, "123")).toBe("PDF p. 7 (printed p. 123)");
  });

  it("labels page ranges", () => {
    expect(sourcePagesLabel(source({ pdf_page_numbers: [4, 5], printed_page_numbers: [null, null] }))).toBe(
      "PDF pp. 5–6",
    );
    expect(sourcePagesLabel(source({ pdf_page_numbers: [], printed_page_numbers: [] }))).toBe("page unknown");
  });

  it("truncates and collapses whitespace", () => {
    expect(truncate("a  b\nc", 10)).toBe("a b c");
    expect(truncate("abcdefghij", 5)).toBe("abcd…");
  });
});

describe("resolveModel", () => {
  it("falls back to the default provider and its first model", () => {
    const { provider, model } = resolveModel(DEFAULT_SETTINGS, models());
    expect(provider?.name).toBe("ollama");
    expect(model).toBe("llama3.2:3b");
  });

  it("uses the saved provider and model when they exist", () => {
    const result = resolveModel({ ...DEFAULT_SETTINGS, provider: "ollama", model: "qwen2.5:7b" }, models());
    expect(result.model).toBe("qwen2.5:7b");
  });

  it("uses the provider default model when the saved model is gone", () => {
    const result = resolveModel({ ...DEFAULT_SETTINGS, provider: "openai", model: "removed" }, models());
    expect(result.provider?.name).toBe("openai");
    expect(result.model).toBe("gpt-test");
  });

  it("falls back to the default provider when the saved one was removed", () => {
    const result = resolveModel({ ...DEFAULT_SETTINGS, provider: "gone", model: "x" }, models());
    expect(result.provider?.name).toBe("ollama");
    expect(result.model).toBe("llama3.2:3b");
  });

  it("keeps the saved model while its provider is unreachable", () => {
    const response = models();
    response.providers[0] = { ...response.providers[0]!, available: false, models: [] };
    const result = resolveModel({ ...DEFAULT_SETTINGS, provider: "ollama", model: "qwen2.5:7b" }, response);
    expect(result.model).toBe("qwen2.5:7b");
  });

  it("returns nothing when no providers are configured", () => {
    expect(resolveModel(DEFAULT_SETTINGS, models({ providers: [] }))).toEqual({ provider: null, model: null });
  });
});
