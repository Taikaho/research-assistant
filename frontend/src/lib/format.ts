import type { SourceReference } from "../api/types";

/**
 * Human-readable page label. The backend uses zero-based PDF page indexes;
 * people (and PDF viewers) count from 1. Printed page labels are shown when the
 * PDF provides them, because they can differ from the PDF page position.
 */
export function pageLabel(pdfPage: number, printed?: string | null): string {
  const label = `PDF p. ${pdfPage + 1}`;
  return printed ? `${label} (printed p. ${printed})` : label;
}

export function sourcePagesLabel(source: SourceReference): string {
  const pages = source.pdf_page_numbers;
  if (pages.length === 0) return "page unknown";
  if (pages.length === 1) return pageLabel(pages[0]!, source.printed_page_numbers[0]);
  const first = pages[0]!;
  const last = pages[pages.length - 1]!;
  const printedFirst = source.printed_page_numbers[0];
  const printedLast = source.printed_page_numbers[pages.length - 1];
  const label = `PDF pp. ${first + 1}–${last + 1}`;
  return printedFirst && printedLast ? `${label} (printed pp. ${printedFirst}–${printedLast})` : label;
}

export function formatScore(score: number): string {
  return score.toFixed(2);
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export function truncate(text: string, max: number): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length <= max ? clean : `${clean.slice(0, max - 1).trimEnd()}…`;
}
