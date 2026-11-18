# 0001 — Use the browser's built-in PDF viewer for the source viewer

- Status: accepted
- Date: 2026-11-18
- Phase: 5 (Frontend)

## Context

SPEC §8.4 requires the source viewer to display the original PDF, navigate to
the referenced page, show the supporting passage and highlight relevant text
"where technically feasible".

Options considered:

1. **Browser built-in viewer in an iframe**, opened at `…/file#page=N`.
2. **PDF.js (`pdfjs-dist` / `react-pdf`)** rendering pages to canvas with a text
   layer, which would allow highlighting inside the rendered page.

## Decision

Use the browser's built-in viewer (option 1). The backend serves the PDF with
`Content-Disposition: inline` from `GET /api/v1/documents/{id}/file`.

Highlighting is done next to the PDF, in the page's extracted text
(`GET /api/v1/documents/{id}/pages/{page}`): chunks are cut verbatim from the
concatenated page texts, so the passage can be located exactly
(`frontend/src/lib/highlight.ts`), including passages that span pages.

## Consequences

- No extra dependency or web-worker setup; the PDF renders with the browser's
  full feature set (zoom, search, print).
- `#page=N` is supported by Chrome, Edge and Firefox. The iframe is remounted on
  page change because not every viewer reacts to a hash change.
- The highlight is shown in the extracted text, not on the rendered PDF page.
  If in-page highlighting becomes a requirement, PDF.js with a text layer can
  replace the iframe without backend changes.
