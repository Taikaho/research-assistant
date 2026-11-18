/** Half-open character range [start, end) within a page's text. */
export interface Range {
  start: number;
  end: number;
}

// Shorter overlaps between a page edge and a passage edge are too likely to be
// coincidental (e.g. a shared word), so they are not highlighted.
const MIN_OVERLAP = 12;

/**
 * Locate the part of a passage that lies on a given page.
 *
 * Chunks are cut from the concatenated page texts without modification, so a
 * passage is either inside the page, covers the whole page, starts on the page
 * and runs past its end, or began on an earlier page and ends on this one.
 * Returns null when the passage cannot be found, e.g. after reprocessing.
 */
export function findPassageRange(pageText: string, passage: string): Range | null {
  if (!pageText || !passage) return null;

  const inside = pageText.indexOf(passage);
  if (inside >= 0) return { start: inside, end: inside + passage.length };

  if (passage.includes(pageText)) return { start: 0, end: pageText.length };

  const max = Math.min(pageText.length, passage.length) - 1;
  // The passage starts on this page: a suffix of the page is a prefix of the passage.
  for (let k = max; k >= MIN_OVERLAP; k--) {
    if (pageText.endsWith(passage.slice(0, k))) {
      return { start: pageText.length - k, end: pageText.length };
    }
  }
  // The passage ends on this page: a prefix of the page is a suffix of the passage.
  for (let k = max; k >= MIN_OVERLAP; k--) {
    if (passage.endsWith(pageText.slice(0, k))) return { start: 0, end: k };
  }
  return null;
}
