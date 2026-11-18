export type AnswerSegment =
  | { kind: "text"; text: string }
  | { kind: "citation"; number: number };

// The backend normalises citations to single markers like [1][3] and removes any
// number that does not match a passage it supplied. Three digits at most, so
// bracketed years such as [2019] stay text (same rule as the backend).
const CITATION = /\[(\d{1,3})\]/g;

/**
 * Split an answer into text and citation segments. Only numbers present in
 * `validNumbers` become citations; anything else is left as plain text, so the
 * UI can never link to a source the backend did not return.
 */
export function splitCitations(answer: string, validNumbers: ReadonlySet<number>): AnswerSegment[] {
  const segments: AnswerSegment[] = [];
  let last = 0;
  for (const match of answer.matchAll(CITATION)) {
    const number = Number(match[1]);
    if (!validNumbers.has(number)) continue;
    const start = match.index ?? 0;
    if (start > last) segments.push({ kind: "text", text: answer.slice(last, start) });
    segments.push({ kind: "citation", number });
    last = start + match[0].length;
  }
  if (last < answer.length) segments.push({ kind: "text", text: answer.slice(last) });
  return segments;
}
