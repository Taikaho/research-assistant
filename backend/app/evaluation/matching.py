"""Deciding whether a retrieved chunk is relevant to a dataset passage.

A chunk is relevant when it comes from the passage's document and covers its
page and, if the passage text is given, contains enough of that text. Passages
are copied by people from a PDF viewer, so text is normalised first (Unicode
forms, hyphenation at line ends, whitespace, case).
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.evaluation.dataset import RelevantPassage

_HYPHEN_BREAK = re.compile(r"(\w)[-­]\s*\n\s*(\w)")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = text.replace("­", "")
    return _SPACE.sub(" ", text).strip().lower()


def passage_coverage(chunk_text: str, passage: str) -> float:
    """Fraction of the passage found as one contiguous block in the chunk (0..1).

    A passage can straddle a chunk boundary, so partial coverage is expected;
    the caller decides the threshold.
    """
    p = normalize(passage)
    c = normalize(chunk_text)
    if not p:
        return 0.0
    if p in c:
        return 1.0
    match = SequenceMatcher(None, p, c, autojunk=False).find_longest_match(0, len(p), 0, len(c))
    return match.size / len(p)


@dataclass(frozen=True)
class ChunkRef:
    """The parts of a retrieved chunk needed for relevance judgements."""

    dataset_document: str | None  # dataset document id, None if not in the dataset
    pages: Sequence[int]  # zero-based
    text: str


def matches(chunk: ChunkRef, relevant: RelevantPassage, threshold: float) -> bool:
    if chunk.dataset_document != relevant.document:
        return False
    if relevant.page_index not in chunk.pages:
        return False
    if relevant.passage is None:
        return True
    return passage_coverage(chunk.text, relevant.passage) >= threshold


def matched_items(
    chunk: ChunkRef, relevant: Sequence[RelevantPassage], threshold: float
) -> set[int]:
    """Indexes of the relevant passages this chunk matches."""
    return {i for i, r in enumerate(relevant) if matches(chunk, r, threshold)}


def passage_on_page(page_text: str, next_page_text: str, passage: str, threshold: float) -> bool:
    """True when the passage is on the page, or starts on it and continues on the next.

    A passage lying entirely on the next page does not count: the dataset's page
    number would then be wrong.
    """
    if passage_coverage(page_text, passage) >= threshold:
        return True
    page = normalize(page_text)
    combined = f"{page} {normalize(next_page_text)}"
    start = combined.find(normalize(passage))
    return 0 <= start < len(page)
