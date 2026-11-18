"""Interpreting model output: insufficient-evidence signal and citation markers.

The model may only point at passages by number. Numbers that do not match a
passage given to it are removed from the answer, so the text can never refer
to a source the backend did not supply.
"""

import re
from dataclasses import dataclass

from app.chat.prompts import INSUFFICIENT_EVIDENCE_MARKER

# [1]  [1, 2]  [1,2,3]. At most three digits, so bracketed years like [2019] are
# left alone rather than treated (and removed) as invalid citations.
CITATION_PATTERN = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
# The marker at the start, possibly wrapped in markdown (**, `, #) or followed by ':'.
_INSUFFICIENT = re.compile(
    rf"^[\s*_`#>-]*{INSUFFICIENT_EVIDENCE_MARKER}[\s*_`:.\-]*", re.IGNORECASE
)
# Collapse whitespace left behind by removed markers before punctuation.
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([.,;:!?])")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


@dataclass(frozen=True)
class ParsedAnswer:
    text: str
    insufficient_evidence: bool
    # Valid passage numbers in order of first citation.
    cited_numbers: list[int]
    # Out-of-range numbers that were removed from the text.
    invalid_numbers: list[int]


def parse_answer(raw: str, passage_count: int) -> ParsedAnswer:
    text = raw.strip()

    insufficient = False
    match = _INSUFFICIENT.match(text)
    if match:
        insufficient = True
        text = text[match.end() :].strip()

    cited: list[int] = []
    invalid: list[int] = []

    def replace(m: re.Match[str]) -> str:
        numbers = [int(n) for n in re.split(r"\s*,\s*", m.group(1))]
        valid = [n for n in numbers if 1 <= n <= passage_count]
        for n in numbers:
            target = cited if n in valid else invalid
            if n not in target:
                target.append(n)
        return "".join(f"[{n}]" for n in valid)

    text = CITATION_PATTERN.sub(replace, text)
    if invalid:
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = _MULTI_SPACE.sub(" ", text).strip()

    return ParsedAnswer(
        text=text,
        insufficient_evidence=insufficient,
        cited_numbers=[] if insufficient else cited,
        invalid_numbers=invalid,
    )


def strip_citations(text: str) -> str:
    """Remove all citation markers, tidying the whitespace they leave behind."""
    text = CITATION_PATTERN.sub("", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return _MULTI_SPACE.sub(" ", text).strip()
