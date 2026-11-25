"""PDF extraction using PyMuPDF (fitz).

Extracts per-page text and basic metadata from a PDF file.
"""

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz  # PyMuPDF


def normalize_text(text: str) -> str:
    """Normalise extracted PDF text to Unicode NFKC and drop NUL characters.

    NFKC folds compatibility characters that PDFs often contain into their plain
    forms, e.g. the ligatures "ﬁ"/"ﬂ" into "fi"/"fl" and full-width letters into
    ASCII, so that a query for "field" matches text extracted as "ﬁeld". It also
    composes decomposed letters ("a" + combining diaeresis -> "ä"). The trade-off
    is that some typographic distinctions are lost, e.g. "m²" becomes "m2".

    PostgreSQL text columns cannot store NUL, which some broken PDFs emit.
    """
    return unicodedata.normalize("NFKC", text).replace("\x00", "")


@dataclass
class PageText:
    pdf_page_number: int  # zero-based
    text: str


@dataclass
class DocumentMeta:
    title: str
    authors: str | None
    publication_year: int | None
    page_count: int
    pages: list[PageText]


def extract_pdf(file_path: Path, fallback_title: str | None = None) -> DocumentMeta:
    """Open a PDF and extract text and metadata.

    Args:
        file_path: Absolute path to the PDF file.
        fallback_title: Title used when the PDF has no title metadata, typically the
            uploaded filename without extension. Defaults to the stored file's stem.

    Returns:
        DocumentMeta with per-page text and document-level metadata.

    Raises:
        fitz.FileNotFoundError: If the file does not exist or cannot be opened.
        ValueError: If the file is not a valid PDF.
    """
    doc = fitz.open(str(file_path))  # type: ignore[no-untyped-call]

    meta = doc.metadata or {}
    title: str = normalize_text(meta.get("title") or "").strip() or fallback_title or file_path.stem

    raw_author = normalize_text(meta.get("author") or "").strip()
    authors: str | None = raw_author if raw_author else None

    year: int | None = None
    creation_date = meta.get("creationDate") or ""
    # PDF creation date format: "D:YYYYMMDDHHmmSS..."
    if creation_date.startswith("D:") and len(creation_date) >= 6:
        try:
            year = int(creation_date[2:6])
        except ValueError:
            pass

    pages: list[PageText] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = normalize_text(page.get_text())  # type: ignore[no-untyped-call]
        pages.append(PageText(pdf_page_number=page_num, text=text))

    doc.close()  # type: ignore[no-untyped-call]

    return DocumentMeta(
        title=title,
        authors=authors,
        publication_year=year,
        page_count=len(pages),
        pages=pages,
    )
