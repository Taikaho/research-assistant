"""PDF extraction using PyMuPDF (fitz).

Extracts per-page text and basic metadata from a PDF file.
"""

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz  # PyMuPDF


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


def extract_pdf(file_path: Path) -> DocumentMeta:
    """Open a PDF and extract text and metadata.

    Args:
        file_path: Absolute path to the PDF file.

    Returns:
        DocumentMeta with per-page text and document-level metadata.

    Raises:
        fitz.FileNotFoundError: If the file does not exist or cannot be opened.
        ValueError: If the file is not a valid PDF.
    """
    doc = fitz.open(str(file_path))  # type: ignore[no-untyped-call]

    meta = doc.metadata or {}
    title: str = (meta.get("title") or "").strip() or file_path.stem

    raw_author = (meta.get("author") or "").strip()
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
        text = page.get_text()  # type: ignore[no-untyped-call]
        pages.append(PageText(pdf_page_number=page_num, text=text))

    doc.close()  # type: ignore[no-untyped-call]

    return DocumentMeta(
        title=title,
        authors=authors,
        publication_year=year,
        page_count=len(pages),
        pages=pages,
    )
