"""Character-based text chunking.

Splits text into overlapping windows of `chunk_size` characters with
`chunk_overlap` characters of context carried over from the previous chunk.
Chunks are associated with the PDF page number they originate from.
"""

from dataclasses import dataclass


@dataclass
class TextChunk:
    text: str
    page_numbers: list[int]
    chunk_index: int


def chunk_page_texts(
    page_texts: list[tuple[int, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Split a sequence of (pdf_page_number, text) pairs into overlapping chunks.

    Pages are concatenated in order. Each resulting chunk records which PDF
    page numbers it spans so that citations can be traced back to the source.

    Args:
        page_texts: Ordered list of (zero-based pdf_page_number, extracted_text) tuples.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters to repeat at the start of the next chunk.

    Returns:
        List of TextChunk objects in order.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    # Build a flat character array annotated with the source page number.
    chars: list[str] = []
    char_pages: list[int] = []
    for page_num, text in page_texts:
        for ch in text:
            chars.append(ch)
            char_pages.append(page_num)

    total = len(chars)
    if total == 0:
        return []

    chunks: list[TextChunk] = []
    start = 0
    index = 0

    while start < total:
        end = min(start + chunk_size, total)
        chunk_text = "".join(chars[start:end])

        # Collect the unique ordered set of page numbers covered by this chunk.
        seen: set[int] = set()
        pages: list[int] = []
        for p in char_pages[start:end]:
            if p not in seen:
                seen.add(p)
                pages.append(p)

        chunks.append(TextChunk(text=chunk_text, page_numbers=pages, chunk_index=index))
        index += 1

        if end == total:
            break
        start = end - chunk_overlap

    return chunks
