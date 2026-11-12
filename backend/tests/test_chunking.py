"""Unit tests for the character-based chunking logic."""

import pytest

from app.documents.chunking import TextChunk, chunk_page_texts


@pytest.mark.unit
class TestChunkPageTexts:
    def test_empty_input_returns_empty(self) -> None:
        assert chunk_page_texts([], chunk_size=100, chunk_overlap=10) == []

    def test_empty_text_returns_empty(self) -> None:
        assert chunk_page_texts([(0, "")], chunk_size=100, chunk_overlap=10) == []

    def test_text_shorter_than_chunk_size_yields_one_chunk(self) -> None:
        result = chunk_page_texts([(0, "hello world")], chunk_size=100, chunk_overlap=10)
        assert len(result) == 1
        assert result[0].text == "hello world"
        assert result[0].chunk_index == 0
        assert result[0].page_numbers == [0]

    def test_exact_chunk_size_yields_one_chunk(self) -> None:
        text = "a" * 50
        result = chunk_page_texts([(0, text)], chunk_size=50, chunk_overlap=0)
        assert len(result) == 1
        assert result[0].text == text

    def test_overlap_carries_context_to_next_chunk(self) -> None:
        text = "abcdefghij"  # 10 chars
        result = chunk_page_texts([(0, text)], chunk_size=6, chunk_overlap=2)
        # chunk 0: chars 0-5  "abcdef"
        # chunk 1: chars 4-9  "efghij"
        assert len(result) == 2
        assert result[0].text == "abcdef"
        assert result[1].text == "efghij"

    def test_chunk_indices_are_sequential(self) -> None:
        text = "x" * 30
        result = chunk_page_texts([(0, text)], chunk_size=10, chunk_overlap=0)
        assert [c.chunk_index for c in result] == list(range(len(result)))

    def test_page_numbers_tracked_across_pages(self) -> None:
        # Page 0: 5 chars, Page 1: 5 chars — with chunk_size=8 the first chunk spans both.
        result = chunk_page_texts(
            [(0, "aaaaa"), (1, "bbbbb")], chunk_size=8, chunk_overlap=0
        )
        assert result[0].page_numbers == [0, 1]
        assert result[1].page_numbers == [1]

    def test_no_overlap_produces_non_overlapping_chunks(self) -> None:
        text = "1234567890"
        result = chunk_page_texts([(0, text)], chunk_size=5, chunk_overlap=0)
        assert result[0].text == "12345"
        assert result[1].text == "67890"
        assert len(result) == 2

    def test_invalid_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_page_texts([(0, "hello")], chunk_size=0, chunk_overlap=0)

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            chunk_page_texts([(0, "hello")], chunk_size=10, chunk_overlap=-1)

    def test_overlap_equal_to_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            chunk_page_texts([(0, "hello")], chunk_size=5, chunk_overlap=5)

    def test_multiple_pages_all_captured(self) -> None:
        pages = [(i, f"page{i}" * 10) for i in range(5)]
        result = chunk_page_texts(pages, chunk_size=20, chunk_overlap=5)
        # All chars should be recoverable from the chunks (allowing overlaps).
        full_text = "".join(text for _, text in pages)
        # First chunk starts at position 0 and last chunk ends at len(full_text).
        assert result[0].text == full_text[:20]
        assert result[-1].text == full_text[-(len(full_text) % 15 or 20):]

    def test_returns_text_chunk_dataclass(self) -> None:
        result = chunk_page_texts([(0, "hello")], chunk_size=100, chunk_overlap=0)
        assert isinstance(result[0], TextChunk)
