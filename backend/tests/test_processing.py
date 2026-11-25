"""Unit tests for PDF text extraction and Unicode normalisation."""

import unicodedata
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.documents.processing import extract_pdf, normalize_text


@pytest.mark.unit
class TestNormalizeText:
    def test_ligatures_are_expanded(self) -> None:
        text = "ﬁeld ﬂuorescence"
        assert normalize_text(text) == "field fluorescence"

    def test_decomposed_finnish_letters_are_composed(self) -> None:
        decomposed = unicodedata.normalize("NFD", "käyttö")
        assert len(decomposed) == 8  # two combining diaereses
        assert normalize_text(decomposed) == "käyttö"

    def test_finnish_and_punctuation_are_kept(self) -> None:
        text = "Äänitaajuus – Schrödinger’s “cat”"
        assert normalize_text(text) == text

    def test_fullwidth_characters_are_folded(self) -> None:
        assert normalize_text("ＰＣＡ１") == "PCA1"

    def test_nul_characters_are_removed(self) -> None:
        assert normalize_text("ab\x00c") == "abc"

    def test_text_is_valid_for_win1252_after_ligatures(self) -> None:
        # The original upload failure: U+FB01 has no WIN1252 equivalent.
        normalize_text("\N{LATIN SMALL LIGATURE FI}eld").encode("cp1252")


def _fake_doc(pages: list[str], metadata: dict[str, str]) -> MagicMock:
    doc = MagicMock()
    doc.metadata = metadata
    doc.__len__.return_value = len(pages)
    page_mocks = []
    for text in pages:
        page = MagicMock()
        page.get_text.return_value = text
        page_mocks.append(page)
    doc.__getitem__.side_effect = lambda i: page_mocks[i]
    return doc


@pytest.mark.unit
class TestExtractPdfNormalisation:
    def test_page_text_and_metadata_are_normalised(self) -> None:
        doc = _fake_doc(
            ["Spectral ﬁlters\x00", "ﬂow cytometry"],
            {"title": "Efﬁcient imaging", "author": "Schrödinger E"},
        )
        with patch("app.documents.processing.fitz.open", return_value=doc):
            meta = extract_pdf(Path("paper.pdf"))

        assert [p.text for p in meta.pages] == ["Spectral filters", "flow cytometry"]
        assert meta.title == "Efficient imaging"
        assert meta.authors == "Schrödinger E"
        assert meta.page_count == 2
