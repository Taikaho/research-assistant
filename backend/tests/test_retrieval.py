"""Tests for semantic retrieval: ranking, collection scoping and de-duplication."""

import uuid
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from httpx import AsyncClient
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Document
from app.retrieval.service import RetrievedChunk, jaccard, remove_near_duplicates
from tests.fakes import FakeEmbeddingProvider


def _candidate(text: str, score: float) -> RetrievedChunk:
    chunk = MagicMock()
    chunk.text = text
    return RetrievedChunk(chunk=chunk, document=MagicMock(), score=score)


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeduplicationUnit:
    def test_jaccard_identical_and_disjoint(self) -> None:
        a = {("a", "b", "c")}
        assert jaccard(a, a) == 1.0
        assert jaccard(a, {("x", "y", "z")}) == 0.0

    def test_near_duplicates_keep_highest_score(self) -> None:
        text = "support vector machines find the maximum margin separating hyperplane"
        candidates = [
            _candidate(text, 0.9),
            _candidate(text.upper() + "  ", 0.85),  # same words, different case/spacing
            _candidate("k nearest neighbours votes among the closest samples", 0.8),
        ]
        kept = remove_near_duplicates(candidates, threshold=0.9, limit=10)
        assert [c.score for c in kept] == [0.9, 0.8]

    def test_limit_applies_after_deduplication(self) -> None:
        candidates = [_candidate("same text here again", 0.9)] * 3 + [
            _candidate(f"distinct passage number {i}", 0.5) for i in range(3)
        ]
        kept = remove_near_duplicates(candidates, threshold=0.9, limit=3)
        assert len(kept) == 3
        assert kept[0].score == 0.9
        assert all(c.score == 0.5 for c in kept[1:])


@pytest.mark.unit
class TestSearchRouterUnit:
    async def test_requires_at_least_one_collection(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/search", json={"query": "q", "collection_ids": []})
        assert resp.status_code == 422

    async def test_rejects_blank_query(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/search", json={"query": "   ", "collection_ids": [str(uuid.uuid4())]}
        )
        assert resp.status_code == 422

    async def test_rejects_top_k_above_maximum(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/search",
            json={"query": "q", "collection_ids": [str(uuid.uuid4())], "top_k": 10_000},
        )
        assert resp.status_code == 422

    async def test_unknown_collection_returns_404(self, client: AsyncClient) -> None:
        missing = uuid.uuid4()
        with (
            patch(
                "app.retrieval.router.find_missing_collection_ids", return_value=[missing]
            ),
            patch("app.retrieval.router.service.search") as mock_search,
        ):
            resp = await client.post(
                "/api/v1/search", json={"query": "q", "collection_ids": [str(missing)]}
            )
        assert resp.status_code == 404
        mock_search.assert_not_called()

    async def test_model_failure_returns_503(self, client: AsyncClient) -> None:
        with (
            patch("app.retrieval.router.find_missing_collection_ids", return_value=[]),
            patch(
                "app.retrieval.router.service.search", side_effect=OSError("model not found")
            ),
        ):
            resp = await client.post(
                "/api/v1/search", json={"query": "q", "collection_ids": [str(uuid.uuid4())]}
            )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

LDA_TEXT = (
    "Linear discriminant analysis is a supervised method that projects spectra onto "
    "axes maximising the separation between classes."
)
KNN_TEXT = (
    "The k nearest neighbours classifier assigns a label by majority vote among the "
    "closest training samples in feature space."
)
SOIL_TEXT = (
    "Soil moisture sensors were deployed across the field and sampled every hour "
    "using low power wireless nodes."
)


def _pdf(*pages: str) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text)
    data: bytes = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[Path]:
    with patch("app.documents.service.settings") as mock_settings:
        mock_settings.storage_path = str(tmp_path)
        mock_settings.max_upload_size_mb = 50
        mock_settings.chunk_size = 2000
        mock_settings.chunk_overlap = 400
        yield tmp_path


async def _collection(client: AsyncClient, name: str) -> str:
    resp = await client.post("/api/v1/collections", json={"name": name})
    assert resp.status_code == 201
    return str(resp.json()["id"])


async def _upload(client: AsyncClient, pdf: bytes, collection_ids: list[str]) -> str:
    resp = await client.post(
        "/api/v1/documents",
        files={"file": ("paper.pdf", pdf, "application/pdf")},
        data={"collection_ids": collection_ids},
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["processing_status"] == "completed"
    return str(resp.json()["id"])


async def _search(
    client: AsyncClient, query: str, collection_ids: list[str], **extra: object
) -> dict:  # type: ignore[type-arg]
    resp = await client.post(
        "/api/v1/search", json={"query": query, "collection_ids": collection_ids, **extra}
    )
    assert resp.status_code == 200, resp.text
    body: dict = resp.json()  # type: ignore[type-arg]
    return body


@pytest.mark.integration
@pytest.mark.usefixtures("storage")
class TestRetrievalIntegration:
    async def test_relevant_passage_ranks_first_with_source_metadata(
        self, integration_client: AsyncClient
    ) -> None:
        ml = await _collection(integration_client, "ML")
        doc_id = await _upload(integration_client, _pdf(SOIL_TEXT, LDA_TEXT), [ml])
        await _upload(integration_client, _pdf(KNN_TEXT), [ml])

        body = await _search(integration_client, "linear discriminant analysis classes", [ml])
        top = body["results"][0]
        assert top["document_id"] == doc_id
        assert "discriminant" in top["text"]
        assert 1 in top["pdf_page_numbers"]  # LDA text is on the second page
        assert top["chunk_id"] and top["document_title"] and top["filename"] == "paper.pdf"
        scores = [r["score"] for r in body["results"]]
        assert scores == sorted(scores, reverse=True)
        assert body["embedding_model"]["name"] == "fake-model"
        assert body["unindexed_document_count"] == 0

    async def test_documents_outside_selected_collections_never_appear(
        self, integration_client: AsyncClient
    ) -> None:
        a = await _collection(integration_client, "A")
        b = await _collection(integration_client, "B")
        c = await _collection(integration_client, "C")
        lda = await _upload(integration_client, _pdf(LDA_TEXT), [a])
        knn = await _upload(integration_client, _pdf(KNN_TEXT), [b])
        soil = await _upload(integration_client, _pdf(SOIL_TEXT), [c])

        query = "linear discriminant analysis classes"
        only_b = await _search(integration_client, query, [b], top_k=50, min_score=-1)
        assert {r["document_id"] for r in only_b["results"]} == {knn}

        a_and_c = await _search(integration_client, query, [a, c], top_k=50, min_score=-1)
        assert {r["document_id"] for r in a_and_c["results"]} == {lda, soil}

    async def test_document_in_several_selected_collections_is_returned_once(
        self, integration_client: AsyncClient
    ) -> None:
        a = await _collection(integration_client, "A")
        b = await _collection(integration_client, "B")
        await _upload(integration_client, _pdf(LDA_TEXT), [a, b])

        body = await _search(integration_client, "discriminant", [a, b], top_k=10)
        chunk_ids = [r["chunk_id"] for r in body["results"]]
        assert len(chunk_ids) == len(set(chunk_ids)) == 1

    async def test_top_k_limits_results(self, integration_client: AsyncClient) -> None:
        col = await _collection(integration_client, "Many")
        for text in (LDA_TEXT, KNN_TEXT, SOIL_TEXT):
            await _upload(integration_client, _pdf(text), [col])
        body = await _search(integration_client, "classes", [col], top_k=2, min_score=-1)
        assert len(body["results"]) == 2

    async def test_removed_document_is_no_longer_in_scope(
        self, integration_client: AsyncClient
    ) -> None:
        col = await _collection(integration_client, "Scope")
        doc = await _upload(integration_client, _pdf(LDA_TEXT), [col])
        resp = await integration_client.delete(f"/api/v1/collections/{col}/documents/{doc}")
        assert resp.status_code == 204
        body = await _search(integration_client, "discriminant", [col], min_score=-1)
        assert body["results"] == []

    async def test_incomplete_documents_are_excluded(
        self, integration_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        col = await _collection(integration_client, "Status")
        doc = await _upload(integration_client, _pdf(LDA_TEXT), [col])
        await db_session.execute(
            update(Document)
            .where(Document.id == uuid.UUID(doc))
            .values(processing_status="processing")
        )
        body = await _search(integration_client, "discriminant", [col], min_score=-1)
        assert body["results"] == []

    async def test_chunks_from_another_model_are_not_mixed(
        self, integration_client: AsyncClient, fake_embedder: FakeEmbeddingProvider
    ) -> None:
        col = await _collection(integration_client, "Models")
        await _upload(integration_client, _pdf(LDA_TEXT), [col])

        # Switch the configured model: same name, new version and dimension.
        fake_embedder._version = "2"
        fake_embedder._dimensions = 32
        body = await _search(integration_client, "discriminant", [col], min_score=-1)
        assert body["results"] == []
        assert body["unindexed_document_count"] == 1
        assert body["embedding_model"]["dimensions"] == 32

        reindex = await integration_client.post("/api/v1/embeddings/reindex")
        assert reindex.json()["chunks_reindexed"] == 1
        body = await _search(integration_client, "discriminant", [col])
        assert len(body["results"]) == 1
        assert body["unindexed_document_count"] == 0

    async def test_min_score_filters_weak_matches(self, integration_client: AsyncClient) -> None:
        col = await _collection(integration_client, "Threshold")
        await _upload(integration_client, _pdf(SOIL_TEXT), [col])
        body = await _search(
            integration_client, "discriminant analysis spectra", [col], min_score=0.99
        )
        assert body["results"] == []

    async def test_deleted_document_chunks_are_removed(
        self, integration_client: AsyncClient
    ) -> None:
        col = await _collection(integration_client, "Delete")
        doc = await _upload(integration_client, _pdf(LDA_TEXT), [col])
        assert (await integration_client.delete(f"/api/v1/documents/{doc}")).status_code == 204
        status = (await integration_client.get("/api/v1/embeddings/status")).json()
        assert status["total_chunks"] == 0
