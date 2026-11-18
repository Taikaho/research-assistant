"""Tests for the endpoints the frontend relies on (Phase 5).

Covers serving the original PDF, extracted page text, reprocessing, collection
membership on documents and document counts on collections.
"""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from httpx import AsyncClient

from app.database.models import Document
from app.documents import service as documents_service


def _fake_document(file_path: str = "/nonexistent/paper.pdf") -> MagicMock:
    fake = MagicMock(spec=Document)
    fake.id = uuid.uuid4()
    fake.filename = "paper.pdf"
    fake.title = "Paper"
    fake.authors = None
    fake.publication_year = None
    fake.page_count = 1
    fake.file_path = file_path
    fake.file_hash = "abc"
    fake.processing_status = "completed"
    fake.created_at = datetime.now(UTC)
    return fake


def _text_pdf(*pages: str) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data: bytes = doc.tobytes()
    doc.close()
    return data


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFrontendSupportUnit:
    async def test_file_404_when_document_missing(self, client: AsyncClient) -> None:
        with patch("app.documents.router.service.get_document", return_value=None):
            resp = await client.get(f"/api/v1/documents/{uuid.uuid4()}/file")
        assert resp.status_code == 404

    async def test_file_404_when_stored_pdf_missing(self, client: AsyncClient) -> None:
        with patch("app.documents.router.service.get_document", return_value=_fake_document()):
            resp = await client.get(f"/api/v1/documents/{uuid.uuid4()}/file")
        assert resp.status_code == 404
        assert "missing" in resp.json()["detail"]

    async def test_file_served_inline_as_pdf(
        self, client: AsyncClient, tmp_path: Path
    ) -> None:
        pdf = tmp_path / "stored.pdf"
        pdf.write_bytes(b"%PDF-1.4 test")
        with patch(
            "app.documents.router.service.get_document",
            return_value=_fake_document(str(pdf)),
        ):
            resp = await client.get(f"/api/v1/documents/{uuid.uuid4()}/file")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        assert resp.headers["content-disposition"].startswith("inline")
        assert resp.content == b"%PDF-1.4 test"

    async def test_page_404_when_page_missing(self, client: AsyncClient) -> None:
        with (
            patch("app.documents.router.service.get_document", return_value=_fake_document()),
            patch("app.documents.router.service.get_page", return_value=None),
        ):
            resp = await client.get(f"/api/v1/documents/{uuid.uuid4()}/pages/3")
        assert resp.status_code == 404

    async def test_reprocess_409_when_file_missing(self, client: AsyncClient) -> None:
        with patch("app.documents.router.service.get_document", return_value=_fake_document()):
            resp = await client.post(f"/api/v1/documents/{uuid.uuid4()}/reprocess")
        assert resp.status_code == 409

    async def test_reprocess_503_when_embedder_unavailable(self, client: AsyncClient) -> None:
        with (
            patch("app.documents.router.service.get_document", return_value=_fake_document()),
            patch(
                "app.documents.router.service.reprocess_document",
                side_effect=documents_service.EmbeddingUnavailableError("no model"),
            ),
        ):
            resp = await client.post(f"/api/v1/documents/{uuid.uuid4()}/reprocess")
        assert resp.status_code == 503

    async def test_document_response_includes_collection_ids(
        self, client: AsyncClient
    ) -> None:
        doc = _fake_document()
        col = uuid.uuid4()
        with (
            patch("app.documents.router.service.list_documents", return_value=[doc]),
            patch(
                "app.documents.router.service.collection_ids_by_document",
                return_value={doc.id: [col]},
            ),
        ):
            resp = await client.get("/api/v1/documents")
        assert resp.status_code == 200
        assert resp.json()[0]["collection_ids"] == [str(col)]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[Path]:
    with patch("app.documents.service.settings") as mock_settings:
        mock_settings.storage_path = str(tmp_path)
        mock_settings.max_upload_size_mb = 50
        mock_settings.chunk_size = 2000
        mock_settings.chunk_overlap = 400
        yield tmp_path


async def _create_collection(client: AsyncClient, name: str) -> str:
    resp = await client.post("/api/v1/collections", json={"name": name})
    assert resp.status_code == 201
    return str(resp.json()["id"])


async def _upload(
    client: AsyncClient, pdf: bytes, collection_ids: list[str] | None = None
) -> dict[str, object]:
    resp = await client.post(
        "/api/v1/documents",
        files={"file": ("paper.pdf", pdf, "application/pdf")},
        data={"collection_ids": collection_ids or []},
    )
    assert resp.status_code == 201, resp.text
    result: dict[str, object] = resp.json()
    return result


@pytest.mark.integration
@pytest.mark.usefixtures("storage")
class TestFrontendSupportIntegration:
    async def test_file_and_page_text(self, integration_client: AsyncClient) -> None:
        pdf = _text_pdf("First page about spectra.", "Second page about classifiers.")
        doc = await _upload(integration_client, pdf)
        # No title metadata: the uploaded filename is used, not the stored file's UUID name.
        assert doc["title"] == "paper"

        file_resp = await integration_client.get(f"/api/v1/documents/{doc['id']}/file")
        assert file_resp.status_code == 200
        assert file_resp.content == pdf

        page = await integration_client.get(f"/api/v1/documents/{doc['id']}/pages/1")
        assert page.status_code == 200
        body = page.json()
        assert body["pdf_page_number"] == 1
        assert "classifiers" in body["text"]

        missing = await integration_client.get(f"/api/v1/documents/{doc['id']}/pages/7")
        assert missing.status_code == 404

    async def test_membership_and_counts(self, integration_client: AsyncClient) -> None:
        a = await _create_collection(integration_client, "A")
        b = await _create_collection(integration_client, "B")
        doc = await _upload(integration_client, _text_pdf("Some text."), [a, b])
        assert sorted(doc["collection_ids"]) == sorted([a, b])  # type: ignore[type-var]

        listed = (await integration_client.get("/api/v1/documents")).json()
        assert sorted(listed[0]["collection_ids"]) == sorted([a, b])

        collections = (await integration_client.get("/api/v1/collections")).json()
        counts = {c["id"]: c["document_count"] for c in collections}
        assert counts == {a: 1, b: 1}

        await integration_client.delete(f"/api/v1/collections/{b}/documents/{doc['id']}")
        assert (await integration_client.get(f"/api/v1/collections/{b}")).json()[
            "document_count"
        ] == 0

    async def test_reprocess_replaces_chunks_and_keeps_membership(
        self, integration_client: AsyncClient
    ) -> None:
        a = await _create_collection(integration_client, "A")
        doc = await _upload(integration_client, _text_pdf("Reprocess me."), [a])

        resp = await integration_client.post(f"/api/v1/documents/{doc['id']}/reprocess")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["processing_status"] == "completed"
        assert body["collection_ids"] == [a]

        search = await integration_client.post(
            "/api/v1/search", json={"query": "reprocess", "collection_ids": [a]}
        )
        assert search.status_code == 200
        results = search.json()["results"]
        assert len(results) == 1
        assert "Reprocess me." in results[0]["text"]
