"""Unit and integration tests for the /api/v1/documents endpoints."""

import uuid
from datetime import UTC
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from app.database.models import Document


def _make_fake_document(**kwargs: object) -> MagicMock:
    from datetime import datetime

    fake = MagicMock(spec=Document)
    fake.id = kwargs.get("id", uuid.uuid4())
    fake.filename = kwargs.get("filename", "test.pdf")
    fake.title = kwargs.get("title", "Test Document")
    fake.authors = kwargs.get("authors", None)
    fake.publication_year = kwargs.get("publication_year", None)
    fake.page_count = kwargs.get("page_count", 1)
    fake.file_hash = kwargs.get("file_hash", "abc123")
    fake.processing_status = kwargs.get("processing_status", "completed")
    fake.created_at = kwargs.get("created_at", datetime.now(UTC))
    return fake


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDocumentRouterUnit:
    async def test_upload_non_pdf_returns_422(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/documents",
            files={"file": ("report.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 422

    async def test_upload_pdf_returns_201(self, client: AsyncClient) -> None:
        fake_doc = _make_fake_document()
        with patch("app.documents.router.service.ingest_document") as mock_ingest:
            mock_ingest.return_value = (fake_doc, True)
            response = await client.post(
                "/api/v1/documents",
                files={"file": ("paper.pdf", b"%PDF-1.4 content", "application/pdf")},
            )
        assert response.status_code == 201

    async def test_upload_duplicate_returns_201_with_existing_doc(
        self, client: AsyncClient
    ) -> None:
        fake_doc = _make_fake_document()
        with patch("app.documents.router.service.ingest_document") as mock_ingest:
            mock_ingest.return_value = (fake_doc, False)
            response = await client.post(
                "/api/v1/documents",
                files={"file": ("paper.pdf", b"%PDF-1.4 content", "application/pdf")},
            )
        assert response.status_code == 201

    async def test_upload_too_large_returns_413(self, client: AsyncClient) -> None:
        with patch("app.documents.router.service.ingest_document") as mock_ingest:
            mock_ingest.side_effect = ValueError("Upload exceeds maximum allowed size of 50 MB.")
            response = await client.post(
                "/api/v1/documents",
                files={"file": ("big.pdf", b"%PDF", "application/pdf")},
            )
        assert response.status_code == 413

    async def test_upload_corrupt_pdf_returns_422(self, client: AsyncClient) -> None:
        with patch("app.documents.router.service.ingest_document") as mock_ingest:
            mock_ingest.side_effect = Exception("cannot open broken file")
            response = await client.post(
                "/api/v1/documents",
                files={"file": ("broken.pdf", b"not a pdf", "application/pdf")},
            )
        assert response.status_code == 422

    async def test_get_document_404(self, client: AsyncClient) -> None:
        with patch("app.documents.router.service.get_document") as mock_get:
            mock_get.return_value = None
            response = await client.get(f"/api/v1/documents/{uuid.uuid4()}")
        assert response.status_code == 404

    async def test_get_document_200(self, client: AsyncClient) -> None:
        fake_doc = _make_fake_document()
        with patch("app.documents.router.service.get_document") as mock_get:
            mock_get.return_value = fake_doc
            response = await client.get(f"/api/v1/documents/{fake_doc.id}")
        assert response.status_code == 200


@pytest.mark.unit
class TestDocumentServiceUnit:
    async def test_deduplication_returns_existing_document(
        self, mock_db: AsyncMock
    ) -> None:
        from app.documents.service import ingest_document

        fake_doc = _make_fake_document(file_hash="deadbeef" * 8)

        with (
            patch("app.documents.service._compute_hash", return_value=(b"data", "deadbeef" * 8)),
            patch("app.documents.service.get_document_by_hash", return_value=fake_doc),
        ):
            doc, created = await ingest_document(mock_db, MagicMock())

        assert created is False
        assert doc is fake_doc

    async def test_processing_error_sets_failed_status(
        self, mock_db: AsyncMock, tmp_path: Path
    ) -> None:
        from app.documents.service import ingest_document

        mock_db.flush = AsyncMock()
        mock_db.refresh = AsyncMock()

        with (
            patch("app.documents.service._compute_hash", return_value=(b"data", "aabbcc")),
            patch("app.documents.service.get_document_by_hash", return_value=None),
            patch("app.documents.service.Path") as mock_path_cls,
            patch("app.documents.service.aiofiles.open"),
            patch("app.documents.service.extract_pdf", side_effect=Exception("bad pdf")),
            patch("app.documents.service.settings") as mock_settings,
        ):
            mock_settings.storage_path = str(tmp_path)
            mock_settings.max_upload_size_mb = 50
            mock_path_instance = MagicMock()
            mock_path_instance.__truediv__ = MagicMock(return_value=MagicMock())
            mock_path_cls.return_value = mock_path_instance

            with pytest.raises(Exception, match="bad pdf"):
                await ingest_document(mock_db, MagicMock(filename="x.pdf"))

        mock_db.commit.assert_awaited()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDocumentsIntegration:
    async def test_upload_minimal_pdf(
        self, integration_client: AsyncClient, tmp_path: Path
    ) -> None:
        """Upload a real minimal PDF and verify the document record is created."""
        # Minimal valid PDF bytes (single empty page)
        minimal_pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF"
        )

        with patch("app.documents.service.settings") as mock_settings:
            mock_settings.storage_path = str(tmp_path)
            mock_settings.max_upload_size_mb = 50
            mock_settings.chunk_size = 2000
            mock_settings.chunk_overlap = 400

            resp = await integration_client.post(
                "/api/v1/documents",
                files={"file": ("paper.pdf", minimal_pdf, "application/pdf")},
            )

        assert resp.status_code == 201
        data = resp.json()
        assert data["filename"] == "paper.pdf"
        assert data["processing_status"] == "completed"
        doc_id = data["id"]

        get_resp = await integration_client.get(f"/api/v1/documents/{doc_id}")
        assert get_resp.status_code == 200

    async def test_duplicate_upload_returns_existing(
        self, integration_client: AsyncClient, tmp_path: Path
    ) -> None:
        minimal_pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF"
        )

        with patch("app.documents.service.settings") as mock_settings:
            mock_settings.storage_path = str(tmp_path)
            mock_settings.max_upload_size_mb = 50
            mock_settings.chunk_size = 2000
            mock_settings.chunk_overlap = 400

            resp1 = await integration_client.post(
                "/api/v1/documents",
                files={"file": ("paper.pdf", minimal_pdf, "application/pdf")},
            )
            resp2 = await integration_client.post(
                "/api/v1/documents",
                files={"file": ("paper_copy.pdf", minimal_pdf, "application/pdf")},
            )

        assert resp1.status_code == 201
        assert resp2.status_code == 201
        assert resp1.json()["id"] == resp2.json()["id"]
