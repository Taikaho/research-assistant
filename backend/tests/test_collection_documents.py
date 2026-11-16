"""Tests for collection membership, document listing and permanent document deletion.

Covers the SPEC requirements that a document can belong to several collections,
that removing it from a collection keeps the document, and that permanently
deleting a document is a separate operation.
"""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.collections import service as collections_service
from app.database.models import Collection, Document, DocumentChunk, DocumentPage
from app.documents import service as documents_service

MINIMAL_PDF = (
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


def _fake_collection() -> MagicMock:
    fake = MagicMock(spec=Collection)
    fake.id = uuid.uuid4()
    fake.name = "Col"
    fake.description = None
    fake.created_at = fake.updated_at = datetime.now(UTC)
    return fake


def _fake_document() -> MagicMock:
    fake = MagicMock(spec=Document)
    fake.id = uuid.uuid4()
    fake.filename = "paper.pdf"
    fake.title = "Paper"
    fake.authors = None
    fake.publication_year = None
    fake.page_count = 1
    fake.file_path = "/nonexistent/paper.pdf"
    fake.file_hash = "abc"
    fake.processing_status = "completed"
    fake.created_at = datetime.now(UTC)
    return fake


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMembershipRouterUnit:
    async def test_list_collection_documents_404_when_collection_missing(
        self, client: AsyncClient
    ) -> None:
        with patch("app.collections.router.service.get_collection", return_value=None):
            resp = await client.get(f"/api/v1/collections/{uuid.uuid4()}/documents")
        assert resp.status_code == 404

    async def test_list_collection_documents_200(self, client: AsyncClient) -> None:
        doc = _fake_document()
        with (
            patch(
                "app.collections.router.service.get_collection",
                return_value=_fake_collection(),
            ),
            patch("app.collections.router.service.list_documents", return_value=[doc]),
        ):
            resp = await client.get(f"/api/v1/collections/{uuid.uuid4()}/documents")
        assert resp.status_code == 200
        assert [d["id"] for d in resp.json()] == [str(doc.id)]

    async def test_add_document_404_when_collection_missing(
        self, client: AsyncClient
    ) -> None:
        with patch("app.collections.router.service.get_collection", return_value=None):
            resp = await client.put(
                f"/api/v1/collections/{uuid.uuid4()}/documents/{uuid.uuid4()}"
            )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Collection not found."

    async def test_add_document_404_when_document_missing(
        self, client: AsyncClient
    ) -> None:
        with (
            patch(
                "app.collections.router.service.get_collection",
                return_value=_fake_collection(),
            ),
            patch("app.collections.router.documents_service.get_document", return_value=None),
            patch("app.collections.router.service.add_document") as mock_add,
        ):
            resp = await client.put(
                f"/api/v1/collections/{uuid.uuid4()}/documents/{uuid.uuid4()}"
            )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Document not found."
        mock_add.assert_not_called()

    async def test_add_document_204(self, client: AsyncClient) -> None:
        collection_id, document_id = uuid.uuid4(), uuid.uuid4()
        with (
            patch(
                "app.collections.router.service.get_collection",
                return_value=_fake_collection(),
            ),
            patch(
                "app.collections.router.documents_service.get_document",
                return_value=_fake_document(),
            ),
            patch("app.collections.router.service.add_document") as mock_add,
        ):
            resp = await client.put(
                f"/api/v1/collections/{collection_id}/documents/{document_id}"
            )
        assert resp.status_code == 204
        mock_add.assert_awaited_once()
        assert mock_add.await_args is not None
        assert mock_add.await_args.args[1:] == (collection_id, document_id)

    async def test_remove_document_404_when_not_member(self, client: AsyncClient) -> None:
        with (
            patch(
                "app.collections.router.service.get_collection",
                return_value=_fake_collection(),
            ),
            patch("app.collections.router.service.remove_document", return_value=False),
        ):
            resp = await client.delete(
                f"/api/v1/collections/{uuid.uuid4()}/documents/{uuid.uuid4()}"
            )
        assert resp.status_code == 404

    async def test_remove_document_204(self, client: AsyncClient) -> None:
        with (
            patch(
                "app.collections.router.service.get_collection",
                return_value=_fake_collection(),
            ),
            patch("app.collections.router.service.remove_document", return_value=True),
        ):
            resp = await client.delete(
                f"/api/v1/collections/{uuid.uuid4()}/documents/{uuid.uuid4()}"
            )
        assert resp.status_code == 204


@pytest.mark.unit
class TestDocumentRouterUnit:
    async def test_list_documents_200(self, client: AsyncClient) -> None:
        with patch("app.documents.router.service.list_documents", return_value=[]):
            resp = await client.get("/api/v1/documents")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_delete_document_404(self, client: AsyncClient) -> None:
        with patch("app.documents.router.service.get_document", return_value=None):
            resp = await client.delete(f"/api/v1/documents/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_delete_document_204(self, client: AsyncClient) -> None:
        with (
            patch(
                "app.documents.router.service.get_document",
                return_value=_fake_document(),
            ),
            patch("app.documents.router.service.delete_document") as mock_delete,
        ):
            resp = await client.delete(f"/api/v1/documents/{uuid.uuid4()}")
        assert resp.status_code == 204
        mock_delete.assert_awaited_once()

    async def test_upload_with_unknown_collection_returns_404_before_ingest(
        self, client: AsyncClient
    ) -> None:
        missing = uuid.uuid4()
        with (
            patch(
                "app.documents.router.collections_service.find_missing_collection_ids",
                return_value=[missing],
            ),
            patch("app.documents.router.service.ingest_document") as mock_ingest,
        ):
            resp = await client.post(
                "/api/v1/documents",
                files={"file": ("paper.pdf", b"%PDF", "application/pdf")},
                data={"collection_ids": [str(missing)]},
            )
        assert resp.status_code == 404
        assert str(missing) in resp.json()["detail"]
        mock_ingest.assert_not_called()

    async def test_upload_passes_collection_ids_to_ingest(
        self, client: AsyncClient
    ) -> None:
        ids = [uuid.uuid4(), uuid.uuid4()]
        with (
            patch(
                "app.documents.router.collections_service.find_missing_collection_ids",
                return_value=[],
            ),
            patch("app.documents.router.service.ingest_document") as mock_ingest,
        ):
            mock_ingest.return_value = (_fake_document(), True)
            resp = await client.post(
                "/api/v1/documents",
                files={"file": ("paper.pdf", b"%PDF", "application/pdf")},
                data={"collection_ids": [str(i) for i in ids]},
            )
        assert resp.status_code == 201
        assert mock_ingest.await_args is not None
        assert mock_ingest.await_args.args[2] == ids

    async def test_upload_invalid_collection_id_returns_422(
        self, client: AsyncClient
    ) -> None:
        resp = await client.post(
            "/api/v1/documents",
            files={"file": ("paper.pdf", b"%PDF", "application/pdf")},
            data={"collection_ids": ["not-a-uuid"]},
        )
        assert resp.status_code == 422


@pytest.mark.unit
class TestMembershipServiceUnit:
    async def test_find_missing_with_no_ids_skips_query(self, mock_db: AsyncMock) -> None:
        assert await collections_service.find_missing_collection_ids(mock_db, []) == []
        mock_db.execute.assert_not_called()

    async def test_find_missing_returns_unknown_ids_once_in_order(
        self, mock_db: AsyncMock
    ) -> None:
        known, unknown_a, unknown_b = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        mock_db.execute.return_value.scalars.return_value.all.return_value = [known]
        missing = await collections_service.find_missing_collection_ids(
            mock_db, [unknown_a, known, unknown_b, unknown_a]
        )
        assert missing == [unknown_a, unknown_b]

    async def test_link_document_with_no_collections_skips_query(
        self, mock_db: AsyncMock
    ) -> None:
        await collections_service.link_document(mock_db, [], uuid.uuid4())
        mock_db.execute.assert_not_called()

    async def test_duplicate_upload_links_existing_document(
        self, mock_db: AsyncMock
    ) -> None:
        existing = _fake_document()
        collection_id = uuid.uuid4()
        with (
            patch("app.documents.service._compute_hash", return_value=(b"x", "h")),
            patch("app.documents.service.get_document_by_hash", return_value=existing),
            patch("app.documents.service.link_document") as mock_link,
        ):
            doc, created = await documents_service.ingest_document(
                mock_db, MagicMock(), [collection_id]
            )
        assert (doc, created) == (existing, False)
        mock_link.assert_awaited_once_with(mock_db, [collection_id], existing.id)
        mock_db.commit.assert_awaited()

    async def test_delete_document_removes_file_after_commit(
        self, mock_db: AsyncMock, tmp_path: Path
    ) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF")
        doc = _fake_document()
        doc.file_path = str(pdf)

        async def assert_file_still_present() -> None:
            assert pdf.exists()

        mock_db.commit.side_effect = assert_file_still_present
        await documents_service.delete_document(mock_db, doc)
        mock_db.delete.assert_awaited_once_with(doc)
        assert not pdf.exists()

    async def test_delete_document_tolerates_missing_file(
        self, mock_db: AsyncMock, tmp_path: Path
    ) -> None:
        doc = _fake_document()
        doc.file_path = str(tmp_path / "already-gone.pdf")
        await documents_service.delete_document(mock_db, doc)
        mock_db.commit.assert_awaited_once()


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
    client: AsyncClient, collection_ids: list[str] | None = None, pdf: bytes = MINIMAL_PDF
) -> dict[str, object]:
    resp = await client.post(
        "/api/v1/documents",
        files={"file": ("paper.pdf", pdf, "application/pdf")},
        data={"collection_ids": collection_ids or []},
    )
    assert resp.status_code == 201, resp.text
    result: dict[str, object] = resp.json()
    return result


async def _collection_doc_ids(client: AsyncClient, collection_id: str) -> list[str]:
    resp = await client.get(f"/api/v1/collections/{collection_id}/documents")
    assert resp.status_code == 200
    return [d["id"] for d in resp.json()]


@pytest.mark.integration
@pytest.mark.usefixtures("storage")
class TestCollectionDocumentsIntegration:
    async def test_upload_into_multiple_collections(
        self, integration_client: AsyncClient
    ) -> None:
        a = await _create_collection(integration_client, "A")
        b = await _create_collection(integration_client, "B")
        doc = await _upload(integration_client, [a, b])

        assert await _collection_doc_ids(integration_client, a) == [doc["id"]]
        assert await _collection_doc_ids(integration_client, b) == [doc["id"]]

    async def test_upload_with_unknown_collection_creates_nothing(
        self, integration_client: AsyncClient
    ) -> None:
        resp = await integration_client.post(
            "/api/v1/documents",
            files={"file": ("paper.pdf", MINIMAL_PDF, "application/pdf")},
            data={"collection_ids": [str(uuid.uuid4())]},
        )
        assert resp.status_code == 404
        assert (await integration_client.get("/api/v1/documents")).json() == []

    async def test_duplicate_upload_adds_existing_document_to_new_collection(
        self, integration_client: AsyncClient
    ) -> None:
        a = await _create_collection(integration_client, "A")
        b = await _create_collection(integration_client, "B")
        first = await _upload(integration_client, [a])
        second = await _upload(integration_client, [a, b])

        assert first["id"] == second["id"]
        assert await _collection_doc_ids(integration_client, a) == [first["id"]]
        assert await _collection_doc_ids(integration_client, b) == [first["id"]]

    async def test_add_is_idempotent_and_remove_keeps_document(
        self, integration_client: AsyncClient
    ) -> None:
        a = await _create_collection(integration_client, "A")
        doc = await _upload(integration_client)
        url = f"/api/v1/collections/{a}/documents/{doc['id']}"

        assert (await integration_client.put(url)).status_code == 204
        assert (await integration_client.put(url)).status_code == 204
        assert await _collection_doc_ids(integration_client, a) == [doc["id"]]

        assert (await integration_client.delete(url)).status_code == 204
        assert await _collection_doc_ids(integration_client, a) == []
        assert (await integration_client.delete(url)).status_code == 404
        # The document itself still exists.
        assert (
            await integration_client.get(f"/api/v1/documents/{doc['id']}")
        ).status_code == 200

    async def test_add_unknown_document_returns_404(
        self, integration_client: AsyncClient
    ) -> None:
        a = await _create_collection(integration_client, "A")
        resp = await integration_client.put(
            f"/api/v1/collections/{a}/documents/{uuid.uuid4()}"
        )
        assert resp.status_code == 404

    async def test_deleting_collection_keeps_documents(
        self, integration_client: AsyncClient
    ) -> None:
        a = await _create_collection(integration_client, "A")
        doc = await _upload(integration_client, [a])

        assert (await integration_client.delete(f"/api/v1/collections/{a}")).status_code == 204
        assert (
            await integration_client.get(f"/api/v1/documents/{doc['id']}")
        ).status_code == 200

    async def test_list_documents(self, integration_client: AsyncClient) -> None:
        doc = await _upload(integration_client)
        resp = await integration_client.get("/api/v1/documents")
        assert resp.status_code == 200
        assert [d["id"] for d in resp.json()] == [doc["id"]]

    async def test_delete_document_removes_it_everywhere(
        self, integration_client: AsyncClient, db_session: AsyncSession, storage: Path
    ) -> None:
        a = await _create_collection(integration_client, "A")
        b = await _create_collection(integration_client, "B")
        doc = await _upload(integration_client, [a, b])
        stored = storage / f"{doc['id']}.pdf"
        assert stored.exists()

        resp = await integration_client.delete(f"/api/v1/documents/{doc['id']}")
        assert resp.status_code == 204

        assert (
            await integration_client.get(f"/api/v1/documents/{doc['id']}")
        ).status_code == 404
        assert await _collection_doc_ids(integration_client, a) == []
        assert await _collection_doc_ids(integration_client, b) == []
        assert not stored.exists()
        doc_id = uuid.UUID(str(doc["id"]))
        for model in (DocumentPage, DocumentChunk):
            count = await db_session.scalar(
                select(func.count()).select_from(model).where(model.document_id == doc_id)
            )
            assert count == 0, model.__tablename__
        # Collections survive.
        assert (await integration_client.get(f"/api/v1/collections/{a}")).status_code == 200
