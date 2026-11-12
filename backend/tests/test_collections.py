"""Unit and integration tests for the /api/v1/collections endpoints."""

import uuid
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.exc import IntegrityError

from app.collections import service
from app.collections.schemas import CollectionCreate
from app.database.models import Collection

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCollectionServiceUnit:
    async def test_create_collection_calls_db(self, mock_db: AsyncMock) -> None:
        mock_db.refresh = AsyncMock()
        data = CollectionCreate(name="ML Papers")
        with patch("app.collections.service.Collection") as mock_coll_cls:
            mock_coll_cls.return_value = MagicMock(spec=Collection)
            await service.create_collection(mock_db, data)
        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited_once()

    async def test_create_collection_rollback_on_integrity_error(
        self, mock_db: AsyncMock
    ) -> None:
        mock_db.commit = AsyncMock(side_effect=IntegrityError("", {}, Exception()))
        mock_db.rollback = AsyncMock()
        data = CollectionCreate(name="Duplicate")
        with pytest.raises(IntegrityError):
            await service.create_collection(mock_db, data)
        mock_db.rollback.assert_awaited_once()


@pytest.mark.unit
class TestCollectionRouterUnit:
    async def test_create_collection_returns_201(self, client: AsyncClient) -> None:
        with patch("app.collections.router.service.create_collection") as mock_create:
            fake = MagicMock(spec=Collection)
            fake.id = uuid.uuid4()
            fake.name = "Test"
            fake.description = None
            from datetime import datetime

            now = datetime.now(UTC)
            fake.created_at = now
            fake.updated_at = now
            mock_create.return_value = fake
            response = await client.post("/api/v1/collections", json={"name": "Test"})
        assert response.status_code == 201

    async def test_create_collection_409_on_duplicate(self, client: AsyncClient) -> None:
        with patch("app.collections.router.service.create_collection") as mock_create:
            mock_create.side_effect = IntegrityError("", {}, Exception())
            response = await client.post("/api/v1/collections", json={"name": "Dupe"})
        assert response.status_code == 409

    async def test_list_collections_returns_200(self, client: AsyncClient) -> None:
        with patch("app.collections.router.service.list_collections") as mock_list:
            mock_list.return_value = []
            response = await client.get("/api/v1/collections")
        assert response.status_code == 200
        assert response.json() == []

    async def test_get_collection_404_when_missing(self, client: AsyncClient) -> None:
        with patch("app.collections.router.service.get_collection") as mock_get:
            mock_get.return_value = None
            response = await client.get(f"/api/v1/collections/{uuid.uuid4()}")
        assert response.status_code == 404

    async def test_delete_collection_404_when_missing(self, client: AsyncClient) -> None:
        with patch("app.collections.router.service.get_collection") as mock_get:
            mock_get.return_value = None
            response = await client.delete(f"/api/v1/collections/{uuid.uuid4()}")
        assert response.status_code == 404

    async def test_create_collection_empty_name_422(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/collections", json={"name": ""})
        assert response.status_code == 422

    async def test_patch_collection_404_when_missing(self, client: AsyncClient) -> None:
        with patch("app.collections.router.service.get_collection") as mock_get:
            mock_get.return_value = None
            response = await client.patch(
                f"/api/v1/collections/{uuid.uuid4()}", json={"name": "New"}
            )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCollectionsIntegration:
    async def test_create_and_retrieve_collection(
        self, integration_client: AsyncClient
    ) -> None:
        resp = await integration_client.post(
            "/api/v1/collections", json={"name": "AI Research", "description": "AI papers"}
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "AI Research"
        assert data["description"] == "AI papers"
        collection_id = data["id"]

        resp2 = await integration_client.get(f"/api/v1/collections/{collection_id}")
        assert resp2.status_code == 200
        assert resp2.json()["id"] == collection_id

    async def test_duplicate_name_returns_409(
        self, integration_client: AsyncClient
    ) -> None:
        await integration_client.post("/api/v1/collections", json={"name": "Unique Name"})
        resp = await integration_client.post("/api/v1/collections", json={"name": "Unique Name"})
        assert resp.status_code == 409

    async def test_list_collections(self, integration_client: AsyncClient) -> None:
        await integration_client.post("/api/v1/collections", json={"name": "ColA"})
        await integration_client.post("/api/v1/collections", json={"name": "ColB"})
        resp = await integration_client.get("/api/v1/collections")
        assert resp.status_code == 200
        names = [c["name"] for c in resp.json()]
        assert "ColA" in names
        assert "ColB" in names

    async def test_delete_collection(self, integration_client: AsyncClient) -> None:
        resp = await integration_client.post(
            "/api/v1/collections", json={"name": "ToDelete"}
        )
        collection_id = resp.json()["id"]
        del_resp = await integration_client.delete(f"/api/v1/collections/{collection_id}")
        assert del_resp.status_code == 204
        get_resp = await integration_client.get(f"/api/v1/collections/{collection_id}")
        assert get_resp.status_code == 404

    async def test_update_collection_name(self, integration_client: AsyncClient) -> None:
        resp = await integration_client.post("/api/v1/collections", json={"name": "Old Name"})
        collection_id = resp.json()["id"]
        patch_resp = await integration_client.patch(
            f"/api/v1/collections/{collection_id}", json={"name": "New Name"}
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["name"] == "New Name"
