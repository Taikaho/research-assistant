"""
Tests for GET /health.

Unit tests  — run by default (`pytest`)
Integration tests — run with `pytest -m integration`
"""

import pytest
from httpx import AsyncClient

from app.core.config import settings


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_health_returns_200(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.unit
async def test_health_response_has_required_fields(client: AsyncClient) -> None:
    response = await client.get("/health")
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "database" in data


@pytest.mark.unit
async def test_health_status_is_ok(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.json()["status"] == "ok"


@pytest.mark.unit
async def test_health_version_matches_config(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.json()["version"] == settings.app_version


@pytest.mark.unit
async def test_health_database_ok_when_reachable(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.json()["database"] == "ok"


@pytest.mark.unit
async def test_health_still_200_when_database_unreachable(
    client_db_error: AsyncClient,
) -> None:
    """The endpoint must return 200 even when the database is down."""
    response = await client_db_error.get("/health")
    assert response.status_code == 200


@pytest.mark.unit
async def test_health_database_error_reported(client_db_error: AsyncClient) -> None:
    response = await client_db_error.get("/health")
    assert response.json()["database"] == "error"


@pytest.mark.unit
async def test_health_status_ok_even_when_database_unreachable(
    client_db_error: AsyncClient,
) -> None:
    """Application status is 'ok' regardless of DB state."""
    response = await client_db_error.get("/health")
    assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Integration tests  (require a running PostgreSQL test database)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_health_database_connected(integration_client: AsyncClient) -> None:
    response = await integration_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "ok"
    assert data["version"] == settings.app_version
