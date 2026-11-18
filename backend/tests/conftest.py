"""
pytest configuration and shared fixtures.

Marker conventions
------------------
@pytest.mark.unit        — no external services required (default when running plain `pytest`)
@pytest.mark.integration — requires a running PostgreSQL test database

Test database isolation
-----------------------
Integration tests use TEST_DATABASE_URL exclusively. The fixture verifies that
this URL contains "test" to prevent accidental writes to the development database.
The test database is started via:

    docker compose -f docker-compose.test.yml up -d

and migrated via:

    DATABASE_URL=$TEST_DATABASE_URL alembic upgrade head

Each integration test session creates all tables (idempotent) and each test
runs inside a transaction that is rolled back on teardown, leaving the database
clean for the next test.
"""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.database.connection import get_db
from app.database.models import Base
from app.embeddings.service import get_embedding_provider
from app.llm.registry import LLMRegistry, get_llm_registry
from app.main import app
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider

# ---------------------------------------------------------------------------
# Embedding provider  (never loads a real model unless a test asks for one)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_embedder() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


# ---------------------------------------------------------------------------
# LLM providers  (never calls Ollama or a cloud API)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_llm() -> FakeLLMProvider:
    """Local fake provider; the registry default."""
    return FakeLLMProvider(name="fake-llm", is_local=True)


@pytest.fixture
def fake_cloud_llm() -> FakeLLMProvider:
    return FakeLLMProvider(name="fake-cloud", is_local=False, default_model="cloud-model")


@pytest.fixture
def llm_registry(fake_llm: FakeLLMProvider, fake_cloud_llm: FakeLLMProvider) -> LLMRegistry:
    return LLMRegistry([fake_llm, fake_cloud_llm], default_provider="fake-llm")


# ---------------------------------------------------------------------------
# Unit test fixtures  (no database required)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db() -> AsyncMock:
    """Mock AsyncSession for unit tests that do not touch the database."""
    mock = AsyncMock(spec=AsyncSession)
    mock.execute = AsyncMock(return_value=MagicMock())
    return mock


@pytest.fixture
async def client(
    mock_db: AsyncMock, fake_embedder: FakeEmbeddingProvider, llm_registry: LLMRegistry
) -> AsyncGenerator[AsyncClient, None]:
    """Test client with the database dependency replaced by a mock."""

    async def override_get_db() -> AsyncGenerator[AsyncMock, None]:
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_embedding_provider] = lambda: fake_embedder
    app.dependency_overrides[get_llm_registry] = lambda: llm_registry
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
async def client_db_error(
    fake_embedder: FakeEmbeddingProvider, llm_registry: LLMRegistry
) -> AsyncGenerator[AsyncClient, None]:
    """Test client whose database dependency always raises an exception."""

    async def override_get_db() -> AsyncGenerator[AsyncMock, None]:
        mock = AsyncMock(spec=AsyncSession)
        mock.execute = AsyncMock(side_effect=Exception("DB unavailable"))
        yield mock

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_embedding_provider] = lambda: fake_embedder
    app.dependency_overrides[get_llm_registry] = lambda: llm_registry
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Integration test fixtures  (require a running PostgreSQL test database)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """
    Return the test database URL and verify it is not the development database.

    The check that the URL contains 'test' is a safeguard against
    misconfiguration. Set TEST_DATABASE_URL in your environment or .env file.
    """
    url = settings.test_database_url
    assert "test" in url.lower(), (
        "TEST_DATABASE_URL must contain 'test' to prevent accidental writes to "
        f"the development database. Got: {url!r}"
    )
    assert url != settings.database_url, (
        "TEST_DATABASE_URL must differ from DATABASE_URL. "
        "Integration tests must never use the development database."
    )
    return url


@pytest.fixture
async def db_session(test_database_url: str) -> AsyncGenerator[AsyncSession, None]:
    """
    Function-scoped database session for integration tests.

    Creates a fresh engine per test (avoids asyncio event-loop sharing issues
    between session-scoped and function-scoped fixtures). Each test runs inside
    a transaction that is rolled back on teardown, keeping the test database
    clean without truncating tables.

    Tables are created idempotently on first use; running `alembic upgrade head`
    beforehand is recommended but not required.
    """
    engine = create_async_engine(test_database_url, echo=False, pool_pre_ping=True)

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Drop and recreate so the schema always matches current ORM models,
        # even when migrations have added constraints since the last test run.
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    async with engine.connect() as conn:
        await conn.begin()
        session_factory = async_sessionmaker(
            bind=conn,
            class_=AsyncSession,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        async with session_factory() as session:
            yield session
        await conn.rollback()

    await engine.dispose()


@pytest.fixture
async def integration_client(
    db_session: AsyncSession, fake_embedder: FakeEmbeddingProvider, llm_registry: LLMRegistry
) -> AsyncGenerator[AsyncClient, None]:
    """Test client wired to the real test database session and a fake embedder."""

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_embedding_provider] = lambda: fake_embedder
    app.dependency_overrides[get_llm_registry] = lambda: llm_registry
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()
