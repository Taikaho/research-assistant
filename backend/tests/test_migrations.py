"""
Alembic migration tests.

Unit tests check the revision graph without a database: every down_revision
must resolve, and there must be exactly one base and one head. This catches
broken chains (e.g. a down_revision pointing at a filename prefix instead of
the real revision ID).

The integration test applies the full chain to a throwaway database:
upgrade head -> downgrade base -> upgrade head, then verifies the migrated
schema matches the ORM models. It uses a separate scratch database so it never
disturbs the shared integration test database.
"""

from collections.abc import Generator
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.database.models import Base

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def _alembic_config() -> Config:
    # Built without alembic.ini so env.py skips fileConfig() and leaves
    # pytest's logging setup alone.
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    return cfg


# ---------------------------------------------------------------------------
# Unit: revision graph
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_revision_chain_resolves() -> None:
    script = ScriptDirectory.from_config(_alembic_config())
    # walk_revisions raises if any down_revision points at an unknown ID.
    revisions = list(script.walk_revisions())
    assert revisions, "No migrations found"


@pytest.mark.unit
def test_single_head_and_base() -> None:
    script = ScriptDirectory.from_config(_alembic_config())
    assert len(script.get_heads()) == 1, f"Multiple heads: {script.get_heads()}"
    assert len(script.get_bases()) == 1, f"Multiple bases: {script.get_bases()}"


# ---------------------------------------------------------------------------
# Integration: apply migrations to a real database
# ---------------------------------------------------------------------------


@pytest.fixture
def scratch_database_url(test_database_url: str) -> Generator[str, None, None]:
    """Create an empty database next to the test DB and drop it afterwards."""
    base_url = make_url(test_database_url).set(drivername="postgresql+psycopg2")
    scratch_name = f"{base_url.database}_migrations"
    admin_engine = create_engine(
        base_url.set(database="postgres"), isolation_level="AUTOCOMMIT"
    )

    def drop() -> None:
        with admin_engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{scratch_name}" WITH (FORCE)'))

    drop()
    with admin_engine.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{scratch_name}"'))
    try:
        yield base_url.set(database=scratch_name).render_as_string(hide_password=False)
    finally:
        drop()
        admin_engine.dispose()


@pytest.mark.integration
def test_upgrade_downgrade_roundtrip(
    scratch_database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # env.py reads DATABASE_URL directly.
    monkeypatch.setenv("DATABASE_URL", scratch_database_url)
    cfg = _alembic_config()

    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")

    engine = create_engine(scratch_database_url)
    try:
        with engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    finally:
        engine.dispose()
    assert diff == [], f"Migrated schema differs from ORM models: {diff}"
