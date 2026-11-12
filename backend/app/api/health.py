from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.connection import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """
    Returns application health status including database connectivity.

    Always returns HTTP 200. Callers should inspect the 'database' field:
    - "ok"    — database is reachable
    - "error" — database is unreachable; the application is still running
    """
    db_status = "ok"
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"

    return {
        "status": "ok",
        "version": settings.app_version,
        "database": db_status,
    }
