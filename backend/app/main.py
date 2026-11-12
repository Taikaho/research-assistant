from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.api.health import router as health_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup actions (database migrations are run via `alembic upgrade head`
    # before the process starts; see Dockerfile CMD and docker-compose.yml)
    yield
    # Shutdown actions


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered scientific document assistant",
    lifespan=lifespan,
)

app.include_router(health_router)
