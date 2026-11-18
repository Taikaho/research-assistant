from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.router import api_router
from app.core.config import settings
from app.llm.registry import get_llm_registry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Ensure the document storage directory exists on startup.
    Path(settings.storage_path).mkdir(parents=True, exist_ok=True)
    yield
    # Close provider HTTP clients only if the registry was created.
    if get_llm_registry.cache_info().currsize:
        await get_llm_registry().aclose()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered scientific document assistant",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(api_router)
