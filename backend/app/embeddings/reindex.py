"""Command-line re-indexing: `python -m app.embeddings.reindex`.

Embeds every chunk that has no embedding from the configured model. Use after
changing EMBEDDING_MODEL_NAME, or for documents ingested before Phase 3.
"""

import asyncio
import logging

from app.database.connection import AsyncSessionLocal, engine
from app.embeddings.service import create_embedding_provider, reindex


async def _main() -> None:
    provider = create_embedding_provider()
    try:
        async with AsyncSessionLocal() as db:
            result = await reindex(db, provider)
    finally:
        await engine.dispose()
    print(
        f"Model {result.model.name}@{result.model.version} ({result.model.dimensions} dims): "
        f"re-indexed {result.chunks} chunks in {result.documents} documents."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main())
