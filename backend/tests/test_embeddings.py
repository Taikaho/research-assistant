"""Tests for the embedding provider interface, registry and re-indexing."""

import sys
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Document, DocumentChunk, EmbeddingModel
from app.embeddings import service
from app.embeddings.sentence_transformer import SentenceTransformerProvider
from tests.fakes import FakeEmbeddingProvider

# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProviderUnit:
    def test_fake_provider_vectors_are_normalised(self) -> None:
        provider = FakeEmbeddingProvider(dimensions=16)
        [vector] = provider.embed_documents(["spectral imaging of plants"])
        assert len(vector) == 16
        assert sum(v * v for v in vector) == pytest.approx(1.0)

    def test_sentence_transformer_provider_loads_lazily(self) -> None:
        sys.modules.pop("sentence_transformers", None)
        provider = SentenceTransformerProvider("some/model")
        assert provider.model_name == "some/model"
        assert "sentence_transformers" not in sys.modules

    def test_create_provider_rejects_unknown_provider(self) -> None:
        with patch.object(service.settings, "embedding_provider", "nope"):
            with pytest.raises(ValueError, match="Unknown embedding provider"):
                service.create_embedding_provider()

    def test_create_provider_uses_settings(self) -> None:
        with patch.object(service.settings, "embedding_model_name", "org/model"):
            provider = service.create_embedding_provider()
        assert isinstance(provider, SentenceTransformerProvider)
        assert provider.model_name == "org/model"

    async def test_embed_chunks_assigns_vector_and_model(self) -> None:
        provider = FakeEmbeddingProvider(dimensions=8)
        model = EmbeddingModel(id=uuid.uuid4(), name="fake-model", version="1", dimensions=8)
        chunks = [DocumentChunk(text="alpha beta"), DocumentChunk(text="gamma")]
        await service.embed_chunks(provider, model, chunks)
        for chunk in chunks:
            assert chunk.embedding_model_id == model.id
            assert len(chunk.embedding) == 8

    async def test_embed_chunks_rejects_dimension_mismatch(self) -> None:
        provider = FakeEmbeddingProvider(dimensions=8)
        model = EmbeddingModel(id=uuid.uuid4(), name="fake-model", version="1", dimensions=4)
        with pytest.raises(RuntimeError, match="expected 4"):
            await service.embed_chunks(provider, model, [DocumentChunk(text="x")])

    async def test_embed_chunks_with_no_chunks_skips_model(self) -> None:
        provider = MagicMock()
        await service.embed_chunks(provider, MagicMock(), [])
        provider.embed_documents.assert_not_called()


@pytest.mark.unit
class TestIngestionEmbeddingUnit:
    async def test_unavailable_model_stores_nothing(self, mock_db: AsyncMock) -> None:
        from app.documents.service import EmbeddingUnavailableError, ingest_document

        with (
            patch("app.documents.service._compute_hash", return_value=(b"x", "h")),
            patch("app.documents.service.get_document_by_hash", return_value=None),
            patch(
                "app.documents.service.register_model",
                new=AsyncMock(side_effect=OSError("no network")),
            ),
            patch("app.documents.service.aiofiles.open") as mock_open,
        ):
            with pytest.raises(EmbeddingUnavailableError, match="no network"):
                await ingest_document(mock_db, MagicMock(filename="a.pdf"), MagicMock())
        mock_open.assert_not_called()
        mock_db.add.assert_not_called()

    async def test_upload_returns_503_when_model_unavailable(self, client: AsyncClient) -> None:
        from app.documents.service import EmbeddingUnavailableError

        with patch(
            "app.documents.router.service.ingest_document",
            side_effect=EmbeddingUnavailableError("model missing"),
        ):
            resp = await client.post(
                "/api/v1/documents",
                files={"file": ("paper.pdf", b"%PDF", "application/pdf")},
            )
        assert resp.status_code == 503
        assert "model missing" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


async def _add_document(db: AsyncSession, texts: list[str], status: str = "completed") -> Document:
    doc = Document(
        filename="d.pdf",
        title="Doc",
        file_path="/nonexistent/d.pdf",
        file_hash=uuid.uuid4().hex,
        processing_status=status,
        page_count=1,
    )
    db.add(doc)
    await db.flush()
    for i, text in enumerate(texts):
        db.add(DocumentChunk(document_id=doc.id, chunk_index=i, text=text, page_numbers=[0]))
    await db.flush()
    return doc


@pytest.mark.integration
class TestRegistryIntegration:
    async def test_register_model_is_idempotent(self, db_session: AsyncSession) -> None:
        provider = FakeEmbeddingProvider()
        first = await service.register_model(db_session, provider)
        second = await service.register_model(db_session, provider)
        assert first.id == second.id
        assert first.dimensions == provider.dimensions
        assert first.is_active

    async def test_new_model_deactivates_previous(self, db_session: AsyncSession) -> None:
        old = await service.register_model(db_session, FakeEmbeddingProvider(version="1"))
        new = await service.register_model(db_session, FakeEmbeddingProvider(version="2"))
        await db_session.refresh(old)
        assert new.is_active and not old.is_active
        active = (
            await db_session.execute(
                select(EmbeddingModel).where(EmbeddingModel.is_active.is_(True))
            )
        ).scalars().all()
        assert [m.id for m in active] == [new.id]


@pytest.mark.integration
class TestReindexIntegration:
    async def test_reindex_embeds_unembedded_chunks(self, db_session: AsyncSession) -> None:
        doc = await _add_document(db_session, ["one", "two", "three"])
        result = await service.reindex(db_session, FakeEmbeddingProvider())
        assert (result.documents, result.chunks) == (1, 3)

        chunks = (
            await db_session.execute(
                select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
            )
        ).scalars().all()
        assert all(c.embedding_model_id == result.model.id for c in chunks)
        assert all(c.embedding is not None for c in chunks)

    async def test_reindex_is_idempotent(self, db_session: AsyncSession) -> None:
        await _add_document(db_session, ["one", "two"])
        provider = FakeEmbeddingProvider()
        await service.reindex(db_session, provider)
        again = await service.reindex(db_session, provider)
        assert again.chunks == 0

    async def test_model_change_reindexes_everything(self, db_session: AsyncSession) -> None:
        await _add_document(db_session, ["one", "two"])
        await service.reindex(db_session, FakeEmbeddingProvider(dimensions=16, version="1"))
        result = await service.reindex(
            db_session, FakeEmbeddingProvider(dimensions=32, version="2")
        )
        assert result.chunks == 2
        assert result.model.dimensions == 32

    async def test_reindex_skips_failed_documents(self, db_session: AsyncSession) -> None:
        await _add_document(db_session, ["one"], status="failed")
        result = await service.reindex(db_session, FakeEmbeddingProvider())
        assert result.chunks == 0

    async def test_status_and_reindex_endpoints(
        self, integration_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _add_document(db_session, ["one", "two"])

        status = (await integration_client.get("/api/v1/embeddings/status")).json()
        assert status["reindex_required"] is True
        assert status["total_chunks"] == 2
        assert status["indexed_chunks"] == 0

        resp = await integration_client.post("/api/v1/embeddings/reindex")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["chunks_reindexed"] == 2
        assert body["model"]["name"] == "fake-model"

        status = (await integration_client.get("/api/v1/embeddings/status")).json()
        assert status["indexed_chunks"] == 2
        assert status["active_model"]["name"] == "fake-model"


# ---------------------------------------------------------------------------
# Real model (opt-in: pytest -m model)
# ---------------------------------------------------------------------------


@pytest.mark.model
def test_real_model_ranks_relevant_passage_first() -> None:
    provider = service.create_embedding_provider()
    passages = [
        "Linear discriminant analysis projects data onto axes that maximise class separation.",
        "The museum opens at nine and closes at five on weekdays.",
    ]
    vectors = provider.embed_documents(passages)
    query = provider.embed_query("How does LDA separate classes?")
    scores = [sum(q * v for q, v in zip(query, vec, strict=True)) for vec in vectors]
    assert len(query) == provider.dimensions
    assert scores[0] > scores[1]
    assert len(provider.model_version) >= 7

