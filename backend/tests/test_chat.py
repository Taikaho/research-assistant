"""RAG question answering: prompts, citation parsing, API and end-to-end pipeline."""

import uuid
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pymupdf
import pytest
from httpx import AsyncClient
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat import service
from app.chat.citations import parse_answer, strip_citations
from app.chat.prompts import INSUFFICIENT_EVIDENCE_MARKER, PromptPassage, build_messages
from app.core.config import settings
from app.database.models import Conversation, DocumentPage, Message, MessageSource
from app.llm.base import (
    ChatMessage,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from tests.fakes import FakeLLMProvider

# ---------------------------------------------------------------------------
# Unit: citation parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseAnswer:
    def test_collects_valid_citations_in_order(self) -> None:
        parsed = parse_answer("LDA projects data [2]. KNN votes [1][2].", passage_count=3)
        assert parsed.cited_numbers == [2, 1]
        assert parsed.invalid_numbers == []
        assert parsed.text == "LDA projects data [2]. KNN votes [1][2]."
        assert not parsed.insufficient_evidence

    def test_comma_separated_markers_are_normalised(self) -> None:
        parsed = parse_answer("Both methods classify [1, 3].", passage_count=3)
        assert parsed.text == "Both methods classify [1][3]."
        assert parsed.cited_numbers == [1, 3]

    def test_out_of_range_citations_are_removed(self) -> None:
        parsed = parse_answer("Claim [7]. Other claim [1, 9].", passage_count=2)
        assert parsed.text == "Claim. Other claim [1]."
        assert parsed.cited_numbers == [1]
        assert parsed.invalid_numbers == [7, 9]

    def test_zero_is_not_a_valid_passage(self) -> None:
        assert parse_answer("Claim [0].", passage_count=2).invalid_numbers == [0]

    @pytest.mark.parametrize(
        "raw",
        [
            f"{INSUFFICIENT_EVIDENCE_MARKER}\nThe passages do not discuss soil.",
            f"**{INSUFFICIENT_EVIDENCE_MARKER}**: The passages do not discuss soil.",
            f"`{INSUFFICIENT_EVIDENCE_MARKER}` The passages do not discuss soil.",
            "insufficient_evidence - The passages do not discuss soil.",
        ],
    )
    def test_insufficient_evidence_marker_variants(self, raw: str) -> None:
        parsed = parse_answer(raw, passage_count=2)
        assert parsed.insufficient_evidence
        assert parsed.text == "The passages do not discuss soil."
        assert parsed.cited_numbers == []

    def test_marker_later_in_text_is_not_a_refusal(self) -> None:
        parsed = parse_answer(f"Answer [1]. Not {INSUFFICIENT_EVIDENCE_MARKER}.", 1)
        assert not parsed.insufficient_evidence

    def test_strip_citations_tidies_whitespace(self) -> None:
        assert strip_citations("LDA projects spectra [1]. KNN votes [2, 3] .") == (
            "LDA projects spectra. KNN votes."
        )

    def test_bracketed_years_are_not_citations(self) -> None:
        parsed = parse_answer("Published in [2019] by the authors [1].", passage_count=2)
        assert parsed.text == "Published in [2019] by the authors [1]."
        assert parsed.cited_numbers == [1]
        assert parsed.invalid_numbers == []


# ---------------------------------------------------------------------------
# Unit: prompts
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPrompts:
    def test_structure_and_grounding_rules(self) -> None:
        history = [ChatMessage("user", "earlier q"), ChatMessage("assistant", "earlier a")]
        messages = build_messages(
            "What is LDA?",
            [PromptPassage(1, "Paper A", "LDA text"), PromptPassage(2, "Paper B", "KNN text")],
            history,
        )
        assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
        system = messages[0].content
        assert "ONLY" in system and INSUFFICIENT_EVIDENCE_MARKER in system
        assert "Never invent" in system
        final = messages[-1].content
        assert "[1] Document: Paper A" in final and "[2] Document: Paper B" in final
        assert final.rstrip().endswith("Question: What is LDA?")

    def test_prompt_contains_no_page_numbers_or_ids(self) -> None:
        # The model gets nothing it could copy into an invented reference.
        content = build_messages("q", [PromptPassage(1, "T", "body")])[-1].content
        assert "page" not in content.lower()


# ---------------------------------------------------------------------------
# Unit: API validation and error mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChatRouterUnit:
    async def test_blank_question_rejected(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/chat", json={"question": "  ", "collection_ids": [str(uuid.uuid4())]}
        )
        assert resp.status_code == 422

    async def test_empty_collection_list_rejected(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/chat", json={"question": "q", "collection_ids": []})
        assert resp.status_code == 422

    @pytest.mark.parametrize(
        ("error", "status_code"),
        [
            (LLMConnectionError("down"), 503),
            (LLMTimeoutError("slow"), 504),
            (LLMModelNotFoundError("no model"), 400),
            (LLMRateLimitError("limit", retry_after=12), 429),
            (service.MissingScopeError("none"), 422),
            (service.ConversationNotFoundError("gone"), 404),
            (service.RetrievalFailedError("embedder"), 503),
        ],
    )
    async def test_errors_map_to_http_status(
        self, client: AsyncClient, error: Exception, status_code: int
    ) -> None:
        with patch("app.chat.router.service.answer_question", side_effect=error):
            resp = await client.post(
                "/api/v1/chat", json={"question": "q", "collection_ids": [str(uuid.uuid4())]}
            )
        assert resp.status_code == status_code
        if isinstance(error, LLMRateLimitError):
            assert resp.headers["retry-after"] == "12"


@pytest.mark.unit
class TestModelsRouterUnit:
    async def test_providers_lists_configuration_without_secrets(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/models/providers")).json()
        assert body["default_provider"] == "fake-llm"
        assert [(p["name"], p["is_local"], p["is_default"]) for p in body["providers"]] == [
            ("fake-llm", True, True),
            ("fake-cloud", False, False),
        ]
        assert "key" not in str(body).lower()

    async def test_models_reports_unavailable_provider(
        self, client: AsyncClient, fake_cloud_llm: FakeLLMProvider
    ) -> None:
        fake_cloud_llm.list_error = LLMConnectionError("cloud unreachable")
        body = (await client.get("/api/v1/models")).json()
        by_name = {p["name"]: p for p in body["providers"]}
        assert by_name["fake-llm"]["available"] is True
        assert by_name["fake-llm"]["models"] == ["fake-model-1", "fake-model-2"]
        assert by_name["fake-cloud"]["available"] is False
        assert by_name["fake-cloud"]["error"] == "cloud unreachable"

    async def test_health(self, client: AsyncClient, fake_llm: FakeLLMProvider) -> None:
        fake_llm.list_error = LLMConnectionError("ollama down")
        body = (await client.get("/api/v1/models/health")).json()
        by_name = {p["name"]: p for p in body["providers"]}
        assert by_name["fake-llm"] == {
            "name": "fake-llm",
            "is_local": True,
            "available": False,
            "detail": "ollama down",
        }
        assert by_name["fake-cloud"]["available"] is True


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

LDA_TEXT = (
    "Linear discriminant analysis is a supervised method that projects spectra onto "
    "axes maximising the separation between classes."
)
KNN_TEXT = (
    "The k nearest neighbours classifier assigns a label by majority vote among the "
    "closest training samples in feature space."
)
SOIL_TEXT = (
    "Soil moisture sensors were deployed across the field and sampled every hour "
    "using low power wireless nodes."
)
LDA_QUESTION = "What does linear discriminant analysis do with spectra and classes?"


def _pdf(*pages: str) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text)
    data: bytes = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[Path]:
    with patch("app.documents.service.settings") as mock_settings:
        mock_settings.storage_path = str(tmp_path)
        mock_settings.max_upload_size_mb = 50
        mock_settings.chunk_size = 2000
        mock_settings.chunk_overlap = 400
        yield tmp_path


@pytest.fixture
def relevance_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    # Bag-of-words fake embeddings score lower than real models; use a gate that
    # separates the on-topic test passages from the off-topic ones.
    monkeypatch.setattr(settings, "chat_min_relevance", 0.3)


async def _collection(client: AsyncClient, name: str) -> str:
    resp = await client.post("/api/v1/collections", json={"name": name})
    assert resp.status_code == 201
    return str(resp.json()["id"])


async def _upload(client: AsyncClient, pdf: bytes, collection_ids: list[str]) -> str:
    resp = await client.post(
        "/api/v1/documents",
        files={"file": ("paper.pdf", pdf, "application/pdf")},
        data={"collection_ids": collection_ids},
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


async def _chat(client: AsyncClient, expected: int = 200, **payload: object) -> dict:  # type: ignore[type-arg]
    resp = await client.post("/api/v1/chat", json=payload)
    assert resp.status_code == expected, resp.text
    body: dict = resp.json()  # type: ignore[type-arg]
    return body


@pytest.mark.integration
@pytest.mark.usefixtures("storage", "relevance_gate")
class TestChatIntegration:
    async def test_answer_with_backend_assembled_sources(
        self, integration_client: AsyncClient, db_session: AsyncSession, fake_llm: FakeLLMProvider
    ) -> None:
        col = await _collection(integration_client, "ML")
        doc = await _upload(integration_client, _pdf(SOIL_TEXT, LDA_TEXT), [col])
        await db_session.execute(
            update(DocumentPage)
            .where(DocumentPage.document_id == uuid.UUID(doc), DocumentPage.pdf_page_number == 1)
            .values(printed_page_number="124")
        )
        fake_llm.replies = ["LDA maximises the separation between classes [1]."]

        body = await _chat(integration_client, question=LDA_QUESTION, collection_ids=[col])

        assert body["answer"] == "LDA maximises the separation between classes [1]."
        assert body["insufficient_evidence"] is False
        assert body["model"] == {"provider": "fake-llm", "name": "fake-model-1", "is_local": True}
        assert body["cloud_processing"] is False
        assert body["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}
        source = body["sources"][0]
        assert source["citation_number"] == 1 and source["cited"] is True
        assert source["document_id"] == doc
        assert source["filename"] == "paper.pdf"
        assert 1 in source["pdf_page_numbers"]
        assert "124" in source["printed_page_numbers"]
        assert "discriminant" in source["excerpt"]

        # The model saw the passage text but no page numbers or chunk ids.
        _, messages, params = fake_llm.calls[0]
        prompt = messages[-1].content
        assert "discriminant" in prompt and source["chunk_id"] not in prompt
        assert params.temperature == settings.llm_temperature

    async def test_answer_is_persisted_with_model_and_configuration(
        self, integration_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        col = await _collection(integration_client, "Persist")
        await _upload(integration_client, _pdf(LDA_TEXT), [col])
        body = await _chat(
            integration_client,
            question=LDA_QUESTION,
            collection_ids=[col],
            model_name="fake-model-2",
            temperature=0.7,
            max_tokens=200,
        )

        assistant = await db_session.get(Message, uuid.UUID(body["message_id"]))
        assert assistant is not None
        assert (assistant.model_provider, assistant.model_name) == ("fake-llm", "fake-model-2")
        meta = assistant.generation_metadata
        assert meta["temperature"] == 0.7 and meta["max_tokens"] == 200
        assert meta["prompt_version"] and meta["embedding_model"]["name"] == "fake-model"
        assert meta["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}
        stored = (
            await db_session.execute(
                select(MessageSource).where(MessageSource.message_id == assistant.id)
            )
        ).scalars().all()
        assert [(s.citation_number, s.cited) for s in stored] == [(1, True)]

        detail = (
            await integration_client.get(f"/api/v1/conversations/{body['conversation_id']}")
        ).json()
        assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
        assert detail["messages"][1]["sources"][0]["chunk_id"] == body["sources"][0]["chunk_id"]
        assert detail["collection_ids"] == [col]

    async def test_no_relevant_passages_skips_the_llm(
        self, integration_client: AsyncClient, fake_llm: FakeLLMProvider
    ) -> None:
        col = await _collection(integration_client, "Soil")
        await _upload(integration_client, _pdf(SOIL_TEXT), [col])
        body = await _chat(integration_client, question=LDA_QUESTION, collection_ids=[col])
        assert body["insufficient_evidence"] is True
        assert body["insufficient_evidence_reason"] == "no_relevant_passages"
        assert body["sources"] == [] and body["model"] is None
        assert fake_llm.calls == []

    async def test_model_reported_insufficient_evidence(
        self, integration_client: AsyncClient, fake_llm: FakeLLMProvider
    ) -> None:
        col = await _collection(integration_client, "Partial")
        await _upload(integration_client, _pdf(LDA_TEXT), [col])
        fake_llm.replies = [f"{INSUFFICIENT_EVIDENCE_MARKER}\nNo accuracy figures are given."]
        body = await _chat(integration_client, question=LDA_QUESTION, collection_ids=[col])
        assert body["insufficient_evidence"] is True
        assert body["insufficient_evidence_reason"] == "model_reported"
        assert body["answer"] == "No accuracy figures are given."
        assert all(not s["cited"] for s in body["sources"])

    async def test_invented_citation_numbers_are_removed(
        self, integration_client: AsyncClient, fake_llm: FakeLLMProvider
    ) -> None:
        col = await _collection(integration_client, "Invented")
        await _upload(integration_client, _pdf(LDA_TEXT), [col])
        fake_llm.replies = ["LDA separates classes [1] as reported elsewhere [5]."]
        body = await _chat(integration_client, question=LDA_QUESTION, collection_ids=[col])
        assert body["answer"] == "LDA separates classes [1] as reported elsewhere."
        assert [s["citation_number"] for s in body["sources"]] == [1]

    async def test_sources_stay_within_selected_collections(
        self, integration_client: AsyncClient
    ) -> None:
        a = await _collection(integration_client, "A")
        b = await _collection(integration_client, "B")
        in_scope = await _upload(integration_client, _pdf(KNN_TEXT + " " + LDA_TEXT), [a])
        await _upload(integration_client, _pdf(LDA_TEXT), [b])
        body = await _chat(integration_client, question=LDA_QUESTION, collection_ids=[a])
        assert {s["document_id"] for s in body["sources"]} == {in_scope}

    async def test_follow_up_uses_history_and_stored_scope(
        self, integration_client: AsyncClient, fake_llm: FakeLLMProvider
    ) -> None:
        col = await _collection(integration_client, "Follow")
        await _upload(integration_client, _pdf(LDA_TEXT), [col])
        fake_llm.replies = ["LDA projects spectra [1].", "It maximises class separation [1]."]
        first = await _chat(integration_client, question=LDA_QUESTION, collection_ids=[col])

        second = await _chat(
            integration_client,
            question="How does linear discriminant analysis separate classes?",
            conversation_id=first["conversation_id"],
        )
        assert second["conversation_id"] == first["conversation_id"]
        assert second["collection_ids"] == [col]
        _, messages, _ = fake_llm.calls[1]
        assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
        assert messages[1].content == LDA_QUESTION
        # Old citation markers are stripped from history.
        assert messages[2].content == "LDA projects spectra."

        conversations = (await integration_client.get("/api/v1/conversations")).json()
        assert conversations[0]["message_count"] == 4

    async def test_follow_up_without_topic_words_is_answered(
        self,
        integration_client: AsyncClient,
        db_session: AsyncSession,
        fake_llm: FakeLLMProvider,
    ) -> None:
        col = await _collection(integration_client, "Elliptic")
        await _upload(integration_client, _pdf(LDA_TEXT), [col])
        first = await _chat(integration_client, question=LDA_QUESTION, collection_ids=[col])
        first_meta = (await db_session.get(Message, uuid.UUID(first["message_id"])))
        assert first_meta is not None
        assert first_meta.generation_metadata["retrieval_queries"] == [LDA_QUESTION]

        follow_up = "Who invented it?"
        # On its own the follow-up scores below the gate (fake embedder: 0.13 vs 0.59).
        alone = await integration_client.post(
            "/api/v1/search", json={"query": follow_up, "collection_ids": [col]}
        )
        assert all(r["score"] < settings.chat_min_relevance for r in alone.json()["results"])

        fake_llm.replies = ["It maximises class separation [1]."]
        body = await _chat(
            integration_client, question=follow_up, conversation_id=first["conversation_id"]
        )
        assert body["insufficient_evidence"] is False
        assert body["sources"] and body["sources"][0]["cited"] is True
        meta = (await db_session.get(Message, uuid.UUID(body["message_id"])))
        assert meta is not None
        assert meta.generation_metadata["retrieval_queries"] == [
            follow_up,
            f"{LDA_QUESTION}\n{follow_up}",
        ]

    async def test_switching_provider_keeps_pipeline_working(
        self, integration_client: AsyncClient, fake_cloud_llm: FakeLLMProvider
    ) -> None:
        col = await _collection(integration_client, "Switch")
        await _upload(integration_client, _pdf(LDA_TEXT), [col])
        fake_cloud_llm.replies = ["Cloud says LDA separates classes [1]."]
        body = await _chat(
            integration_client,
            question=LDA_QUESTION,
            collection_ids=[col],
            model_provider="fake-cloud",
        )
        assert body["model"] == {"provider": "fake-cloud", "name": "cloud-model", "is_local": False}
        assert body["cloud_processing"] is True
        assert body["sources"][0]["cited"] is True

    async def test_unknown_provider_is_rejected(self, integration_client: AsyncClient) -> None:
        col = await _collection(integration_client, "Unknown")
        resp = await integration_client.post(
            "/api/v1/chat",
            json={"question": "q", "collection_ids": [col], "model_provider": "openai"},
        )
        assert resp.status_code == 400
        assert "not configured" in resp.json()["detail"]

    async def test_provider_failure_persists_nothing(
        self,
        integration_client: AsyncClient,
        db_session: AsyncSession,
        fake_llm: FakeLLMProvider,
    ) -> None:
        col = await _collection(integration_client, "Fail")
        await _upload(integration_client, _pdf(LDA_TEXT), [col])
        fake_llm.replies = [LLMConnectionError("Cannot reach Ollama")]
        resp = await integration_client.post(
            "/api/v1/chat", json={"question": LDA_QUESTION, "collection_ids": [col]}
        )
        assert resp.status_code == 503
        assert "Cannot reach Ollama" in resp.json()["detail"]
        count = (await db_session.execute(select(func.count()).select_from(Conversation))).scalar()
        assert count == 0

    async def test_unknown_collection_and_conversation(
        self, integration_client: AsyncClient
    ) -> None:
        await _chat(
            integration_client, 404, question="q", collection_ids=[str(uuid.uuid4())]
        )
        await _chat(integration_client, 404, question="q", conversation_id=str(uuid.uuid4()))
        await _chat(integration_client, 422, question="q")

    async def test_delete_conversation_keeps_documents(
        self, integration_client: AsyncClient
    ) -> None:
        col = await _collection(integration_client, "Delete")
        doc = await _upload(integration_client, _pdf(LDA_TEXT), [col])
        body = await _chat(integration_client, question=LDA_QUESTION, collection_ids=[col])
        conv = body["conversation_id"]
        assert (await integration_client.delete(f"/api/v1/conversations/{conv}")).status_code == 204
        assert (await integration_client.get(f"/api/v1/conversations/{conv}")).status_code == 404
        assert (await integration_client.get(f"/api/v1/documents/{doc}")).status_code == 200
