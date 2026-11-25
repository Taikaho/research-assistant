"""Tests for the RAG evaluation framework (app.evaluation)."""

import csv
import json
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, settings
from app.evaluation.config import ExperimentConfig
from app.evaluation.dataset import Dataset, load_dataset
from app.evaluation.experiment import run_experiment
from app.evaluation.judge import JudgeParseError, parse_judge_reply
from app.evaluation.matching import ChunkRef, matched_items, normalize, passage_coverage
from app.evaluation.metrics import (
    best_balanced_threshold,
    gate_curve,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    recall_preserving_threshold,
    reciprocal_rank,
)
from app.evaluation.report import (
    render_comparison,
    render_report,
    summarize_generation,
    summarize_manual_review,
    summarize_retrieval,
    write_manual_review,
)
from app.llm.registry import LLMRegistry
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _question(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "q1",
        "question": "What?",
        "expected_answer": "That.",
        "relevant": [{"document": "d1", "pdf_page": 1, "passage": "text"}],
    }
    return {**base, **overrides}


def _dataset(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "t",
        "documents": [{"id": "d1", "file": "d1.pdf"}],
        "questions": [_question()],
    }
    return {**base, **overrides}


def _write_pdf(path: Path, pages: list[str]) -> None:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
    doc.save(path)
    doc.close()


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMatching:
    def test_normalize_joins_hyphenated_line_breaks_and_whitespace(self) -> None:
        assert normalize("Classi-\n  fication  of\nMinerals") == "classification of minerals"

    def test_full_and_partial_coverage(self) -> None:
        chunk = "Intro. LDA reached 91.3 percent accuracy. End of chunk"
        assert passage_coverage(chunk, "LDA reached 91.3\npercent accuracy.") == 1.0
        # Passage continues beyond the chunk: only its first part is present.
        partial = passage_coverage(chunk, "End of chunk and more text on the next")
        assert 0.25 < partial < 0.5

    def test_chunk_must_be_from_the_document_and_page(self) -> None:
        relevant = Dataset.model_validate(_dataset()).questions[0].relevant
        text = "some text here"
        assert matched_items(ChunkRef("d1", [0], text), relevant, 0.5) == {0}
        assert matched_items(ChunkRef("d1", [1], text), relevant, 0.5) == set()  # wrong page
        assert matched_items(ChunkRef("d2", [0], text), relevant, 0.5) == set()  # wrong doc
        assert matched_items(ChunkRef(None, [0], text), relevant, 0.5) == set()

    def test_passage_free_relevance_is_page_level(self) -> None:
        relevant = (
            Dataset.model_validate(
                _dataset(questions=[_question(relevant=[{"document": "d1", "pdf_page": 2}])])
            )
            .questions[0]
            .relevant
        )
        assert matched_items(ChunkRef("d1", [0, 1], "anything"), relevant, 0.5) == {0}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetrics:
    ranked = [set(), {0}, set(), {1}, {0}]

    def test_recall_precision_hit(self) -> None:
        assert recall_at_k(self.ranked, 2, 1) == 0.0
        assert recall_at_k(self.ranked, 2, 2) == 0.5
        assert recall_at_k(self.ranked, 2, 4) == 1.0
        assert precision_at_k(self.ranked, 5) == 3 / 5
        # Divides by k even when fewer results were returned.
        assert precision_at_k([{0}], 5) == 1 / 5
        assert hit_at_k(self.ranked, 1) == 0.0
        assert hit_at_k(self.ranked, 2) == 1.0

    def test_reciprocal_rank(self) -> None:
        assert reciprocal_rank(self.ranked) == 0.5
        assert reciprocal_rank([set(), set()]) == 0.0

    def test_gate_recommendations(self) -> None:
        answerable = [0.45, 0.55, 0.7]
        unanswerable = [0.3, 0.5]
        curve = gate_curve(answerable, unanswerable, 0.2, 0.8, 0.05)
        preserving = recall_preserving_threshold(curve, 1.0)
        assert preserving is not None
        assert preserving.threshold == 0.45
        assert preserving.answerable_pass_rate == 1.0
        assert preserving.unanswerable_reject_rate == 0.5
        balanced = best_balanced_threshold(curve)
        assert balanced is not None
        assert balanced.balanced_accuracy == pytest.approx(5 / 6)

    def test_no_threshold_meets_impossible_target(self) -> None:
        assert recall_preserving_threshold([], 0.9) is None


# ---------------------------------------------------------------------------
# Dataset, config, judge
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDatasetAndConfig:
    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"questions": [_question(relevant=[])]}, "need 'relevant'"),
            ({"questions": [_question(expected_answer=None)]}, "need 'expected_answer'"),
            ({"questions": [_question(answerable=False)]}, "must not list passages"),
            (
                {"questions": [_question(relevant=[{"document": "x", "pdf_page": 1}])]},
                "unknown document",
            ),
            ({"questions": [_question(), _question()]}, "question ids must be unique"),
        ],
    )
    def test_invalid_datasets(self, overrides: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            Dataset.model_validate(_dataset(**overrides))

    def test_load_dataset_requires_pdfs_and_fingerprints_them(self, tmp_path: Path) -> None:
        (tmp_path / "dataset.json").write_text(json.dumps(_dataset()), encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path)
        (tmp_path / "d1.pdf").write_bytes(b"one")
        first = load_dataset(tmp_path).fingerprint
        (tmp_path / "d1.pdf").write_bytes(b"two")
        assert load_dataset(tmp_path / "dataset.json").fingerprint != first

    def test_config_applies_overrides_only(self) -> None:
        target = Settings()
        original_prefix = target.embedding_query_prefix
        config = ExperimentConfig.model_validate(
            {
                "name": "x",
                "embedding": {"model_name": "BAAI/bge-m3"},
                "chunking": {"chunk_size": 800},
                "generation": {"min_relevance": 0.45},
            }
        )
        config.apply(target)
        assert target.embedding_model_name == "BAAI/bge-m3"
        assert target.embedding_query_prefix == original_prefix  # not given: unchanged
        assert target.chunk_size == 800
        assert target.chat_min_relevance == 0.45
        assert config.search_depth == 10

    def test_judge_reply_parsing(self) -> None:
        scores = parse_judge_reply(
            'Sure: {"correctness": 4, "faithfulness": "5", "relevance": 4.0, "explanation": "ok"}'
        )
        assert (scores.correctness, scores.faithfulness, scores.relevance) == (4, 5, 4)
        with pytest.raises(JudgeParseError):
            parse_judge_reply("no json here")
        with pytest.raises(JudgeParseError):
            parse_judge_reply('{"correctness": 9, "faithfulness": 1, "relevance": 1}')
        with pytest.raises(JudgeParseError):
            parse_judge_reply('{"correctness": 3}')


# ---------------------------------------------------------------------------
# Summaries and reports
# ---------------------------------------------------------------------------


def _retrieval_record(qid: str, answerable: bool, top: float, rr: float, lang: str = "en") -> Any:
    record: dict[str, Any] = {
        "id": qid,
        "language": lang,
        "answerable": answerable,
        "top_score": top,
        "latency_ms": 10.0,
        "results": [{"score": top, "matched": [0] if rr == 1.0 else []}],
    }
    if answerable:
        record["metrics"] = {"recall@1": rr, "precision@1": rr, "hit@1": rr, "rr": rr}
    return record


def _generation_record(qid: str, answerable: bool, abstained: bool, cited_ok: bool) -> Any:
    return {
        "id": qid,
        "language": "en",
        "answerable": answerable,
        "question": "Q?",
        "expected_answer": "A." if answerable else None,
        "answer": "Answer [1].",
        "abstained": abstained,
        "sources": [
            {
                "number": 1,
                "cited": not abstained,
                "document": "d1",
                "pdf_pages": [2],
                "relevant": cited_ok,
            }
        ],
        "cited_count": 0 if abstained else 1,
        "cited_relevant_count": 1 if (cited_ok and not abstained) else 0,
        "citation_hit": (cited_ok if answerable and not abstained else None),
        "uncited_answer": False,
        "context_hit": cited_ok if answerable else None,
        "invalid_citation_numbers": [],
        "latency_ms": {"retrieval": 5, "generation": 100, "total": 110},
        "usage": {"prompt_tokens": 500, "completion_tokens": 40},
        "judge": {"correctness": 4, "faithfulness": 5, "relevance": 5, "explanation": ""},
    }


@pytest.mark.unit
class TestReports:
    def test_summaries_and_report_rendering(self, tmp_path: Path) -> None:
        retrieval = summarize_retrieval(
            [
                _retrieval_record("a", True, 0.7, 1.0),
                _retrieval_record("b", True, 0.5, 0.0, lang="fi"),
                _retrieval_record("c", False, 0.3, 0.0),
            ],
            [1],
            current_gate=0.6,
        )
        assert retrieval["metrics"]["mrr"] == 0.5
        assert retrieval["by_language"]["fi"]["hit@1"] == 0.0
        assert retrieval["gate"]["recommended"]["threshold"] == 0.5

        records = [
            _generation_record("a", True, False, True),
            _generation_record("b", True, True, False),
            _generation_record("c", False, True, False),
            {
                "id": "d",
                "language": "en",
                "answerable": True,
                "question": "Q",
                "error": "LLMTimeoutError: slow",
            },
        ]
        generation = summarize_generation(records)
        assert generation["errors"] == 1
        assert generation["answer_rate"] == 0.5
        assert generation["correct_abstention_rate"] == 1.0
        assert generation["citation_precision"] == 1.0
        assert generation["judge"]["correctness"] == 4

        config = {
            "experiment": {"name": "run1", "retrieval": {"k_values": [1]}},
            "dataset": {
                "name": "t",
                "kind": "manual",
                "questions": 4,
                "verified_questions": 4,
                "fingerprint": "abc",
            },
            "embedding_model": {"name": "m", "version": "v", "dimensions": 3},
            "settings": {"chunk_size": 2000, "chunk_overlap": 400, "chat_min_relevance": 0.6},
            "git": {"commit": "123", "dirty": False},
            "started_at": "now",
            "prompt_version": "p",
            "llm": {"provider": "ollama", "model": "x", "is_local": True},
        }
        run = {
            "config": config,
            "retrieval": retrieval,
            "generation": generation,
            "dataset_issues": [{"question": "a", "problem": "passage not found"}],
        }
        text = render_report(run)
        assert "MRR: **0.500**" in text
        assert "passage not found" in text
        assert "0.50 (recommended)" in text
        assert "| run1 |" in render_comparison([run])

        sheet = tmp_path / "review.csv"
        write_manual_review(sheet, records)
        with sheet.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        rows[0]["manual_correct"] = "1"
        rows[1]["manual_correct"] = "0"
        with sheet.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manual = summarize_manual_review(sheet)
        assert manual["correct"] == {"graded": 2, "rate": 0.5}


# ---------------------------------------------------------------------------
# Integration: a full run against the test database with fake models
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_run_experiment_end_to_end(
    db_session: AsyncSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    _write_pdf(
        data_dir / "lda.pdf",
        [
            "Linear discriminant analysis reached 91 percent accuracy on mineral spectra.",
            "Soil moisture was measured with probes at a depth of five centimetres.",
        ],
    )
    dataset = _dataset(
        name="it",
        documents=[{"id": "lda", "file": "lda.pdf"}],
        questions=[
            _question(
                id="ok",
                question="What accuracy did linear discriminant analysis reach on spectra?",
                expected_answer="91 percent",
                relevant=[
                    {"document": "lda", "pdf_page": 1, "passage": "reached 91 percent accuracy"}
                ],
            ),
            _question(
                id="wrong-page",
                question="At what depth was soil moisture measured?",
                expected_answer="Five centimetres",
                relevant=[
                    {"document": "lda", "pdf_page": 1, "passage": "at a depth of five centimetres"}
                ],
            ),
            {
                "id": "none",
                "question": "Who won the 2019 hockey world championship?",
                "answerable": False,
            },
        ],
    )
    (data_dir / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")

    monkeypatch.setattr(settings, "storage_path", str(tmp_path / "storage"))
    monkeypatch.setattr(settings, "chat_min_relevance", 0.0)
    answerer = FakeLLMProvider()
    answerer.default_reply = "It reached 91 percent [1]."
    judge = FakeLLMProvider(name="judge")
    judge.default_reply = (
        '{"correctness": 5, "faithfulness": 4, "relevance": 5, "explanation": "x"}'
    )
    registry = LLMRegistry([answerer, judge], "fake-llm")
    config = ExperimentConfig.model_validate(
        {
            "name": "it-run",
            "retrieval": {"k_values": [1, 3]},
            "generation": {"provider": "fake-llm"},
            "judge": {"enabled": True, "provider": "judge"},
        }
    )

    out = await run_experiment(
        db_session,
        load_dataset(data_dir),
        config,
        FakeEmbeddingProvider(),
        registry,
        settings,
        tmp_path / "results",
    )

    for name in (
        "config.json",
        "summary.json",
        "report.md",
        "retrieval.jsonl",
        "generation.jsonl",
        "manual_review.csv",
    ):
        assert (out / name).is_file(), name
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))

    # The wrong-page passage is reported (it is on PDF page 2, not 1).
    issues = summary["dataset_issues"]
    assert [i["question"] for i in issues] == ["wrong-page"]
    assert "found on page(s) [2]" in issues[0]["problem"]

    assert summary["config"]["embedding_model"]["name"] == "fake-model"
    assert summary["config"]["llm"] == {
        "provider": "fake-llm",
        "model": "fake-model-1",
        "is_local": True,
    }
    assert summary["retrieval"]["answerable"] == 2
    generation = summary["generation"]
    assert generation["errors"] == 0
    assert generation["judge"]["judged"] >= 1
    assert generation["judge"]["correctness"] == 5

    records = {
        json.loads(line)["id"]: json.loads(line)
        for line in (out / "generation.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert records["ok"]["citation_hit"] is True
    assert records["ok"]["cited_relevant_count"] == 1

    # A second run reuses the ingested document instead of uploading it again.
    second = await run_experiment(
        db_session,
        load_dataset(data_dir),
        config,
        FakeEmbeddingProvider(),
        registry,
        settings,
        tmp_path / "results2",
    )
    assert json.loads((second / "config.json").read_text())["reprocessed_documents"] == []
