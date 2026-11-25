"""Run a complete experiment and record everything needed to reproduce it."""

import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.prompts import PROMPT_VERSION
from app.core.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.evaluation import report
from app.evaluation.config import ExperimentConfig, settings_snapshot
from app.evaluation.dataset import LoadedDataset
from app.evaluation.judge import JUDGE_PROMPT_VERSION
from app.evaluation.runner import (
    Progress,
    _noop,
    check_dataset,
    evaluate_generation,
    evaluate_retrieval,
    prepare_corpus,
)
from app.llm.registry import LLMRegistry

REPO_ROOT = Path(__file__).resolve().parents[3]


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def environment() -> dict[str, Any]:
    def version(package: str) -> str | None:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "packages": {p: version(p) for p in ("torch", "sentence-transformers", "pymupdf")},
    }


async def run_experiment(
    db: AsyncSession,
    loaded: LoadedDataset,
    config: ExperimentConfig,
    embedder: EmbeddingProvider,
    registry: LLMRegistry,
    settings: Settings,
    results_root: Path,
    progress: Progress = _noop,
) -> Path:
    """Run the experiment and write its results directory; returns its path.

    `config.apply(settings)` must already have been called and `embedder` built
    from the resulting settings.
    """
    started = datetime.now(UTC)
    t0 = time.perf_counter()
    out = results_root / f"{started.strftime('%Y%m%d-%H%M%S')}_{config.name}"
    out.mkdir(parents=True, exist_ok=False)

    dataset = loaded.dataset
    run_config: dict[str, Any] = {
        "experiment": config.model_dump(),
        "dataset": {
            "name": dataset.name,
            "kind": dataset.kind,
            "path": str(loaded.directory),
            "fingerprint": loaded.fingerprint,
            "documents": len(dataset.documents),
            "questions": len(dataset.questions),
            "verified_questions": sum(1 for q in dataset.questions if q.verified),
        },
        "settings": settings_snapshot(settings),
        "prompt_version": PROMPT_VERSION,
        "git": git_state(),
        "environment": environment(),
        "started_at": started.isoformat(timespec="seconds"),
    }

    corpus = await prepare_corpus(db, loaded, embedder, settings, progress)
    run_config["embedding_model"] = {
        "name": corpus.embedding_model.name,
        "version": corpus.embedding_model.version,
        "dimensions": corpus.embedding_model.dimensions,
    }
    run_config["reprocessed_documents"] = corpus.reprocessed

    threshold = config.retrieval.passage_match_threshold
    issues = await check_dataset(db, loaded, corpus, threshold)
    for issue in issues:
        progress(f"Dataset issue: {issue['question']}: {issue['problem']}")

    retrieval_records = await evaluate_retrieval(
        db, embedder, dataset.questions, corpus, config, progress
    )
    report.write_jsonl(out / "retrieval.jsonl", retrieval_records)
    summary: dict[str, Any] = {
        "dataset_issues": issues,
        "retrieval": report.summarize_retrieval(
            retrieval_records,
            config.retrieval.k_values,
            settings.chat_min_relevance,
            config.retrieval.gate_min_pass_rate,
        ),
    }

    if config.generation.enabled:
        provider = registry.get(config.generation.provider)
        run_config["llm"] = {
            "provider": provider.name,
            "model": await provider.resolve_model(config.generation.model),
            "is_local": provider.is_local,
        }
        if config.judge.enabled:
            judge = registry.get(config.judge.provider)
            run_config["judge"] = {
                "provider": judge.name,
                "model": await judge.resolve_model(config.judge.model),
                "prompt_version": JUDGE_PROMPT_VERSION,
            }
        generation_records = await evaluate_generation(
            db, embedder, registry, dataset.questions, corpus, config, progress
        )
        report.write_jsonl(out / "generation.jsonl", generation_records)
        report.write_manual_review(out / "manual_review.csv", generation_records)
        summary["generation"] = report.summarize_generation(generation_records)

    run_config["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    run_config["duration_s"] = round(time.perf_counter() - t0, 1)
    summary = {"config": run_config, **summary}

    (out / "config.json").write_text(
        json.dumps(run_config, indent=2, default=str), encoding="utf-8"
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out / "report.md").write_text(report.render_report(summary), encoding="utf-8")
    return out
