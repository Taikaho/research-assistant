"""Evaluation command line.

    python -m app.evaluation setup-db
    python -m app.evaluation validate ../evaluation/datasets/smoke
    python -m app.evaluation run --dataset ../evaluation/datasets/smoke \\
        --config ../evaluation/configs/bge-m3_llama3.2-3b.json
    python -m app.evaluation compare ../evaluation/results/<run1> ../evaluation/results/<run2>
    python -m app.evaluation review ../evaluation/results/<run>

DATABASE_URL must point to a dedicated evaluation database whose name contains
"eval" (e.g. research_assistant_eval): runs switch the active embedding model
and re-embed every chunk in that database.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from app.core.config import settings

DEFAULT_RESULTS = Path(__file__).resolve().parents[3] / "evaluation" / "results"


def _require_eval_database() -> None:
    database = settings.database_url.rsplit("/", 1)[-1].split("?")[0]
    if "eval" not in database:
        sys.exit(
            f"Refusing to run: DATABASE_URL points to database {database!r}. Use a dedicated "
            "evaluation database whose name contains 'eval', e.g.\n"
            "  DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/research_assistant_eval"
        )


async def _setup_db() -> None:
    """Create the evaluation database if needed and apply migrations."""
    import asyncpg
    from alembic import command
    from alembic.config import Config

    url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    base, database = url.rsplit("/", 1)
    conn = await asyncpg.connect(f"{base}/postgres")
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", database)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{database}"')
            print(f"Created database {database}")
    finally:
        await conn.close()

    alembic_cfg = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    await asyncio.to_thread(command.upgrade, alembic_cfg, "head")
    print(f"Database {database} is migrated to head")


async def _run(dataset_path: str, config_path: str, results: Path) -> Path:
    from app.database.connection import AsyncSessionLocal, engine
    from app.embeddings.service import create_embedding_provider
    from app.evaluation.config import load_config
    from app.evaluation.dataset import load_dataset
    from app.evaluation.experiment import run_experiment
    from app.llm.registry import create_llm_registry

    loaded = load_dataset(dataset_path)
    config = load_config(config_path)
    config.apply(settings)
    embedder = create_embedding_provider()
    registry = create_llm_registry()
    try:
        async with AsyncSessionLocal() as db:
            return await run_experiment(
                db, loaded, config, embedder, registry, settings, results, progress=print
            )
    finally:
        await registry.aclose()
        await engine.dispose()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup-db", help="create and migrate the evaluation database")

    validate = sub.add_parser("validate", help="check a dataset file and its PDFs")
    validate.add_argument("dataset")

    run = sub.add_parser("run", help="run an experiment")
    run.add_argument("--dataset", required=True, help="dataset directory or dataset.json")
    run.add_argument("--config", required=True, help="experiment config JSON")
    run.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="results root directory")

    compare = sub.add_parser("compare", help="compare runs side by side (markdown)")
    compare.add_argument("runs", nargs="+", type=Path)
    compare.add_argument("--out", type=Path, help="write the table to this file")

    review = sub.add_parser("review", help="add a filled manual_review.csv to a run's report")
    review.add_argument("run", type=Path)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    if args.command == "setup-db":
        _require_eval_database()
        asyncio.run(_setup_db())
    elif args.command == "validate":
        from app.evaluation.dataset import load_dataset

        loaded = load_dataset(args.dataset)
        ds = loaded.dataset
        unverified = [q.id for q in ds.questions if not q.verified]
        print(
            f"OK: {ds.name} ({ds.kind}) - {len(ds.documents)} documents, {len(ds.questions)} "
            f"questions ({sum(q.answerable for q in ds.questions)} answerable), "
            f"languages: {', '.join(sorted({q.language for q in ds.questions}))}"
        )
        if unverified:
            print(f"Not yet verified ({len(unverified)}): {', '.join(unverified)}")
    elif args.command == "run":
        _require_eval_database()
        out = asyncio.run(_run(args.dataset, args.config, args.results))
        print(f"\nResults: {out}\n")
        print((out / "report.md").read_text(encoding="utf-8"))
    elif args.command == "compare":
        from app.evaluation.report import load_run, render_comparison

        table = render_comparison([load_run(r) for r in args.runs])
        if args.out:
            args.out.write_text(table, encoding="utf-8")
        print(table)
    elif args.command == "review":
        from app.evaluation.report import render_report, summarize_manual_review

        summary_path = args.run / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["manual_review"] = summarize_manual_review(args.run / "manual_review.csv")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (args.run / "report.md").write_text(render_report(summary), encoding="utf-8")
        print(json.dumps(summary["manual_review"], indent=2))


if __name__ == "__main__":
    main()
