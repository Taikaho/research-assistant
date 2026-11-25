"""Download the sample papers used in the README's "Try it" section.

The papers are openly licensed (CC BY 4.0) arXiv articles about retrieval and
retrieval-augmented generation. They are downloaded from arXiv, not shipped
with the repository.

Usage (any Python 3.10+, no dependencies):

    python samples/download_samples.py            # into samples/pdfs/
    python samples/download_samples.py --out DIR

Then upload the PDFs in the Library page (or POST /api/v1/documents).
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path
from typing import NamedTuple


class Paper(NamedTuple):
    arxiv_id: str
    filename: str
    title: str


# Every paper below is licensed CC BY 4.0 on arXiv (checked 2026-11-25).
PAPERS: tuple[Paper, ...] = (
    Paper(
        "2402.03216",
        "bge-m3.pdf",
        "M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity "
        "Text Embeddings Through Self-Knowledge Distillation",
    ),
    Paper(
        "2310.11511",
        "self-rag.pdf",
        "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
    ),
    Paper(
        "2112.01488",
        "colbertv2.pdf",
        "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction",
    ),
    Paper(
        "2112.09118",
        "contriever.pdf",
        "Unsupervised Dense Information Retrieval with Contrastive Learning",
    ),
    Paper(
        "2212.03533",
        "e5.pdf",
        "Text Embeddings by Weakly-Supervised Contrastive Pre-training",
    ),
)

USER_AGENT = "research-assistant-samples/1.0"


def download(paper: Paper, out_dir: Path) -> bool:
    """Download one paper; return True if a new file was written."""
    target = out_dir / paper.filename
    if target.exists() and target.stat().st_size > 0:
        print(f"  exists   {target.name}")
        return False

    request = urllib.request.Request(
        f"https://arxiv.org/pdf/{paper.arxiv_id}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if not data.startswith(b"%PDF"):
        raise RuntimeError(f"arXiv did not return a PDF for {paper.arxiv_id}")

    partial = target.with_suffix(".part")
    partial.write_bytes(data)
    partial.replace(target)
    print(f"  saved    {target.name}  ({len(data) // 1024} KB)  {paper.title}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "pdfs",
        help="directory for the PDFs (default: samples/pdfs)",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(PAPERS)} CC BY 4.0 papers from arXiv into {args.out}")
    failures = 0
    for paper in PAPERS:
        try:
            if download(paper, args.out):
                time.sleep(3)  # arXiv asks automated clients to pace requests
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            failures += 1
            print(f"  FAILED   {paper.filename}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
