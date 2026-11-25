"""Evaluation dataset format.

A dataset is a directory holding `dataset.json` and the PDFs it refers to.
Page numbers in the dataset are ONE-BASED, as shown by PDF viewers, because
people write the dataset while reading the PDF; they are converted to the
zero-based indexes used internally (see `RelevantPassage.page_index`).
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DocumentSpec(BaseModel):
    # Short identifier used by questions, e.g. "smith2021".
    id: str = Field(min_length=1)
    # PDF path relative to the dataset directory.
    file: str = Field(min_length=1)


class RelevantPassage(BaseModel):
    document: str
    # One-based PDF page (as in a PDF viewer), not the printed page label.
    pdf_page: int = Field(ge=1)
    # Supporting text copied from the PDF. Optional: without it, any chunk that
    # covers the page counts as relevant (coarser).
    passage: str | None = None

    @property
    def page_index(self) -> int:
        return self.pdf_page - 1


class Question(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    # ISO 639-1 code, e.g. "en", "fi"; lets results be broken down by language.
    language: str = "en"
    # False for questions the documents cannot answer (tests abstention).
    answerable: bool = True
    expected_answer: str | None = None
    relevant: list[RelevantPassage] = []
    # Free-form category, e.g. "factual", "comparison", "numeric", "off-topic".
    category: str | None = None
    # True once a person has checked the question, answer, page and passage.
    verified: bool = False
    notes: str | None = None

    @model_validator(mode="after")
    def _check_answerable(self) -> "Question":
        if self.answerable and not self.relevant:
            raise ValueError(f"question {self.id}: answerable questions need 'relevant' passages")
        if self.answerable and not self.expected_answer:
            raise ValueError(f"question {self.id}: answerable questions need 'expected_answer'")
        if not self.answerable and self.relevant:
            raise ValueError(f"question {self.id}: unanswerable questions must not list passages")
        return self


class Dataset(BaseModel):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    description: str | None = None
    # "manual" = verified by a person; "synthetic" = generated (e.g. the smoke set).
    kind: Literal["manual", "synthetic"] = "manual"
    documents: list[DocumentSpec]
    questions: list[Question]

    @model_validator(mode="after")
    def _check_references(self) -> "Dataset":
        doc_ids = [d.id for d in self.documents]
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("document ids must be unique")
        question_ids = [q.id for q in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("question ids must be unique")
        known = set(doc_ids)
        for q in self.questions:
            for r in q.relevant:
                if r.document not in known:
                    raise ValueError(f"question {q.id}: unknown document {r.document!r}")
        return self


class LoadedDataset(BaseModel):
    dataset: Dataset
    directory: Path
    # SHA-256 of dataset.json plus every PDF, so results identify the exact data.
    fingerprint: str

    def pdf_path(self, document_id: str) -> Path:
        spec = next(d for d in self.dataset.documents if d.id == document_id)
        return self.directory / spec.file


def load_dataset(path: str | Path) -> LoadedDataset:
    """Load and validate a dataset from a directory or a dataset.json path."""
    path = Path(path)
    file = path / "dataset.json" if path.is_dir() else path
    raw = file.read_bytes()
    dataset = Dataset.model_validate(json.loads(raw))

    digest = hashlib.sha256(raw)
    missing = []
    for doc in dataset.documents:
        pdf = file.parent / doc.file
        if not pdf.is_file():
            missing.append(str(pdf))
            continue
        digest.update(pdf.read_bytes())
    if missing:
        raise FileNotFoundError("Dataset PDFs not found: " + ", ".join(missing))
    return LoadedDataset(dataset=dataset, directory=file.parent, fingerprint=digest.hexdigest())
