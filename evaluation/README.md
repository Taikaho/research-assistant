# Evaluation

Reproducible evaluation of the RAG pipeline (SPEC §11.3–11.4, Phase 6). The
framework lives in `backend/app/evaluation/` and runs the production code paths:
ingestion, retrieval and the chat pipeline (`answer_question`), not copies of them.

```
evaluation/
├── configs/            experiment configs (models and settings to compare)
├── datasets/
│   └── smoke/          synthetic smoke-test set (generated PDFs, 23 questions)
└── results/            run outputs (git-ignored; they contain document text)
```

## What is measured

| Area | Metric | Definition |
|---|---|---|
| Retrieval | Recall@k | Share of a question's relevant passages found in the top k chunks |
| | Precision@k | Share of the top k positions holding a relevant chunk (divided by k) |
| | Hit@k | Question has at least one relevant chunk in the top k |
| | MRR | Mean of 1 / rank of the first relevant chunk |
| | Score distributions | Cosine similarity of relevant vs irrelevant chunks, best chunk of answerable vs unanswerable questions |
| Relevance gate | Recommended `CHAT_MIN_RELEVANCE` | Highest threshold still passing ≥ 95 % of answerable questions (configurable); the balanced-accuracy optimum is shown too |
| Generation | Answer rate / false abstention | Answerable questions answered vs refused |
| | Correct abstention | Unanswerable questions refused (gate or model) |
| | Context hit | A relevant passage was among those given to the LLM |
| | Citation hit | The answer cites at least one relevant passage |
| | Citation precision | Share of all citations that point to a relevant passage |
| | Uncited answers | Answers without any `[n]` citation |
| | LLM judge (optional) | Correctness, faithfulness, relevance 1–5 — **auxiliary, not ground truth** |
| Operational | Latency | Retrieval, generation and total (median, p90) |
| | Tokens | Prompt / completion tokens per answer |
| | Identification | Embedding model + weights revision, LLM provider + model, prompt version |

A retrieved chunk counts as **relevant** when it comes from the passage's
document, covers its PDF page and contains at least 50 % of the passage text
(after normalising whitespace, hyphenation and case). Without a passage, page
coverage alone counts.

Every run records its configuration: the experiment config, the resolved
settings (chunking, top-k, gate, temperature …), embedding model name, weights
revision and dimension, LLM and judge, prompt versions, dataset fingerprint
(SHA-256 of `dataset.json` + PDFs), git commit (and whether the tree was dirty)
and package versions.

## Setup

Runs switch the active embedding model and re-embed every chunk in their
database, so they use a **dedicated evaluation database** (its name must
contain `eval`; the CLI refuses anything else):

```bash
cd backend
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/research_assistant_eval
export STORAGE_PATH=../storage/eval            # keeps evaluation PDF copies apart
python -m app.evaluation setup-db               # create + migrate
```

LLM runs need Ollama with the models named in the config (e.g.
`ollama pull llama3.2:3b`, `ollama pull qwen2.5:7b`).

## Running

```bash
python -m app.evaluation validate ../evaluation/datasets/smoke
python -m app.evaluation run --dataset ../evaluation/datasets/smoke \
    --config ../evaluation/configs/retrieval_bge-m3.json
python -m app.evaluation compare ../evaluation/results/<run-a> ../evaluation/results/<run-b>
```

Each run writes `evaluation/results/<timestamp>_<config>/`:

| File | Contents |
|---|---|
| `report.md` | Human-readable summary |
| `summary.json` | All aggregate metrics + the run configuration |
| `config.json` | Reproducibility record |
| `retrieval.jsonl` | Per question: ranked chunks, scores, which passages matched |
| `generation.jsonl` | Per question: answer, sources, citations, latency, tokens, judge |
| `manual_review.csv` | Grading sheet for a person |

### Manual review

LLM-judge scores are not ground truth (SPEC §11.4). Open `manual_review.csv`,
fill `manual_correct`, `manual_faithful` and `manual_citations_ok` with `1`/`0`
(and notes), then:

```bash
python -m app.evaluation review ../evaluation/results/<run>
```

The manual rates are added to `summary.json` and `report.md`.

## Experiment configs

```json
{
  "name": "bge-m3_qwen2.5-7b",
  "embedding": { "model_name": "BAAI/bge-m3", "query_prefix": "" },
  "chunking": { "chunk_size": 2000, "chunk_overlap": 400 },
  "retrieval": { "k_values": [1, 3, 5], "passage_match_threshold": 0.5, "gate_min_pass_rate": 0.95 },
  "generation": { "provider": "ollama", "model": "qwen2.5:7b", "temperature": 0.1, "min_relevance": 0.45 },
  "judge": { "enabled": true, "provider": "ollama", "model": "qwen2.5:7b" }
}
```

Omitted fields use the application settings. `"generation": {"enabled": false}`
gives a fast retrieval-only run (embedding model and gate comparisons). Changing
`chunking` re-processes the dataset documents automatically. A cloud provider
(`"provider": "openai"`) works when `OPENAI_API_KEY` is set; the question and
retrieved passages are then sent to that provider.

## Writing the real dataset

SPEC §11.3 asks for 30–50 manually verified questions over real articles.
Create `evaluation/datasets/<name>/` with the PDFs (git-ignored) and a
`dataset.json`:

```json
{
  "name": "articles-v1",
  "kind": "manual",
  "description": "What the set covers",
  "documents": [{ "id": "smith2021", "file": "smith2021.pdf" }],
  "questions": [
    {
      "id": "q01",
      "language": "en",
      "category": "numeric",
      "question": "What accuracy did LDA reach?",
      "expected_answer": "91.3 % overall accuracy.",
      "relevant": [
        { "document": "smith2021", "pdf_page": 7, "passage": "LDA achieved an overall accuracy of 91.3 percent" }
      ],
      "verified": true
    },
    { "id": "q40", "question": "Which CNN architecture was used?", "answerable": false, "verified": true }
  ]
}
```

Guidelines:

- `pdf_page` is **one-based, as shown by the PDF viewer** — not the printed page label.
- Copy `passage` from the PDF: a sentence or two that supports the answer.
  Several `relevant` entries are allowed (answers spread over passages/documents).
- Include ~20 % **unanswerable** questions: in-domain but not covered by the
  articles, and off-topic. They calibrate the relevance gate and test abstention.
- Mix question languages if users will ask in several (`language: "fi"`).
- Set `verified: true` only after checking question, answer, page and passage.
- `validate` checks the file; every run also checks each passage against the
  extracted text and lists mismatches under "Dataset checks" — fix those first,
  they make the retrieval metrics wrong.

## Smoke dataset

`datasets/smoke/` holds three short **fictional** articles generated by
`generate_pdfs.py` (run it once: `python ../evaluation/datasets/smoke/generate_pdfs.py`
from `backend/`) and 23 questions: 12 English and 6 Finnish answerable, 5
unanswerable. It proves the pipeline works end to end and gives a first look at
model differences; with ~2 chunks per document, retrieval is far easier than on
real articles. **Its numbers are not evidence of quality on real data.**

## Results

Smoke dataset, CPU only, 2026-11-18. Full reports are reproducible with the
configs above.

| Metric | retrieval_bge-small | retrieval_bge-m3 | bge-m3_llama3.2-3b | bge-m3_qwen2.5-7b |
|---|---|---|---|---|
| Dataset | smoke (b64920b5) | smoke (b64920b5) | smoke (b64920b5) | smoke (b64920b5) |
| Embedding model | BAAI/bge-small-en-v1.5 | BAAI/bge-m3 | BAAI/bge-m3 | BAAI/bge-m3 |
| LLM | – | – | ollama/llama3.2:3b | ollama/qwen2.5:7b |
| Hit@1 | 0.889 | 0.944 | 0.944 | 0.944 |
| Recall@k (largest k) | 1.000 | 1.000 | 1.000 | 1.000 |
| MRR | 0.928 | 0.963 | 0.963 | 0.963 |
| MRR (en) | 1.000 | 1.000 | 1.000 | 1.000 |
| MRR (fi) | 0.783 | 0.889 | 0.889 | 0.889 |
| Recommended gate | 0.49 | 0.45 | 0.45 | 0.45 |
| Gate balanced accuracy | 0.600 | 0.700 | 0.700 | 0.700 |
| Answer rate | – | – | 100.0 % | 94.4 % |
| Correct abstention | – | – | 60.0 % | 100.0 % |
| Citation hit rate | – | – | 50.0 % | 76.5 % |
| Citation precision | – | – | 75.0 % | 86.4 % |
| Uncited answers | – | – | 38.9 % | 23.5 % |
| Judge correctness | – | – | 4.89 | 4.71 |
| Judge faithfulness | – | – | 4.89 | 4.71 |
| Total latency median (ms) | – | – | 3669 | 992 |

Observations (synthetic data; confirm on the real dataset):

- **Embedding model.** bge-m3 retrieves Finnish questions better (MRR 0.889 vs
  0.783) and separates unrelated text far more clearly (irrelevant chunks max
  0.52 vs 0.68). With bge-small, Finnish answerable questions (0.49-0.58)
  scored like a Finnish off-topic question (0.48). Basis for switching the
  default to bge-m3 with `CHAT_MIN_RELEVANCE=0.45`
  (`docs/decisions/0002-multilingual-embedding-model.md`).
- **llama3.2:3b** answers everything but omits citations in 39 % of answers,
  abstains in prose without the `INSUFFICIENT_EVIDENCE` marker (counted as an
  answer), and invented a fact for one unanswerable question ("L-band radar was
  used"; the article only names it as future work).
- **qwen2.5:7b** follows the protocol much better: all 5 unanswerable questions
  refused, citation hit rate 76.5 %, precision 86.4 %. It wrongly refused one
  Finnish question whose relevant passage was ranked only third, and still
  leaves 23.5 % of answers uncited (mostly Finnish answers).
- The LLM judge rated both models ≈ 4.7–4.9 / 5 and did not reveal these
  differences — the protocol metrics and the manual review did. This is why
  judge scores are reported as auxiliary only.
- Latency was measured on this CPU-only machine with models already loaded; it
  depends heavily on hardware.
