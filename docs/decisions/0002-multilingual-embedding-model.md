# 0002 — Switch the default embedding model to BAAI/bge-m3

- Status: accepted (gate value provisional)
- Date: 2026-11-18
- Phase: 6 (Evaluation)

## Context

Phases 3–5 used `BAAI/bge-small-en-v1.5`, an English-only model. In the Phase 4
end-to-end test a Finnish question about an English article scored 0.557, the
same as unrelated text (0.554–0.560), so the relevance gate
(`CHAT_MIN_RELEVANCE=0.6`) rejected every Finnish question. Users ask in Finnish.

Candidates: `BAAI/bge-m3` (1024 dims, 8192 tokens, ~2.2 GB, no query prefix) and
`intfloat/multilingual-e5-base` (768 dims, 512 tokens, ~1.1 GB, needs
`query:`/`passage:` prefixes, which would require passage-prefix support).

## Decision

Make `BAAI/bge-m3` the default (`EMBEDDING_QUERY_PREFIX=""`) and lower
`CHAT_MIN_RELEVANCE` to 0.45, both chosen with the Phase 6 evaluation framework
(retrieval-only runs `retrieval_bge-small` vs `retrieval_bge-m3` on the smoke set).

## Evidence (synthetic smoke set, 18 answerable / 5 unanswerable questions)

| | bge-small-en-v1.5 | bge-m3 |
|---|---|---|
| MRR (English / Finnish) | 1.000 / 0.783 | 1.000 / 0.889 |
| Finnish answerable, best score | 0.49–0.58 | 0.48–0.69 |
| Off-topic, best score | 0.48–0.51 | 0.29–0.36 |
| Irrelevant chunks, max score | 0.684 | 0.515 |
| Recommended gate (≥ 95 % answerable pass) | 0.49 (rejects 1/5 unanswerable) | 0.45 (rejects 2/5) |
| Old gate 0.6: answerable passed | 66.7 % (no Finnish) | 50.0 % |

bge-m3 separates relevant from irrelevant text clearly in both languages,
whereas with bge-small Finnish questions score like off-topic ones. The score
scale differs between models, which is why the gate had to change with it:
keeping 0.6 would have rejected half of the answerable questions.

## Consequences

- Existing libraries must be re-indexed once (`POST /api/v1/embeddings/reindex`
  or the Library page banner); no migration (the vector column is dimensionless).
- ~2.2 GB download and ~17× the parameters of bge-small (568 M vs 33 M), so
  ingestion is slower on CPU. Query retrieval stayed around 0.1 s (median, incl.
  database) on the test machine; ingestion time on long articles was not measured.
- The gate value comes from a small synthetic set. **Recalibrate it on the
  manually verified dataset of real articles** (the recommendation is printed in
  every evaluation report) before relying on it.
- Unanswerable questions that are close to the documents' topic still pass the
  gate; refusing them is the LLM's job (`INSUFFICIENT_EVIDENCE`), measured as
  "correct abstention" in the generation evaluation.
