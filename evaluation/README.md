# Evaluation

This directory will contain the RAG evaluation framework (Phase 6).

## Planned contents

- `datasets/` — manually verified question/answer pairs with source references
- `results/` — evaluation run outputs (excluded from Git if large)

## Planned metrics

**Retrieval:** Recall@k, Precision@k, MRR
**Generation:** answer correctness, faithfulness, relevance, citation correctness
**Operational:** retrieval latency, total response latency, token usage

See SPEC.md Section 11 for full requirements.
