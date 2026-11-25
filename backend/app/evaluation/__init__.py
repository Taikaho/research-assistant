"""RAG evaluation framework (Phase 6).

Measures retrieval (Recall@k, Precision@k, MRR), the relevance gate,
answer generation (abstention, citation correctness, optional LLM judge) and
operational figures (latency, tokens) against a manually verified dataset.
Run with `python -m app.evaluation --help`.
"""
