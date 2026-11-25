"""Aggregate per-question records into summaries, reports and comparisons."""

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from app.evaluation.metrics import (
    best_balanced_threshold,
    distribution,
    gate_curve,
    recall_preserving_threshold,
    safe_mean,
)

# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_retrieval(
    records: Sequence[dict[str, Any]],
    k_values: Sequence[int],
    current_gate: float,
    gate_min_pass_rate: float = 0.95,
) -> dict[str, Any]:
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    def means(rows: Sequence[dict[str, Any]]) -> dict[str, float | None]:
        keys = [f"{m}@{k}" for m in ("recall", "precision", "hit") for k in k_values]
        out = {key: safe_mean([r["metrics"][key] for r in rows]) for key in keys}
        out["mrr"] = safe_mean([r["metrics"]["rr"] for r in rows])
        return out

    by_language = {
        lang: {"n": len(rows), **means(rows)}
        for lang in sorted({r["language"] for r in answerable})
        if (rows := [r for r in answerable if r["language"] == lang])
    }

    def top(r: dict[str, Any]) -> float:
        return float(r["top_score"]) if r["top_score"] is not None else -1.0

    relevant_scores = [h["score"] for r in answerable for h in r["results"] if h["matched"]]
    irrelevant_scores = [h["score"] for r in answerable for h in r["results"] if not h["matched"]]
    answerable_top = [top(r) for r in answerable]
    unanswerable_top = [top(r) for r in unanswerable]

    gate: dict[str, Any] = {"current_threshold": current_gate}
    if answerable_top and unanswerable_top:
        curve = gate_curve(answerable_top, unanswerable_top)
        recommended = recall_preserving_threshold(curve, gate_min_pass_rate)
        balanced = best_balanced_threshold(curve)
        current = gate_curve(answerable_top, unanswerable_top, current_gate, current_gate, 1.0)[0]
        gate.update(
            {
                "current": _gate_point(current),
                "min_pass_rate": gate_min_pass_rate,
                "recommended": _gate_point(recommended) if recommended else None,
                "best_balanced": _gate_point(balanced) if balanced else None,
                "curve": [_gate_point(p) for p in curve],
            }
        )
    else:
        gate["note"] = "Calibration needs both answerable and unanswerable questions."

    return {
        "questions": len(records),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "metrics": means(answerable) if answerable else {},
        "by_language": by_language,
        "scores": {
            "relevant_chunks": distribution(relevant_scores),
            "irrelevant_chunks": distribution(irrelevant_scores),
            "top1_answerable": distribution(
                [r["top_score"] for r in answerable if r["top_score"] is not None]
            ),
            "top1_unanswerable": distribution(
                [r["top_score"] for r in unanswerable if r["top_score"] is not None]
            ),
        },
        "gate": gate,
        "latency_ms": distribution([r["latency_ms"] for r in records]),
    }


def _gate_point(p: Any) -> dict[str, float]:
    return {
        "threshold": p.threshold,
        "answerable_pass_rate": p.answerable_pass_rate,
        "unanswerable_reject_rate": p.unanswerable_reject_rate,
        "balanced_accuracy": p.balanced_accuracy,
    }


def _generation_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    answered = [r for r in answerable if not r["abstained"]]
    cited_total = sum(r["cited_count"] for r in ok)
    cited_relevant = sum(r["cited_relevant_count"] for r in ok)
    return {
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "answerable": len(answerable),
        "answer_rate": _ratio(len(answered), len(answerable)),
        "false_abstention_rate": _ratio(len(answerable) - len(answered), len(answerable)),
        "context_hit_rate": safe_mean([float(r["context_hit"]) for r in answerable]),
        "citation_hit_rate": safe_mean([float(r["citation_hit"]) for r in answered]),
        # Micro-average over all citations: share pointing at a relevant passage.
        "citation_precision": _ratio(cited_relevant, cited_total),
        "uncited_answer_rate": safe_mean([float(r["uncited_answer"]) for r in answered]),
        "invalid_citation_markers": sum(len(r["invalid_citation_numbers"]) for r in ok),
        "unanswerable": len(unanswerable),
        "correct_abstention_rate": _ratio(
            sum(1 for r in unanswerable if r["abstained"]), len(unanswerable)
        ),
    }


def summarize_generation(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if "error" not in r]
    judged = [r["judge"] for r in ok if "judge" in r and "error" not in r["judge"]]
    usage = [r["usage"] for r in ok if r.get("usage")]

    def lat(key: str) -> dict[str, Any]:
        return distribution(
            [r["latency_ms"][key] for r in ok if r["latency_ms"].get(key) is not None]
        )

    return {
        **_generation_block(records),
        "by_language": {
            lang: _generation_block([r for r in records if r["language"] == lang])
            for lang in sorted({r["language"] for r in records})
        },
        "latency_ms": {
            "retrieval": lat("retrieval"),
            "generation": lat("generation"),
            "total": lat("total"),
        },
        "tokens": {
            "prompt_mean": safe_mean([u["prompt_tokens"] for u in usage]),
            "completion_mean": safe_mean([u["completion_tokens"] for u in usage]),
        },
        "judge": {
            "note": "LLM-judge scores (1-5) are an auxiliary signal, not ground truth.",
            "judged": len(judged),
            "failures": sum(1 for r in ok if "judge" in r and "error" in r["judge"]),
            "correctness": safe_mean([j["correctness"] for j in judged]),
            "faithfulness": safe_mean([j["faithfulness"] for j in judged]),
            "relevance": safe_mean([j["relevance"] for j in judged]),
        },
    }


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


MANUAL_COLUMNS = ["manual_correct", "manual_faithful", "manual_citations_ok", "manual_notes"]


def write_manual_review(path: Path, generation: Sequence[dict[str, Any]]) -> None:
    """A sheet for a person to grade answers (SPEC §11.4: verify important results manually).

    Fill the manual_* columns with 1 (yes) / 0 (no), then run
    `python -m app.evaluation review <run dir>`.
    """
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "language",
                "answerable",
                "question",
                "expected_answer",
                "answer",
                "abstained",
                "cited_sources",
                "judge_correctness",
                "judge_faithfulness",
                *MANUAL_COLUMNS,
            ]
        )
        for r in generation:
            cited = "; ".join(
                f"[{s['number']}] {s['document']} p.{','.join(map(str, s['pdf_pages']))}"
                + (" (relevant)" if s["relevant"] else "")
                for s in r.get("sources", [])
                if s["cited"]
            )
            judge = r.get("judge", {})
            writer.writerow(
                [
                    r["id"],
                    r["language"],
                    r["answerable"],
                    r["question"],
                    r.get("expected_answer") or "",
                    r.get("answer", r.get("error", "")),
                    r.get("abstained", ""),
                    cited,
                    judge.get("correctness", ""),
                    judge.get("faithfulness", ""),
                    "",
                    "",
                    "",
                    "",
                ]
            )


def summarize_manual_review(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    def rate(column: str) -> dict[str, Any]:
        values = [row[column].strip() for row in rows if row.get(column, "").strip() in {"0", "1"}]
        return {"graded": len(values), "rate": _ratio(values.count("1"), len(values))}

    return {
        "rows": len(rows),
        "correct": rate("manual_correct"),
        "faithful": rate("manual_faithful"),
        "citations_ok": rate("manual_citations_ok"),
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.1f} %"


def table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    """Markdown table lines."""
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _header(cfg: dict[str, Any]) -> list[str]:
    ds, emb, st = cfg["dataset"], cfg["embedding_model"], cfg["settings"]
    commit = (cfg["git"].get("commit") or "unknown")[:12]
    lines = [
        f"# Evaluation run: {cfg['experiment']['name']}",
        "",
        f"- Started: {cfg['started_at']} · duration {fmt(cfg.get('duration_s'), 1)} s",
        f"- Dataset: `{ds['name']}` ({ds['kind']}, {ds['questions']} questions, "
        f"{ds['verified_questions']} verified) · fingerprint `{ds['fingerprint'][:12]}`",
        f"- Embedding model: `{emb['name']}` @ `{(emb['version'] or 'unknown')[:12]}` "
        f"({emb['dimensions']} dims)",
        f"- Chunking: {st['chunk_size']} chars, overlap {st['chunk_overlap']}",
        f"- Git: `{commit}`" + (" (uncommitted changes)" if cfg["git"].get("dirty") else ""),
    ]
    if cfg.get("llm"):
        llm = cfg["llm"]
        where = "local" if llm["is_local"] else "cloud"
        lines.append(
            f"- LLM: `{llm['provider']}` / `{llm['model']}` ({where}), "
            f"prompt `{cfg['prompt_version']}`, relevance gate {st['chat_min_relevance']}"
        )
    if cfg.get("judge"):
        judge = cfg["judge"]
        lines.append(
            f"- Judge: `{judge['provider']}` / `{judge['model']}` (`{judge['prompt_version']}`)"
        )
    if ds["kind"] == "synthetic":
        lines += [
            "",
            "> Synthetic dataset: a smoke test of the pipeline, "
            "not evidence of quality on real articles.",
        ]
    return lines


def _retrieval_section(ret: dict[str, Any], k_values: Sequence[int]) -> list[str]:
    m = ret["metrics"]
    kmax = max(k_values)
    lines = ["", f"## Retrieval ({ret['answerable']} answerable questions)", ""]
    lines += table(
        ["k", "Recall@k", "Precision@k", "Hit@k"],
        [
            [k, fmt(m.get(f"recall@{k}")), fmt(m.get(f"precision@{k}")), fmt(m.get(f"hit@{k}"))]
            for k in k_values
        ],
    )
    lines += [
        "",
        f"MRR: **{fmt(m.get('mrr'))}** · retrieval latency median "
        f"{fmt(ret['latency_ms']['median'], 0)} ms",
    ]
    if ret["by_language"]:
        lines.append("")
        lines += table(
            ["Language", "n", f"Recall@{kmax}", "Hit@1", "MRR"],
            [
                [lang, v["n"], fmt(v.get(f"recall@{kmax}")), fmt(v.get("hit@1")), fmt(v.get("mrr"))]
                for lang, v in ret["by_language"].items()
            ],
        )

    s = ret["scores"]
    groups = [
        ("relevant_chunks", "Relevant chunks"),
        ("irrelevant_chunks", "Irrelevant chunks"),
        ("top1_answerable", "Best chunk, answerable"),
        ("top1_unanswerable", "Best chunk, unanswerable"),
    ]
    lines += ["", "### Similarity scores", ""]
    lines += table(
        ["Group", "n", "min", "median", "max"],
        [
            [label, s[key]["n"], fmt(s[key]["min"]), fmt(s[key]["median"]), fmt(s[key]["max"])]
            for key, label in groups
        ],
    )

    gate = ret["gate"]
    lines += ["", "### Relevance gate (CHAT_MIN_RELEVANCE)", ""]
    if gate.get("current"):
        points = [
            (label, gate[key])
            for key, label in (
                ("current", "current"),
                ("recommended", "recommended"),
                ("best_balanced", "best balanced accuracy"),
            )
            if gate.get(key)
        ]
        lines += table(
            ["Threshold", "Answerable passed", "Unanswerable rejected", "Balanced accuracy"],
            [
                [
                    f"{fmt(p['threshold'], 2)} ({label})",
                    pct(p["answerable_pass_rate"]),
                    pct(p["unanswerable_reject_rate"]),
                    fmt(p["balanced_accuracy"]),
                ]
                for label, p in points
            ],
        )
        lines += [
            "",
            f"Recommended = highest threshold passing at least {pct(gate['min_pass_rate'])} of "
            "answerable questions. Unanswerable questions that pass are left to the model's "
            "insufficient-evidence check.",
        ]
    else:
        lines.append(gate.get("note", "Not computed."))
    return lines


def _generation_section(gen: dict[str, Any]) -> list[str]:
    lat, tok = gen["latency_ms"], gen["tokens"]
    lines = ["", f"## Generation ({gen['questions']} questions, {gen['errors']} errors)", ""]
    lines += table(
        ["Metric", "Value"],
        [
            ["Answer rate (answerable)", pct(gen["answer_rate"])],
            ["False abstention rate", pct(gen["false_abstention_rate"])],
            ["Correct abstention (unanswerable)", pct(gen["correct_abstention_rate"])],
            ["Relevant passage given to the LLM", pct(gen["context_hit_rate"])],
            ["Answers citing a relevant passage", pct(gen["citation_hit_rate"])],
            ["Citation precision", pct(gen["citation_precision"])],
            ["Answers without citations", pct(gen["uncited_answer_rate"])],
            ["Invalid citation markers removed", gen["invalid_citation_markers"]],
            [
                "Total latency median / p90",
                f"{fmt(lat['total']['median'], 0)} / {fmt(lat['total']['p90'], 0)} ms",
            ],
            ["Generation latency median", f"{fmt(lat['generation']['median'], 0)} ms"],
            [
                "Tokens mean (prompt / completion)",
                f"{fmt(tok['prompt_mean'], 0)} / {fmt(tok['completion_mean'], 0)}",
            ],
        ],
    )
    if len(gen["by_language"]) > 1:
        lines.append("")
        lines += table(
            ["Language", "Answer rate", "Citation hit", "Correct abstention"],
            [
                [
                    lang,
                    pct(v["answer_rate"]),
                    pct(v["citation_hit_rate"]),
                    pct(v["correct_abstention_rate"]),
                ]
                for lang, v in gen["by_language"].items()
            ],
        )
    j = gen["judge"]
    if j["judged"] or j["failures"]:
        lines += [
            "",
            f"LLM judge ({j['judged']} answers, {j['failures']} failures; "
            "auxiliary, not ground truth): "
            f"correctness {fmt(j['correctness'], 2)}, faithfulness {fmt(j['faithfulness'], 2)}, "
            f"relevance {fmt(j['relevance'], 2)} (1–5).",
        ]
    return lines


def render_report(run: dict[str, Any]) -> str:
    cfg = run["config"]
    lines = _header(cfg)

    issues = run.get("dataset_issues") or []
    lines += ["", "## Dataset checks", ""]
    lines += [f"- **{i['question']}**: {i['problem']}" for i in issues] or ["No problems found."]

    lines += _retrieval_section(run["retrieval"], cfg["experiment"]["retrieval"]["k_values"])
    if run.get("generation"):
        lines += _generation_section(run["generation"])

    manual = run.get("manual_review")
    if manual:
        labels = (
            ("correct", "Correct"),
            ("faithful", "faithful"),
            ("citations_ok", "citations OK"),
        )
        parts = [
            f"{label} {pct(manual[key]['rate'])} ({manual[key]['graded']} graded)"
            for key, label in labels
        ]
        lines += ["", "## Manual review", "", " · ".join(parts)]
    return "\n".join(lines) + "\n"


def load_run(directory: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    return data


def render_comparison(runs: Sequence[dict[str, Any]]) -> str:
    """Side-by-side table of the key metrics of several runs."""

    def row(label: str, getter: Any) -> str:
        cells = []
        for run in runs:
            try:
                cells.append(getter(run))
            except (KeyError, TypeError):
                cells.append("–")
        return f"| {label} | " + " | ".join(cells) + " |"

    def kmax(run: dict[str, Any]) -> int:
        return int(max(run["config"]["experiment"]["retrieval"]["k_values"]))

    header = "| Metric | " + " | ".join(r["config"]["experiment"]["name"] for r in runs) + " |"
    lines = [
        "# Run comparison",
        "",
        header,
        "|---|" + "---|" * len(runs),
        row(
            "Dataset",
            lambda r: (
                f"{r['config']['dataset']['name']} ({r['config']['dataset']['fingerprint'][:8]})"
            ),
        ),
        row("Embedding model", lambda r: r["config"]["embedding_model"]["name"]),
        row("LLM", lambda r: f"{r['config']['llm']['provider']}/{r['config']['llm']['model']}"),
        row("Hit@1", lambda r: fmt(r["retrieval"]["metrics"]["hit@1"])),
        row("Recall@k (largest k)", lambda r: fmt(r["retrieval"]["metrics"][f"recall@{kmax(r)}"])),
        row("MRR", lambda r: fmt(r["retrieval"]["metrics"]["mrr"])),
    ]
    languages = sorted({lang for r in runs for lang in r["retrieval"]["by_language"]})
    for lang in languages:
        lines.append(
            row(
                f"MRR ({lang})",
                lambda r, lang=lang: fmt(r["retrieval"]["by_language"][lang]["mrr"]),
            )
        )
    lines += [
        row(
            "Recommended gate", lambda r: fmt(r["retrieval"]["gate"]["recommended"]["threshold"], 2)
        ),
        row(
            "Gate balanced accuracy",
            lambda r: fmt(r["retrieval"]["gate"]["recommended"]["balanced_accuracy"]),
        ),
        row("Answer rate", lambda r: pct(r["generation"]["answer_rate"])),
        row("Correct abstention", lambda r: pct(r["generation"]["correct_abstention_rate"])),
        row("Citation hit rate", lambda r: pct(r["generation"]["citation_hit_rate"])),
        row("Citation precision", lambda r: pct(r["generation"]["citation_precision"])),
        row("Uncited answers", lambda r: pct(r["generation"]["uncited_answer_rate"])),
        row("Judge correctness", lambda r: fmt(r["generation"]["judge"]["correctness"], 2)),
        row("Judge faithfulness", lambda r: fmt(r["generation"]["judge"]["faithfulness"], 2)),
        row(
            "Total latency median (ms)",
            lambda r: fmt(r["generation"]["latency_ms"]["total"]["median"], 0),
        ),
    ]
    return "\n".join(lines) + "\n"
