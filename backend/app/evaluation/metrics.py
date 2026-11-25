"""Metric definitions. Pure functions over relevance judgements, no I/O.

Retrieval metrics take, per question, the ranked list of sets of relevant-passage
indexes each retrieved chunk matched (empty set = not relevant) and the number
of relevant passages in the dataset.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, median


def recall_at_k(ranked: Sequence[set[int]], relevant_count: int, k: int) -> float:
    """Share of the question's relevant passages found in the top k chunks."""
    if relevant_count == 0:
        return 0.0
    found: set[int] = set().union(*ranked[:k]) if ranked[:k] else set()
    return len(found) / relevant_count


def precision_at_k(ranked: Sequence[set[int]], k: int) -> float:
    """Share of the top k positions holding a relevant chunk.

    Divides by k even when fewer chunks were returned, so returning nothing is
    not rewarded.
    """
    return sum(1 for m in ranked[:k] if m) / k


def hit_at_k(ranked: Sequence[set[int]], k: int) -> float:
    return 1.0 if any(ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[set[int]]) -> float:
    for rank, m in enumerate(ranked, start=1):
        if m:
            return 1.0 / rank
    return 0.0


def safe_mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return mean(present) if present else None


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "median": None, "p90": None, "max": None, "mean": None}
    return {
        "n": len(values),
        "min": min(values),
        "median": median(values),
        "p90": percentile(values, 90),
        "max": max(values),
        "mean": mean(values),
    }


# ---------------------------------------------------------------------------
# Relevance gate calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatePoint:
    threshold: float
    # Answerable questions whose best passage passes the gate.
    answerable_pass_rate: float
    # Unanswerable questions whose best passage is rejected by the gate.
    unanswerable_reject_rate: float

    @property
    def balanced_accuracy(self) -> float:
        return (self.answerable_pass_rate + self.unanswerable_reject_rate) / 2


def gate_curve(
    answerable_top: Sequence[float],
    unanswerable_top: Sequence[float],
    start: float = 0.2,
    stop: float = 0.95,
    step: float = 0.01,
) -> list[GatePoint]:
    """Gate outcomes for each threshold, from the best retrieval score per question.

    Questions with no retrieved chunk should be passed as -1.0 (always rejected).
    """
    points = []
    steps = round((stop - start) / step)
    for i in range(steps + 1):
        t = round(start + i * step, 4)
        a = (
            sum(1 for s in answerable_top if s >= t) / len(answerable_top)
            if answerable_top
            else 0.0
        )
        u = (
            sum(1 for s in unanswerable_top if s < t) / len(unanswerable_top)
            if unanswerable_top
            else 0.0
        )
        points.append(GatePoint(t, a, u))
    return points


def best_balanced_threshold(points: Sequence[GatePoint]) -> GatePoint | None:
    """Threshold with the best balanced accuracy (middle of the best plateau)."""
    if not points:
        return None
    best = max(p.balanced_accuracy for p in points)
    plateau = [p for p in points if math.isclose(p.balanced_accuracy, best)]
    return plateau[len(plateau) // 2]


def recall_preserving_threshold(
    points: Sequence[GatePoint], min_pass_rate: float
) -> GatePoint | None:
    """Highest threshold that still passes at least `min_pass_rate` of answerable questions.

    This is the recommended gate. Its errors are asymmetric: rejecting an
    answerable question leaves the user without an answer, while an unanswerable
    question that passes still meets the model's own insufficient-evidence check.
    So the gate should only remove what is clearly unrelated, and among the
    thresholds that keep answerable questions, the highest rejects the most.
    """
    eligible = [p for p in points if p.answerable_pass_rate >= min_pass_rate]
    return max(eligible, key=lambda p: p.threshold) if eligible else None
