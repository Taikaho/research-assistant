"""LLM-as-judge scoring of generated answers.

An auxiliary signal only: judge scores are NOT ground truth (SPEC §11.4). They
help find answers worth a manual look; reported results should be checked by a
person via the manual review sheet.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.llm.base import ChatMessage, GenerationParams, LLMProvider

JUDGE_PROMPT_VERSION = "judge-v1"

JUDGE_SYSTEM = """\
You grade answers produced by a retrieval-augmented research assistant.
You receive a question, a reference answer written by a person, the source passages
the assistant was given, and the assistant's answer. Grade on a 1-5 scale:

- correctness: agreement with the reference answer (5 = same facts, nothing wrong;
  1 = wrong or contradicts it). Different wording or language is fine.
- faithfulness: every claim in the answer is supported by the source passages
  (5 = fully supported; 1 = mostly unsupported or invented).
- relevance: the answer addresses the question directly (5 = fully; 1 = off-topic).

Reply with a single JSON object and nothing else:
{"correctness": <1-5>, "faithfulness": <1-5>, "relevance": <1-5>, \
"explanation": "<one sentence>"}"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
CRITERIA = ("correctness", "faithfulness", "relevance")


@dataclass(frozen=True)
class JudgeScores:
    correctness: int
    faithfulness: int
    relevance: int
    explanation: str


class JudgeParseError(ValueError):
    pass


def build_judge_messages(
    question: str, reference: str, passages: Sequence[str], answer: str
) -> list[ChatMessage]:
    blocks = "\n\n".join(
        f"[{i}]\n<passage>\n{p.strip()}\n</passage>" for i, p in enumerate(passages, start=1)
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference}\n\n"
        f"Source passages:\n{blocks or '(none)'}\n\n"
        f"Assistant's answer:\n{answer}"
    )
    return [ChatMessage("system", JUDGE_SYSTEM), ChatMessage("user", user)]


def parse_judge_reply(text: str) -> JudgeScores:
    match = _JSON_OBJECT.search(text)
    if not match:
        raise JudgeParseError(f"No JSON object in judge reply: {text[:200]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"Invalid JSON in judge reply: {exc}") from exc

    scores: dict[str, int] = {}
    for key in CRITERIA:
        value = data.get(key)
        try:
            number = int(round(float(value)))
        except (TypeError, ValueError) as exc:
            raise JudgeParseError(f"Missing or non-numeric {key!r}: {value!r}") from exc
        if not 1 <= number <= 5:
            raise JudgeParseError(f"{key!r} out of range: {number}")
        scores[key] = number
    return JudgeScores(**scores, explanation=str(data.get("explanation", "")).strip())


async def judge_answer(
    provider: LLMProvider,
    model: str,
    question: str,
    reference: str,
    passages: Sequence[str],
    answer: str,
    max_tokens: int = 300,
) -> JudgeScores:
    response = await provider.generate(
        model,
        build_judge_messages(question, reference, passages, answer),
        GenerationParams(temperature=0.0, max_tokens=max_tokens),
    )
    return parse_judge_reply(response.text)
