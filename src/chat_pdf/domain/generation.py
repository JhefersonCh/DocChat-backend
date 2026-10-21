"""One-shot generations (follow-up suggestions, document overview): prompts, sampling, parsing."""

import json
from dataclasses import dataclass
from typing import Any

from chat_pdf.domain.models import Chunk
from chat_pdf.domain.prompting import strip_citations
from chat_pdf.domain.prompts import (
    OVERVIEW_SYSTEM,
    OVERVIEW_USER_TEMPLATE,
    SUGGESTIONS_SECTIONS,
    SUGGESTIONS_SYSTEM,
    SUGGESTIONS_USER_TEMPLATE,
)

MAX_QUESTIONS = 3
MAX_QUESTION_CHARS = 90
MAX_ANSWER_CHARS = 3000
SAMPLE_BUDGET = 12000  # chars of document text sent to build the overview
EXCERPT_CHARS = 700  # cap per sampled chunk
HEAD_CHUNKS = 3  # always sampled: the start says what the document is
SECTIONS_CHARS = 1500


@dataclass(frozen=True)
class Overview:
    summary: str
    questions: list[str]


def extract_json(text: str) -> Any:
    """First JSON array/object in `text` (tolerates code fences and prose around it)."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char in "[{":
            try:
                return decoder.raw_decode(text, start)[0]
            except ValueError:
                continue
    return None


def clean_questions(items: Any) -> list[str]:
    """Non-empty distinct strings, at most MAX_QUESTIONS (fewer is fine)."""
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        question = " ".join(item.split()) if isinstance(item, str) else ""
        if question and question not in out:
            out.append(question)
    return out[:MAX_QUESTIONS]


def parse_questions(text: str) -> list[str]:
    data = extract_json(text)
    if isinstance(data, dict):  # some models wrap the array: {"questions": [...]}
        data = data.get("questions")
    return clean_questions(data)


def parse_overview(text: str) -> Overview | None:
    data = extract_json(text)
    if not isinstance(data, dict):
        return None
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return None
    return Overview(summary.strip(), clean_questions(data.get("questions")))


def _spread(items: list, n: int) -> list:
    """`n` items evenly spaced over `items`, in order (all of them if n >= len)."""
    if n >= len(items):
        return items
    if n <= 1:
        return items[:n]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def section_titles(chunks: list[Chunk]) -> list[str]:
    return list(dict.fromkeys(c.section for c in chunks if c.section))


def _joined_titles(titles: list[str], limit: int = SECTIONS_CHARS) -> str:
    n = len(titles)
    while n > 1 and len("; ".join(_spread(titles, n))) > limit:
        n -= 1
    return "; ".join(_spread(titles, n))[:limit]


def sample_text(chunks: list[Chunk], budget: int = SAMPLE_BUDGET) -> tuple[str, str]:
    """(sections, excerpts): the first chunks plus evenly spaced ones across the document,
    each capped, so that a long book still fits the budget. Both in document order."""
    sections = _joined_titles(section_titles(chunks))
    slots = max((budget - len(sections)) // (EXCERPT_CHARS + 2), 1)  # 2 = block separator
    head = chunks[:HEAD_CHUNKS]
    picked = head + _spread(chunks[HEAD_CHUNKS:], max(slots - len(head), 0))
    excerpts = "\n\n".join(c.content[:EXCERPT_CHARS] for c in picked[:slots])
    return sections, excerpts


def suggestion_messages(question: str, answer: str, sections: list[str]) -> list[dict[str, str]]:
    shown = _joined_titles(sections, 600)
    user = SUGGESTIONS_USER_TEMPLATE.format(
        question=question,
        answer=strip_citations(answer)[:MAX_ANSWER_CHARS],
        sections=SUGGESTIONS_SECTIONS.format(sections=shown) if shown else "",
        max_chars=MAX_QUESTION_CHARS,
    )
    return [{"role": "system", "content": SUGGESTIONS_SYSTEM}, {"role": "user", "content": user}]


def overview_messages(chunks: list[Chunk]) -> list[dict[str, str]]:
    sections, excerpts = sample_text(chunks)
    user = OVERVIEW_USER_TEMPLATE.format(
        sections=sections or "(sin secciones)", excerpts=excerpts, max_chars=MAX_QUESTION_CHARS
    )
    return [{"role": "system", "content": OVERVIEW_SYSTEM}, {"role": "user", "content": user}]
