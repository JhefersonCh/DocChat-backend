"""Socratic tutor: rubric parsing (quotes verified against the chunks), grading, lexical support
of each "covered" judgment and a small SM-2 style scheduler. Pure functions, no I/O."""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from chat_pdf.domain.artifacts import _excerpt, select_chunks  # _excerpt: "[pág. N] text"
from chat_pdf.domain.generation import (
    SECTIONS_CHARS,
    _joined_titles,
    extract_json,
    section_titles,
)
from chat_pdf.domain.models import Chunk
from chat_pdf.domain.tutor_prompts import (
    AVOID,
    GRADE_SYSTEM,
    GRADE_USER,
    QUESTIONS_SYSTEM,
    QUESTIONS_USER,
)

MIN_POINTS, MAX_POINTS = 2, 4
MIN_QUOTE_CHARS = 10
MAX_ANSWER_CHARS = 4000
MAX_QUESTIONS = 8
PASS_SCORE, FAIL_SCORE = 0.8, 0.5
MIN_EASE, MAX_EASE = 1.3, 3.0
_TOKEN = re.compile(r"\w{4,}")
_STOP = frozenset(
    "para como esto este esta estos estas pero porque cuando donde entre desde hasta sobre "
    "tambien cada otro otra otros otras seria puede pueden debe deben tiene tienen the and "
    "that with from this".split()
)


def norm(text: str) -> str:
    """Lowercase, accents stripped, whitespace collapsed: for comparing quotes with chunks."""
    plain = unicodedata.normalize("NFKD", text.lower())
    return " ".join("".join(c for c in plain if not unicodedata.combining(c)).split())


def _clean(value: Any, limit: int) -> str:
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


# --- Generation -------------------------------------------------------------------------------


def question_messages(
    chunks: list[Chunk], count: int, existing: list[str], budget: int = 12000
) -> list[dict[str, str]]:
    picked = select_chunks(chunks, budget)
    user = QUESTIONS_USER.format(
        sections=_joined_titles(section_titles(chunks), SECTIONS_CHARS) or "(sin secciones)",
        excerpts="\n\n".join(map(_excerpt, picked)),
        count=count,
        avoid=AVOID.format(questions=" | ".join(existing[-20:])) if existing else "",
    )
    return [{"role": "system", "content": QUESTIONS_SYSTEM}, {"role": "user", "content": user}]


def _locate(quote: str, page: Any, chunks: list[Chunk]) -> Chunk | None:
    """The chunk that really contains `quote` (the claimed page breaks ties)."""
    wanted = norm(quote).strip(" .…\"'«»")
    if len(wanted) < MIN_QUOTE_CHARS:
        return None
    found = [c for c in chunks if wanted in norm(c.content)]

    def on_page(c: Chunk) -> bool:
        return (
            isinstance(page, int)
            and bool(c.page_start)
            and (c.page_start <= page <= (c.page_end or c.page_start))
        )

    found_on_page = [c for c in found if on_page(c)]
    return (found_on_page or found or [None])[0]


def parse_questions(text: str, chunks: list[Chunk], existing: Sequence[str] = ()) -> list[dict]:
    """Valid questions: 2-4 key points whose quote exists in the document. Page and chunk come
    from the chunk the quote was found in, never from the model's claim."""
    data = extract_json(text)
    items = data.get("questions") if isinstance(data, dict) else data
    seen = {norm(q) for q in existing}
    out: list[dict] = []
    for item in items if isinstance(items, list) else []:
        question = _clean(item.get("question"), 300) if isinstance(item, dict) else ""
        if not question or norm(question) in seen:
            continue
        points = []
        for kp in item.get("key_points") or []:
            point = _clean(kp.get("point"), 240) if isinstance(kp, dict) else ""
            chunk = _locate(str(kp.get("quote") or ""), kp.get("page"), chunks) if point else None
            if chunk is None:
                continue
            points.append({
                "point": point,
                "quote": _clean(kp.get("quote"), 300),
                "page": chunk.page_start,
                "chunk_id": str(chunk.id) if chunk.id else None,
                "document_id": str(chunk.document_id) if chunk.document_id else None,
                "section": chunk.section,
            })  # fmt: skip
        if len(points) >= MIN_POINTS:
            seen.add(norm(question))
            out.append({"question": question, "rubric": points[:MAX_POINTS]})
    return out


# --- Grading ----------------------------------------------------------------------------------


def grade_messages(question: str, rubric: list[dict], answer: str) -> list[dict[str, str]]:
    lines = "\n".join(f"{i}. {p['point']}" for i, p in enumerate(rubric, start=1))
    user = GRADE_USER.format(question=question, rubric=lines, answer=answer)
    return [{"role": "system", "content": GRADE_SYSTEM}, {"role": "user", "content": user}]


@dataclass(frozen=True)
class Judgment:
    covered: list[bool]  # one per rubric point
    feedback: str
    follow_up: str


def parse_grade(text: str, n_points: int) -> Judgment | None:
    data = extract_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("points"), list):
        return None
    covered = [False] * n_points
    for item in data["points"]:
        n = item.get("n") if isinstance(item, dict) else None
        if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= n_points:
            covered[n - 1] = item.get("covered") is True
    return Judgment(covered, _clean(data.get("feedback"), 600), _clean(data.get("follow_up"), 300))


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(norm(text))) - _STOP


def lexical_support(student: str, point: str, quote: str = "") -> str | None:
    """The student sentence sharing the most content words with the key point (or its source
    quote); None when nothing in the answer touches it."""
    wanted = _tokens(point) | _tokens(quote)
    best, best_overlap = None, 0
    for sentence in re.split(r"[.!?;\n]+", student):
        overlap = len(_tokens(sentence) & wanted)
        if overlap > best_overlap:
            best, best_overlap = sentence.strip(), overlap
    return _clean(best, 240) if best else None


def grade(rubric: list[dict], judgment: Judgment, student: str) -> dict:
    """Per-point result. A "yes" with no lexical support at all in the student's text is
    downgraded to "no": the model cannot credit what the student never wrote about."""
    points = []
    for point, said_yes in zip(rubric, judgment.covered, strict=True):
        quote = lexical_support(student, point["point"], point["quote"]) if said_yes else None
        points.append({
            "point": point["point"],
            "covered": quote is not None,
            "student_quote": quote,
            "source_quote": point["quote"],
            "page": point["page"],
            "chunk_id": point["chunk_id"],
            "document_id": point.get("document_id"),
        })  # fmt: skip
    score = round(sum(p["covered"] for p in points) / len(points), 3) if points else 0.0
    return {
        "points": points,
        "score": score,
        "feedback": judgment.feedback,
        "follow_up_question": judgment.follow_up,
    }


# --- Spaced repetition (SM-2 lite) --------------------------------------------------------------


@dataclass(frozen=True)
class Sched:
    ease: float = 2.5
    interval_days: float = 0.0
    reps: int = 0
    lapses: int = 0


def next_schedule(s: Sched, score: float) -> Sched:
    """Failed (< 0.5): back to the start, due now. Partial: tomorrow. Passed (>= 0.8): the
    interval grows 1 -> 3 -> interval * ease."""
    if score < FAIL_SCORE:
        return Sched(max(MIN_EASE, s.ease - 0.2), 0.0, 0, s.lapses + 1)
    if score < PASS_SCORE:
        return Sched(max(MIN_EASE, s.ease - 0.05), 1.0, s.reps, s.lapses)
    reps = s.reps + 1
    interval = 1.0 if reps == 1 else 3.0 if reps == 2 else round(s.interval_days * s.ease, 1)
    return Sched(min(MAX_EASE, s.ease + 0.1), interval, reps, s.lapses)


def due_at(now: datetime, s: Sched) -> datetime:
    return now + timedelta(days=s.interval_days)


def order_for_review(items: list, now: datetime) -> list:
    """Due items first (the earliest due, i.e. the failed ones, before the rest), then the
    upcoming ones by date. Items expose `due_at` and `lapses`."""
    return sorted(items, key=lambda q: (q.due_at > now, q.due_at, -q.lapses))
