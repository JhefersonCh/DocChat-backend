"""Message batches per artifact kind: one call for most, up to MAX_CALLS for audit/entity_graph."""

from typing import Any

from chat_pdf.domain.artifact_prompts import ARTIFACT_SYSTEM, CONTEXT, INSTRUCTIONS
from chat_pdf.domain.artifacts import (
    ARTIFACT_BUDGET,
    EXCERPT_CHARS,
    _excerpt,
    artifact_messages,
    select_chunks,
)
from chat_pdf.domain.generation import SECTIONS_CHARS, _joined_titles, section_titles
from chat_pdf.domain.models import Chunk

MAX_CALLS = 3
MIN_DIGITS = 3
PER_CALL = max(ARTIFACT_BUDGET // (EXCERPT_CHARS + 2), 1)


def numeric_chunks(chunks: list[Chunk], limit: int) -> list[Chunk]:
    """The `limit` chunks richest in digits (by density), in document order."""

    def digits(c: Chunk) -> int:
        return sum(ch.isdigit() for ch in c.content)

    ranked = sorted(
        (c for c in chunks if digits(c) >= MIN_DIGITS),
        key=lambda c: -digits(c) / max(len(c.content), 1),
    )
    return sorted(ranked[:limit], key=lambda c: c.ordinal)


def _batch(kind: str, picked: list[Chunk], chunks: list[Chunk]) -> list[dict[str, str]]:
    sections = _joined_titles(section_titles(chunks), SECTIONS_CHARS) or "(sin secciones)"
    context = CONTEXT.format(sections=sections, excerpts="\n\n".join(map(_excerpt, picked)))
    return [
        {"role": "system", "content": ARTIFACT_SYSTEM},
        {"role": "user", "content": context + INSTRUCTIONS[kind].format()},
    ]


def _split(picked: list[Chunk]) -> list[list[Chunk]]:
    return [picked[i : i + PER_CALL] for i in range(0, len(picked), PER_CALL)][:MAX_CALLS]


def artifact_batches(
    kind: str, options: dict[str, Any], chunks: list[Chunk]
) -> list[list[dict[str, str]]]:
    if kind == "audit":
        picked = numeric_chunks(chunks, PER_CALL * MAX_CALLS)
        return [_batch(kind, group, chunks) for group in _split(picked)] or [
            _batch(kind, chunks[:PER_CALL], chunks)
        ]
    if kind == "entity_graph":
        picked = select_chunks(chunks, ARTIFACT_BUDGET * MAX_CALLS)
        return [_batch(kind, group, chunks) for group in _split(picked)]
    return [artifact_messages(kind, options, chunks)]
