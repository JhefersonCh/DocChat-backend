"""Global questions: detect them, describe the document (index) and spread chunks over it."""

import re
import unicodedata
from collections.abc import Collection
from uuid import UUID

from chat_pdf.domain.generation import _spread, section_titles
from chat_pdf.domain.models import Chunk, Document, Source
from chat_pdf.domain.prompts import (
    INDEX_HEADER,
    INDEX_NO_SECTIONS,
    INDEX_SECTIONS,
    INDEX_SOURCE,
    INDEX_SUMMARY,
)

MAX_INDEX_TITLES = 60
MAX_INDEX_CHARS = 3000
MAX_INDEX_SUMMARY_CHARS = 1500
MAX_PANORAMA_CHUNKS = 30
MIN_COVERED_SECTIONS = 3  # a global question needs at least this many sections in its sources

# Matched on accent-stripped, lowercase text.
_GLOBAL = re.compile(
    r"\b(?:"
    r"resum\w*|sintesis|sinopsis|panorama|"
    r"de que (?:se )?(?:trata|habla|va)\w*|"
    r"en general|a grandes rasgos|"
    r"(?:idea|ideas|tema|temas|punto|puntos|vision|descripcion)(?: \w+)? "
    r"(?:general|generales|principal|principales|clave|claves)|"
    r"(?:principal|principales|general|generales)\w* (?:tema|temas|idea|ideas|punto|puntos)|"
    r"conclusion\w*"
    r")\b"
)


def _plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def is_global_question(question: str) -> bool:
    """Light heuristic: does the question ask about the document as a whole?"""
    return _GLOBAL.search(_plain(question)) is not None


def document_index(summary: str | None, chunks: list[Chunk]) -> str:
    """Summary (if any) + ordered distinct section titles, for the planner's system message.

    One distinct title or none means the document has no real sections (a single title repeated
    on every chunk is the document title, not a structure to search by).
    """
    lines = [INDEX_HEADER]
    if summary and summary.strip():
        lines.append(
            INDEX_SUMMARY.format(summary=" ".join(summary.split())[:MAX_INDEX_SUMMARY_CHARS])
        )
    titles = section_titles(chunks)
    if len(titles) <= 1:
        lines.append(INDEX_NO_SECTIONS)
        return "\n".join(lines)
    n = min(len(titles), MAX_INDEX_TITLES)
    while n > 1 and len("; ".join(_spread(titles, n))) > MAX_INDEX_CHARS:
        n -= 1
    shown = _spread(titles, n)
    lines.append(INDEX_SECTIONS.format(count=len(titles), titles="; ".join(shown)))
    return "\n".join(lines)


def collection_index(entries: list[tuple[str, Document, list[Chunk]]]) -> str:
    """The planner's index of the readable sources. One source: `document_index` as always.
    Several: each source's own index under its `[S1 · name]` heading."""
    if len(entries) == 1:
        _, doc, chunks = entries[0]
        return document_index(doc.summary, chunks)
    blocks = []
    for alias, doc, chunks in entries:
        index = document_index(doc.summary, chunks).split("\n", 1)[-1]  # one shared header
        blocks.append(f"{INDEX_SOURCE.format(alias=alias, name=doc.filename)}\n{index}")
    return "\n".join([INDEX_HEADER, *blocks])


def covered_sections(sources: list[Source]) -> set[str]:
    return {s.chunk.section for s in sources if s.chunk.section}


def has_low_coverage(sources: list[Source], searches: int) -> bool:
    return searches <= 1 or len(covered_sections(sources)) < MIN_COVERED_SECTIONS


def panorama_chunks(chunks: list[Chunk], sources: list[Source], budget_chars: int) -> list[Chunk]:
    """Chunks that spread over the document, in reading order, not already in `sources`.

    One per section not yet covered (its first chunk); evenly spaced when the document has no
    sections. Takes as many as fit in `budget_chars` minus what `sources` already uses.
    """
    seen: Collection[UUID | None] = {s.chunk.id for s in sources}
    remaining = budget_chars - sum(len(s.chunk.content) for s in sources)
    fresh = [
        c for c in chunks if c.id not in seen and c.content.strip()
    ]  # media may be undescribed
    if len(section_titles(chunks)) > 1:
        done = covered_sections(sources)
        firsts: dict[str, Chunk] = {}
        for c in fresh:
            if c.section and c.section not in done:
                firsts.setdefault(c.section, c)
        candidates = list(firsts.values())
    else:
        candidates = fresh
    n = min(len(candidates), MAX_PANORAMA_CHUNKS)
    while n > 0 and sum(len(c.content) for c in _spread(candidates, n)) > remaining:
        n -= 1
    return _spread(candidates, n) if n else []
