"""Exhaustive search ("¿dónde hay un perro?", plan §3.3): which segments are worth checking,
how a vision model's yes/no is read, and how confirmed regions are ordered and collapsed.

Pure rules; the use case (application/find_all.py) does the I/O."""

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from chat_pdf.domain.calibration import CalibrationTable, chunk_margin, space_of
from chat_pdf.domain.models import Chunk, Source

RECALL_SLACK = 0.04  # candidates are taken this far under the native threshold (recall first)
CLEAR_MARGIN = 0.08  # native cosine this far over the threshold AND a text match: no need to ask
MAX_OCCURRENCES = 40  # most occurrences returned (the rest is only counted)
CONTAINMENT_TOLERANCE = 0.02  # normalized slack when testing "region A holds region B"
NESTED_AREA_RATIO = 1.2  # the holder must be clearly bigger than the held region
VISUAL = ("image", "frame")

VERIFY_PROMPT = (
    "¿La imagen muestra lo siguiente: «{query}»? "
    "Empezá tu respuesta con «sí» o «no». Si es sí, indicá en pocas palabras dónde se ve."
)

_WORD = re.compile(r"[a-z0-9]+")
_VERDICT = re.compile(r"\b(si|yes|no)\b")


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize_query(query: str) -> str:
    """Cache key of a query: accents, case and punctuation do not matter ("¿Un Perro?" and
    "un perro" share a verdict); the words themselves do."""
    return " ".join(_WORD.findall(_fold(query)))


def verify_prompt(query: str) -> str:
    return VERIFY_PROMPT.format(query=" ".join(query.split()))


def parse_verdict(answer: str) -> bool | None:
    """True / False for a leading sí / no; None when the model did not commit."""
    found = _VERDICT.search(_fold(answer))
    return None if found is None else found.group(1) != "no"


@dataclass(frozen=True)
class MediaMatch:
    """A media segment with its cosines to the query and whether its description matched the
    query words in the full-text index."""

    chunk: Chunk
    fts: bool = False


def is_candidate(match: MediaMatch, table: CalibrationTable) -> bool:
    """Recall rule: native cosine near its threshold, or description close in text space, or a
    keyword hit in the description."""
    chunk = match.chunk
    if chunk.native_similarity is not None:
        if chunk.native_similarity >= table[space_of(chunk.modality)].threshold - RECALL_SLACK:
            return True
    if chunk.similarity is not None and chunk.similarity >= table["text"].threshold:
        return True
    return match.fts


def is_clear(match: MediaMatch, table: CalibrationTable) -> bool:
    """Strong native evidence plus a keyword hit in the description: accepted unasked."""
    chunk = match.chunk
    if not match.fts or chunk.native_similarity is None:
        return False
    threshold = table[space_of(chunk.modality)].threshold
    return chunk.native_similarity - threshold >= CLEAR_MARGIN


def strength(match: MediaMatch, table: CalibrationTable) -> float:
    """Ordering of candidates (strongest checked first when the budget is short)."""
    margin = chunk_margin(match.chunk, table)
    return (margin if margin is not None else -1.0) + (0.05 if match.fts else 0.0)


def confidence(match: MediaMatch, table: CalibrationTable) -> str:
    """Label of an unverified occurrence."""
    return "alta" if is_clear(match, table) else "media"


def reading_order(chunks: Iterable[Chunk]) -> list[Chunk]:
    """Reading order inside each source; the sources keep the order they first appear in."""
    chunks = list(chunks)
    first_seen: dict[object, int] = {}
    for c in chunks:
        first_seen.setdefault(c.document_id, len(first_seen))

    def key(c: Chunk):
        y0, x0 = (c.bbox[1], c.bbox[0]) if c.bbox else (0.0, 0.0)
        return (first_seen[c.document_id], c.page_start or 0, c.t_start or 0.0, y0, x0, c.ordinal)

    return sorted(chunks, key=key)


def _holds(outer: list[float], inner: list[float]) -> bool:
    eps = CONTAINMENT_TOLERANCE
    inside = (
        outer[0] <= inner[0] + eps and outer[1] <= inner[1] + eps
        and outer[2] >= inner[2] - eps and outer[3] >= inner[3] - eps
    )  # fmt: skip

    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    return inside and area(outer) > area(inner) * NESTED_AREA_RATIO


def collapse_regions(chunks: list[Chunk]) -> list[Chunk]:
    """Drop a confirmed region when a smaller confirmed one on the same page sits inside it
    (the scanned page and the tile of it that holds the dog: keep the tile)."""
    kept = []
    for chunk in chunks:
        holds_smaller = chunk.bbox is not None and any(
            other is not chunk
            and other.bbox is not None
            and other.document_id == chunk.document_id
            and other.page_start == chunk.page_start
            and _holds(chunk.bbox, other.bbox)
            for other in chunks
        )
        if not holds_smaller:
            kept.append(chunk)
    return kept


@dataclass(frozen=True)
class FindAllResult:
    """Outcome of an exhaustive search, ready to be shown to the model."""

    sources: list[Source]  # the occurrences, in reading order, numbered [n]
    confidence: dict[int, str]  # n -> "alta" | "media" for occurrences no vision model checked
    checked: int = 0  # occurrences confirmed by the vision model
    discarded: int = 0  # candidates the vision model rejected
    extra: int = 0  # occurrences beyond MAX_OCCURRENCES that are not listed
