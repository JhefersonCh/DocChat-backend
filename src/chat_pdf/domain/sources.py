"""Multi-source chats (plan §5): aliases, which sources a question reads, titles, the overview.

A chat's sources are the documents of its collection. Their aliases `S1..Sn` follow the order
they were added; the model and the citations use the alias, never the UUID. Pure functions."""

import re
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from chat_pdf.domain.models import Chunk, Document

MAX_SOURCES = 12
ALL = "all"
MAX_SUGGESTED = 6
_ALIAS = re.compile(r"^s(\d+)$", re.IGNORECASE)


def ordered(docs: Iterable[Document]) -> list[Document]:
    """Sources in the order they were added (the stable order the aliases follow)."""
    return sorted(docs, key=lambda d: (d.created_at, str(d.id)))


def alias_for(position: int) -> str:
    return f"S{position}"


def assign_aliases(docs: Iterable[Document]) -> dict[UUID, str]:
    return {d.id: alias_for(i) for i, d in enumerate(ordered(docs), start=1)}


def chat_title(filenames: Sequence[str]) -> str:
    """One source: its name. Several: "N fuentes". None: untitled."""
    if not filenames:
        return ""
    return filenames[0] if len(filenames) == 1 else f"{len(filenames)} fuentes"


def readable_sources(
    docs: Iterable[Document],
    disabled: Collection[UUID] = (),
    requested: Collection[UUID] | None = None,
) -> list[Document]:
    """Ready sources a question may read: `requested` (per-ask) wins over the chat's muted list."""
    ready = [d for d in ordered(docs) if d.status == "ready"]
    if requested:
        return [d for d in ready if d.id in requested]
    return [d for d in ready if d.id not in disabled]


def parse_source_filter(raw: Any, aliases: dict[str, UUID]) -> list[UUID] | None:
    """The model's `source` argument ("all", "S2", "S1, S3" or a list) -> document ids.

    None means every source; an empty list means the model named only unknown aliases."""
    if raw is None or raw == "":
        return None
    items = raw if isinstance(raw, list) else str(raw).split(",")
    names = [str(i).strip() for i in items if str(i).strip()]
    if not names or any(n.lower() == ALL for n in names):
        return None
    lookup = {a.lower(): d for a, d in aliases.items()}
    found = [lookup[n.lower()] for n in names if n.lower() in lookup]
    return list(dict.fromkeys(found))


def is_alias(text: str) -> bool:
    return _ALIAS.match(text.strip()) is not None


def balance_sources(ranked: list[Chunk], k: int, fallback: UUID | None = None) -> list[Chunk]:
    """Top `k` of a ranking over several sources, guaranteeing each source that has candidates
    a fair share (`k // sources`, at least 1) of its best ones: a dominant source cannot starve
    the others. The rest of the slots go by rank; the result keeps the ranking order."""
    groups: dict[UUID | None, list[Chunk]] = {}
    for chunk in ranked:
        groups.setdefault(chunk.document_id or fallback, []).append(chunk)
    if len(groups) <= 1:
        return ranked[:k]
    quota = max(1, k // len(groups))
    reserved = {c.id for chunks in groups.values() for c in chunks[:quota]}
    keep = [c for c in ranked if c.id in reserved][:k]
    taken = {c.id for c in keep}
    for chunk in ranked:
        if len(keep) >= k:
            break
        if chunk.id not in taken:
            taken.add(chunk.id)
            keep.append(chunk)
    order = {c.id: i for i, c in enumerate(ranked)}
    return sorted(keep, key=lambda c: order[c.id])


@dataclass(frozen=True)
class SourceSummary:
    alias: str
    document_id: UUID
    filename: str
    kind: str
    summary: str | None


@dataclass(frozen=True)
class CollectionOverview:
    sources: list[SourceSummary]
    summary: str | None
    suggested_questions: list[str]


CROSS_QUESTION = "¿Qué tienen en común y en qué se diferencian estas fuentes?"


def collection_overview(docs: Iterable[Document]) -> CollectionOverview | None:
    """Aggregate of the per-source overviews (no extra model call): None for fewer than 2 sources.

    The summary lists each source's own; the questions interleave each source's first ones and
    end with a cross-source one."""
    docs = ordered(docs)
    if len(docs) < 2:
        return None
    aliases = assign_aliases(docs)
    sources = [SourceSummary(aliases[d.id], d.id, d.filename, d.kind, d.summary) for d in docs]
    parts = [
        f"{s.alias} · {s.filename}: {' '.join(s.summary.split())}" for s in sources if s.summary
    ]
    questions: list[str] = []
    depth = max((len(d.suggested_questions) for d in docs), default=0)
    for level in range(depth):
        questions += [
            d.suggested_questions[level] for d in docs if level < len(d.suggested_questions)
        ]
    questions = list(dict.fromkeys(questions))[: MAX_SUGGESTED - 1] + [CROSS_QUESTION]
    return CollectionOverview(sources, "\n\n".join(parts) or None, questions)
