import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from chat_pdf.domain.calibration import calibration_table, chunk_margin
from chat_pdf.domain.models import Chunk, Source, SourceRef

# What the planner may ask `search_document` to favor (Spanish; the English synonyms are lenient).
MODALITY_HINTS = ("auto", "texto", "imagen", "audio", "video")
_HINT_ALIASES = {"text": "texto", "image": "imagen", "foto": "imagen", "frame": "video"}


def parse_hint(raw: Any) -> str:
    """Normalize the model's `modality` argument; anything unknown means "auto"."""
    text = unicodedata.normalize("NFKD", str(raw or "auto").strip().lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _HINT_ALIASES.get(text, text)
    return text if text in MODALITY_HINTS else "auto"


@dataclass(frozen=True)
class RankingPlan:
    """How one search weighs its four rankings and which segment modalities it looks at."""

    weights: tuple[float, float, float, float]  # text vector, full text, image, audio
    modalities: tuple[str, ...] | None  # None: every modality


def plan_rankings(
    hint: str, text_weight: float, native_weight: float, boosted_weight: float
) -> RankingPlan:
    """Weights of R1..R4 (plan §3.1). Native rankings are rank-fused, never compared by cosine."""
    base = (1.0, text_weight)
    if hint == "texto":
        return RankingPlan((*base, 0.0, 0.0), ("text",))
    if hint == "imagen":
        return RankingPlan((*base, boosted_weight, 0.0), ("image", "frame"))
    if hint == "audio":
        return RankingPlan((*base, 0.0, boosted_weight), ("audio",))
    if hint == "video":
        return RankingPlan((*base, boosted_weight, boosted_weight), ("frame", "audio"))
    return RankingPlan((*base, native_weight, native_weight), None)


class SourceSet:
    """Sources of one answer: deduplicated by chunk, with stable `[1..n]` numbering.

    `refs` are the chat's sources by document id: chunks found by a search carry their
    `document_id`, and `labeled` (several sources in the chat) makes the prompt name each one."""

    def __init__(self, refs: Mapping[UUID, SourceRef] | None = None, labeled: bool = False) -> None:
        self._by_chunk: dict[UUID | None, Source] = {}
        self._refs = refs or {}
        self._labeled = labeled

    @property
    def sources(self) -> list[Source]:
        return list(self._by_chunk.values())

    def add(
        self,
        chunks: list[Chunk],
        document_id: UUID | None = None,
        filename: str = "",
        source_kind: str | None = None,
    ) -> list[Source]:
        """Register `chunks`; returns their sources (a chunk seen before keeps its number).

        A chunk belongs to `document_id` when given, else to the source its own `document_id`
        names (a single known source is the fallback for chunks that carry none)."""
        found = []
        for chunk in chunks:
            if chunk.id not in self._by_chunk:
                n = len(self._by_chunk) + 1
                self._by_chunk[chunk.id] = self._source(
                    n, chunk, document_id, filename, source_kind
                )
            found.append(self._by_chunk[chunk.id])
        return found

    def _source(
        self, n: int, chunk: Chunk, document_id: UUID | None, filename: str, kind: str | None
    ) -> Source:
        owner = document_id or chunk.document_id
        if owner is None and len(self._refs) == 1:
            owner = next(iter(self._refs))
        ref = self._refs.get(owner) if owner else None
        if ref is not None:
            return Source(
                n, chunk, ref.document_id, ref.filename, ref.kind, ref.alias, self._labeled
            )
        return Source(n, chunk, owner, filename, kind)  # type: ignore[arg-type]


def fit_sources(sources: list[Source], max_chars: int) -> list[Source]:
    """Keep whole sources in `[n]` order while they fit the budget (always at least the first).

    The kept sources are a prefix, so their numbers stay `1..m`: prompt, citations and `cited`
    all use the same numbering.
    """
    kept: list[Source] = []
    used = 0
    for source in sources:
        used += len(source.chunk.content)
        if kept and used > max_chars:
            break
        kept.append(source)
    return kept


def clamp_k(requested: int | None, default_k: int, max_k: int) -> int:
    """The model proposes k; the backend imposes `1 <= k <= max_k` (no k -> default)."""
    k = default_k if requested is None else requested
    return max(1, min(k, max_k))


def reciprocal_rank_fusion(
    rankings: list[list[Chunk]], k: int, weights: list[float] | None = None
) -> list[Chunk]:
    """Merge rankings by (weighted) RRF: score = sum of weight / (k + rank) over the rankings.

    Rank-based, so vector distances and full-text ranks need no common scale. Ties keep the order
    of first appearance (earlier rankings first).
    """
    scores: dict[UUID | None, float] = {}
    chunks: dict[UUID | None, Chunk] = {}
    for ranking, weight in zip(rankings, weights or [1.0] * len(rankings), strict=True):
        for rank, chunk in enumerate(ranking, start=1):
            chunks.setdefault(chunk.id, chunk)
            scores[chunk.id] = scores.get(chunk.id, 0.0) + weight / (k + rank)
    order = sorted(chunks, key=lambda key: -scores[key])  # stable: first appearance on ties
    return [chunks[key] for key in order]


def is_weak_evidence(
    sources: list[Source], threshold: float, calibration: Mapping[str, Any] | None = None
) -> bool:
    """True when no source clears the threshold of its own modality (plan §3.2).

    `threshold` is the text one; `calibration` (a profile's) overrides per modality. A source whose
    similarity is unknown (panorama chunks) means the evidence cannot be judged: never weak.
    """
    table = calibration_table(calibration, threshold)
    margins = [chunk_margin(s.chunk, table) for s in sources]
    if None in margins:
        return False
    return max(margins, default=-1.0) < 0  # type: ignore[type-var]
