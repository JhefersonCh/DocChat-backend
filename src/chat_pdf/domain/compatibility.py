"""Source compatibility: do the sources of a chat / project talk about related things?

Pure functions over sampled text-space chunk vectors (text chunks, captions, transcripts; never a
native image / audio vector: those live in another region of the space). Only vectors of the same
embedding profile are compared; anything else is "unknown".

PAIR SCORE. The mean of a document's chunk vectors compresses differences (unrelated documents
scored 0.55-0.88 with it, a business plan vs a favicon 0.87), so two documents are compared chunk
by chunk: for each chunk of A its best match in B; the mean of the top TOP_SHARE of those, averaged
with the same from B to A (symmetric). A shared subtopic counts, filler does not.

CALIBRATION (remote profile, 768 d; pair score; real documents + throwaway texts):
  related (plan de negocios / estudio financiero / plan de cafe)     0.76 - 0.84
  related (lab report TCP/UDP vs Wireshark practice)                 0.89
  unrelated (business vs networking vs novel vs recipe vs favicon)   0.55 - 0.68
  same file uploaded twice (images, captions regenerated)            0.93 - 0.97
  RELATED_FROM 0.72 / UNRELATED_BELOW 0.70: both sit in the empty 0.68-0.76 gap.

GROUPS. A graph links the sources whose pair score reaches RELATED_FROM; the connected components
say who is with whom. Several components -> every source outside the largest one (a tie: all of
them) is flagged. Copies of one source (same file hash or score >= DUPLICATE_FROM) are one node,
never related to each other, flagged "duplicate". Thresholds are per embedding model."""

import math
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from chat_pdf.domain.library import Vector

RELATED_FROM = 0.72
UNRELATED_BELOW = 0.70
DUPLICATE_FROM = 0.95
TOP_SHARE = 0.3
MIN_TEXT = 2  # text chunks that make a document be judged by its text, not by picture captions

PENDING = "Todavía sin texto ni descripción: no se puede comparar (pendiente de descripción)"
OTHER_PROFILE = "Otro perfil de embeddings: no es comparable con las demás"

Status = Literal["ok", "weak", "unrelated", "duplicate", "unknown"]


@dataclass(frozen=True)
class CompatItem:
    """A source to judge. `vectors` is empty while it is not indexed (no text-space vector)."""

    document_id: UUID
    alias: str
    filename: str
    profile_id: UUID
    vectors: tuple[Vector, ...] = ()
    chunks: int | None = None  # chunk count behind the vectors: part of the cache key
    summary: str | None = None
    sha: str | None = None  # file hash: the same file twice is a duplicate


@dataclass(frozen=True)
class BestMatch:
    document_id: UUID
    alias: str
    similarity: float


@dataclass(frozen=True)
class SourceCompat:
    document_id: UUID
    alias: str
    filename: str
    status: Status
    best_match: BestMatch | None = None
    reason: str | None = None


@dataclass(frozen=True)
class Compatibility:
    coherent: bool
    sources: list[SourceCompat] = field(default_factory=list)


def topic_vectors(rows: list[tuple[str, Vector]]) -> tuple[Vector, ...]:
    """The vectors that say what a document is about: its text chunks when it has some, else the
    captions / transcripts of its media (a picture's own regions are its whole content)."""
    texts = [v for m, v in rows if m == "text"]
    return tuple(texts if len(texts) >= MIN_TEXT else [v for _, v in rows])


def _unit(v: Vector) -> Vector:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _dot(u: Vector, w: Vector) -> float:
    return sum(x * y for x, y in zip(u, w, strict=True))


def _side(a: list[Vector], b: list[Vector]) -> float:
    best = sorted((max(_dot(u, w) for w in b) for u in a), reverse=True)
    k = max(1, math.ceil(len(best) * TOP_SHARE))
    return sum(best[:k]) / k


def similarity(a: tuple[Vector, ...], b: tuple[Vector, ...]) -> float:
    """Symmetric top-share chunk match score of two documents (see module doc)."""
    ua, ub = [_unit(v) for v in a], [_unit(v) for v in b]
    return (_side(ua, ub) + _side(ub, ua)) / 2


def classify(score: float) -> Status:
    if score >= RELATED_FROM:
        return "ok"
    return "unrelated" if score < UNRELATED_BELOW else "weak"


def analyze(items: list[CompatItem]) -> Compatibility:
    """Per-source status; `coherent` is false when the sources do not form one related group."""
    if len(items) < 2:
        return Compatibility(
            True, [SourceCompat(i.document_id, i.alias, i.filename, "ok") for i in items]
        )
    by_profile: dict[UUID, list[CompatItem]] = {}
    for i in items:
        if i.vectors:
            by_profile.setdefault(i.profile_id, []).append(i)
    verdicts: dict[UUID, SourceCompat] = {}
    outside = False
    for item in items:
        group = by_profile.get(item.profile_id, [])
        if not item.vectors:
            verdicts[item.document_id] = _unknown(item, PENDING)
        elif len(group) < 2 or len(group[0].vectors[0]) != len(item.vectors[0]):
            verdicts[item.document_id] = _unknown(item, OTHER_PROFILE)
    for group in by_profile.values():
        if len(group) < 2:
            continue
        judged, has_outliers = _judge_group(group)
        verdicts.update(judged)
        outside = outside or has_outliers
    out = [verdicts[i.document_id] for i in items]
    return Compatibility(not outside, out)


def _judge_group(group: list[CompatItem]) -> tuple[dict[UUID, SourceCompat], bool]:
    n = len(group)
    sim = [[0.0] * n for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            sim[a][b] = sim[b][a] = similarity(group[a].vectors, group[b].vectors)
    dup = [[False] * n for _ in range(n)]
    nodes = list(range(n))  # union-find over copies

    def find(x: int) -> int:
        while nodes[x] != x:
            nodes[x] = nodes[nodes[x]]
            x = nodes[x]
        return x

    for a in range(n):
        for b in range(a + 1, n):
            same_file = bool(group[a].sha) and group[a].sha == group[b].sha
            if same_file or sim[a][b] >= DUPLICATE_FROM:
                dup[a][b] = dup[b][a] = True
                nodes[find(a)] = find(b)
    # components of the related graph over the nodes (a copy and its twin are one node)
    comp = {r: r for r in {find(x) for x in range(n)}}

    def croot(r: int) -> int:
        while comp[r] != r:
            r = comp[r]
        return r

    for a in range(n):
        for b in range(a + 1, n):
            if not dup[a][b] and find(a) != find(b) and sim[a][b] >= RELATED_FROM:
                comp[croot(find(a))] = croot(find(b))
    size: dict[int, int] = {}
    for r in {find(x) for x in range(n)}:
        size[croot(r)] = size.get(croot(r), 0) + 1
    top = max(size.values())
    main = [c for c, s in size.items() if s == top]
    one_node = len(size) == 1  # only copies of one source: nothing to be unrelated to
    out: dict[UUID, SourceCompat] = {}
    outliers = 0
    for a, item in enumerate(group):
        twins = [b for b in range(n) if b != a and dup[a][b]]
        others = [b for b in range(n) if b != a and not dup[a][b]]
        in_main = len(main) == 1 and top > 1 and croot(find(a)) == main[0]
        flagged_out = not one_node and not in_main
        outliers += flagged_out
        if twins:
            t = max(twins, key=lambda b: sim[a][b])
            reason = f"Parece la misma fuente que {group[t].alias}"
            if flagged_out:
                reason += " y no se relaciona con las demás"
            best = BestMatch(group[t].document_id, group[t].alias, round(sim[a][t], 3))
            out[item.document_id] = SourceCompat(
                item.document_id, item.alias, item.filename, "duplicate", best, reason
            )
            continue
        b = max(others, key=lambda x: sim[a][x])
        best = BestMatch(group[b].document_id, group[b].alias, round(sim[a][b], 3))
        if not flagged_out:
            status: Status = "ok"
        elif sim[a][b] >= RELATED_FROM:
            status = "weak"  # related to a smaller group only
        else:
            status = classify(sim[a][b])
        out[item.document_id] = SourceCompat(
            item.document_id, item.alias, item.filename, status, best
        )
    return out, outliers > 0


def _unknown(item: CompatItem, reason: str) -> SourceCompat:
    return SourceCompat(item.document_id, item.alias, item.filename, "unknown", None, reason)


def flagged(result: Compatibility) -> list[SourceCompat]:
    """Sources worth a one-line reason from the model (a duplicate has its own static one)."""
    return [s for s in result.sources if s.status in ("weak", "unrelated")]


def cache_key(items: list[CompatItem]) -> frozenset:
    """The set of documents and their chunk counts: changes when a source is added, removed or
    re-indexed."""
    return frozenset((i.document_id, i.chunks) for i in items)


REASON_SYSTEM = (
    "Recibís las fuentes de una conversación con un resumen de cada una y cuáles parecen poco "
    "relacionadas con las demás. Respondé UNA sola frase corta en español rioplatense (máximo 25 "
    "palabras), sin comillas ni markdown, que diga de qué trata cada fuente marcada y de qué "
    'tratan las demás. Ejemplo: "S3 es una receta de cocina; las demás son planes de negocio."'
)


def reason_messages(items: list[CompatItem], marked: list[SourceCompat]) -> list[dict]:
    lines = [
        f"{i.alias} ({i.filename}): {' '.join((i.summary or 'sin resumen').split())[:280]}"
        for i in items
    ]
    flags = ", ".join(s.alias for s in marked)
    return [
        {"role": "system", "content": REASON_SYSTEM},
        {"role": "user", "content": "\n".join(lines) + f"\n\nPoco relacionadas: {flags}"},
    ]


def clean_reason(raw: str | None) -> str | None:
    text = " ".join((raw or "").split()).strip(" \"'")
    return text[:240] or None
