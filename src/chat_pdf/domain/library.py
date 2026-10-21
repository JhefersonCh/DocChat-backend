"""Library rules (pure): clustering documents by their vectors, naming prompt, grouping hits.

Auto-categorization = k-means over unit-normalized document vectors (cosine geometry), with `k`
chosen by silhouette over 2..MAX_K. Pure Python on purpose (no numpy dependency): vectors are
first reduced by a seeded random projection when they are wide, so tens or hundreds of documents
cluster in well under a second. Deterministic: same input, same clusters."""

import math
import random
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from chat_pdf.domain.generation import extract_json

MAX_K = 8
MIN_DOCS_TO_CLUSTER = 3
PROJECTED_DIMS = 48
MAX_NAME_CHARS = 40
NAMING_SUMMARY_CHARS = 280
SEED = 7

Vector = list[float]


def normalize(v: Vector) -> Vector:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else list(v)


def cosine(a: Vector, b: Vector) -> float:
    """Cosine of two vectors of the same length (0.0 when either is the zero vector)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def project(vectors: list[Vector], dims: int = PROJECTED_DIMS, seed: int = SEED) -> list[Vector]:
    """Seeded Gaussian random projection (Johnson-Lindenstrauss) of unit vectors to `dims`."""
    width = len(vectors[0]) if vectors else 0
    if width <= dims:
        return vectors
    rng = random.Random(seed)
    matrix = [[rng.gauss(0, 1 / math.sqrt(dims)) for _ in range(dims)] for _ in range(width)]
    out = []
    for v in vectors:
        row = [0.0] * dims
        for x, weights in zip(v, matrix, strict=True):
            if x:
                for j in range(dims):
                    row[j] += x * weights[j]
        out.append(normalize(row))
    return out


def _dist(a: Vector, b: Vector) -> float:
    return 1.0 - sum(x * y for x, y in zip(a, b, strict=True))  # unit vectors: cosine distance


def kmeans(points: list[Vector], k: int, seed: int = SEED, iterations: int = 30) -> list[int]:
    """Cluster index of each unit vector. Farthest-point init (first center seeded), Lloyd steps."""
    rng = random.Random(seed)
    centers = [points[rng.randrange(len(points))]]
    while len(centers) < k:
        far = max(points, key=lambda p: min(_dist(p, c) for c in centers))
        centers.append(far)
    labels = [-1] * len(points)
    for _ in range(iterations):
        new = [min(range(k), key=lambda j: _dist(p, centers[j])) for p in points]
        if new == labels:
            break
        labels = new
        for j in range(k):
            members = [p for p, label in zip(points, labels, strict=True) if label == j]
            if members:
                centers[j] = normalize([sum(col) for col in zip(*members, strict=True)])
    return labels


def silhouette(points: list[Vector], labels: list[int]) -> float:
    """Mean silhouette in [-1, 1] (higher = better separated); 0 if fewer than 2 clusters."""
    groups: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(i)
    if len(groups) < 2:
        return 0.0
    scores = []
    for i, p in enumerate(points):
        own = groups[labels[i]]
        if len(own) == 1:
            scores.append(0.0)
            continue
        a = sum(_dist(p, points[j]) for j in own if j != i) / (len(own) - 1)
        b = min(
            sum(_dist(p, points[j]) for j in members) / len(members)
            for label, members in groups.items()
            if label != labels[i]
        )
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return sum(scores) / len(scores)


def cluster_documents(vectors: dict[UUID, Vector], max_k: int = MAX_K) -> list[list[UUID]]:
    """Groups of document ids (largest first). `k` = the one with the best silhouette in
    2..min(max_k, n // 2 or so); fewer than MIN_DOCS_TO_CLUSTER documents -> no clusters."""
    ids = sorted(vectors, key=str)
    if len(ids) < MIN_DOCS_TO_CLUSTER:
        return []
    points = project([normalize(vectors[i]) for i in ids])
    best_labels, best_score = None, -2.0
    for k in range(2, min(max_k, len(ids) - 1) + 1):
        labels = kmeans(points, k)
        if len(set(labels)) < 2:
            continue
        score = silhouette(points, labels)
        if score > best_score:
            best_labels, best_score = labels, score
    if best_labels is None:
        return []
    groups: dict[int, list[UUID]] = {}
    for doc_id, label in zip(ids, best_labels, strict=True):
        groups.setdefault(label, []).append(doc_id)
    return sorted(groups.values(), key=lambda g: -len(g))


NAMING_SYSTEM = (
    "Sos un bibliotecario. Recibís grupos de documentos (nombre y resumen). Poné a CADA grupo un "
    "nombre de categoría corto (1 a 3 palabras, en español, temático, sin comillas ni punto "
    "final). "
    'Respondé SOLO un JSON: [{"grupo": 1, "nombre": "..."}, ...] con todos los grupos.'
)


@dataclass(frozen=True)
class NamingDoc:
    filename: str
    summary: str | None


def _brief(summary: str | None) -> str:
    return " ".join((summary or "sin resumen").split())[:NAMING_SUMMARY_CHARS]


def naming_messages(groups: list[list[NamingDoc]]) -> list[dict[str, Any]]:
    blocks = []
    for n, docs in enumerate(groups, start=1):
        lines = [f"- {d.filename}: {_brief(d.summary)}" for d in docs[:6]]
        blocks.append(f"Grupo {n}:\n" + "\n".join(lines))
    return [
        {"role": "system", "content": NAMING_SYSTEM},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]


def parse_names(raw: str | None, count: int) -> list[str]:
    """Category names for `count` groups from the model's JSON; a missing / invalid one is
    "Grupo N". Names are trimmed, capped and made distinct."""
    data = extract_json(raw or "")
    named: dict[int, str] = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("nombre"), str):
                try:
                    group = int(item.get("grupo"))
                except (TypeError, ValueError):
                    continue
                name = " ".join(item["nombre"].split())[:MAX_NAME_CHARS].strip(" .\"'")
                named.setdefault(group, name)  # the first answer for a group wins
    names: list[str] = []
    for n in range(1, count + 1):
        base = named.get(n) or f"Grupo {n}"
        name, i = base, 2
        while name.lower() in {x.lower() for x in names}:
            name, i = f"{base} {i}", i + 1
        names.append(name)
    return names


@dataclass(frozen=True)
class DocHit:
    """Best passages of one document for a query (a group of chunk hits)."""

    document_id: UUID
    score: float
    doc_score: float | None  # cosine of the query to the document vector (None: no vector)
    chunk_score: float | None  # best chunk relevance in [0, 1]


def blend(doc_score: float | None, chunk_score: float | None, doc_weight: float = 0.4) -> float:
    """Document ranking score: the best chunk counts most, the document vector breaks ties and
    rescues documents whose passages did not surface. Missing parts do not penalize."""
    if doc_score is None and chunk_score is None:
        return 0.0
    if doc_score is None:
        return chunk_score or 0.0
    if chunk_score is None:
        return doc_score * doc_weight
    return (1 - doc_weight) * chunk_score + doc_weight * doc_score


# --- Read models --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Project:
    id: UUID
    name: str
    created_at: Any
    description: str | None = None
    color: str | None = None
    icon: str | None = None
    source_count: int = 0
    ready_count: int = 0
    chat_count: int = 0
    pinned_at: Any = None


@dataclass(frozen=True)
class Category:
    id: UUID
    name: str
    origin: str  # user | auto
    created_at: Any
    color: str | None = None
    doc_count: int = 0


@dataclass(frozen=True)
class CategoryRef:
    id: UUID
    name: str
    origin: str  # of the assignment
    color: str | None = None


@dataclass(frozen=True)
class LibraryDoc:
    """A document in the library with where it lives (its chat or project) and its categories."""

    id: UUID
    filename: str
    kind: str
    status: str
    mime: str
    created_at: Any
    collection_id: UUID
    embedding_profile_id: UUID
    page_count: int | None = None
    duration_s: float | None = None
    summary: str | None = None
    project_id: UUID | None = None
    project_name: str | None = None
    chat_id: UUID | None = None  # first chat of the collection (the chat itself for a loose one)
    chat_title: str | None = None
    has_vector: bool = False
    categories: list[CategoryRef] = field(default_factory=list)
