"""Semantic structure of a concept map: topics found by clustering the text-space vectors of the
chunks (spherical k-means, k by silhouette), representative chunks per topic, and the pairs of
concepts whose embeddings are close (synonyms to merge, cross-links to propose). Pure Python."""

import math
import random
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from chat_pdf.domain.models import Chunk

MIN_TOPICS, MAX_TOPICS = 3, 8
MIN_POINTS = 6  # fewer vectors than this: no meaningful clustering
FIT_POINTS = 600  # k-means runs on at most this many chunks; the rest are assigned afterwards
SILHOUETTE_POINTS = 120
ITERATIONS = 12
RESTARTS = 5
MIN_SHARE = 0.06  # a topic smaller than this share of the sample is noise, not a branch
# Calibrated on live EmbeddingGemma numbers: related-but-different labels ("Energía solar" vs
# "Energías renovables") reach 0.94, so only near-identical labels merge automatically.
SYNONYM_COSINE = 0.96
CROSS_COSINE = 0.78  # label+summary of concepts of different branches (top pairs sit at 0.78-0.82)
MAX_CROSS = 6


def unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else list(v)


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _mean(vectors: list[list[float]]) -> list[float]:
    size = len(vectors[0])
    total = [0.0] * size
    for v in vectors:
        for i, x in enumerate(v):
            total[i] += x
    return unit(total)


def _seeds(vectors: list[list[float]], k: int, rng: random.Random) -> list[list[float]]:
    """k-means++ seeding: each new seed is drawn with probability ~ squared distance."""
    chosen = [rng.randrange(len(vectors))]
    best = [max(0.0, 1 - dot(v, vectors[chosen[0]])) for v in vectors]
    while len(chosen) < k:
        weights = [d * d for d in best]
        total = sum(weights)
        if total <= 0:
            nxt = rng.randrange(len(vectors))
        else:
            r, acc, nxt = rng.random() * total, 0.0, len(vectors) - 1
            for i, w in enumerate(weights):
                acc += w
                if acc >= r:
                    nxt = i
                    break
        chosen.append(nxt)
        best = [
            min(b, max(0.0, 1 - dot(v, vectors[nxt]))) for b, v in zip(best, vectors, strict=True)
        ]
    return [vectors[i] for i in chosen]


def _kmeans_once(vectors, k, rng):
    centroids = _seeds(vectors, k, rng)
    assign = [-1] * len(vectors)
    for _ in range(ITERATIONS):
        new = [max(range(k), key=lambda c: dot(v, centroids[c])) for v in vectors]
        if new == assign:
            break
        assign = new
        for c in range(k):
            members = [v for v, a in zip(vectors, assign, strict=True) if a == c]
            if members:
                centroids[c] = _mean(members)
    cost = sum(1 - dot(v, centroids[a]) for v, a in zip(vectors, assign, strict=True))
    return cost, assign, centroids


def kmeans(vectors: list[list[float]], k: int) -> tuple[list[int], list[list[float]]]:
    """Spherical k-means over unit vectors (best of a few seeded restarts, deterministic):
    `(assignment per vector, centroids)`."""
    k = max(1, min(k, len(vectors)))
    rng = random.Random(7)
    _, assign, centroids = min(
        (_kmeans_once(vectors, k, rng) for _ in range(RESTARTS)), key=lambda r: r[0]
    )
    return assign, centroids


def silhouette(vectors: list[list[float]], assign: list[int], k: int) -> float:
    """Mean silhouette with cosine distance (1 - cos); -1 when it cannot be computed."""
    n = len(vectors)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist[i][j] = dist[j][i] = 1 - dot(vectors[i], vectors[j])
    scores = []
    for i in range(n):
        sums = [0.0] * k
        counts = [0] * k
        for j in range(n):
            if j != i:
                sums[assign[j]] += dist[i][j]
                counts[assign[j]] += 1
        if counts[assign[i]] == 0:
            continue
        a = sums[assign[i]] / counts[assign[i]]
        others = [sums[c] / counts[c] for c in range(k) if c != assign[i] and counts[c]]
        if not others:
            continue
        b = min(others)
        scores.append((b - a) / max(a, b, 1e-9))
    return sum(scores) / len(scores) if scores else -1.0


def choose_k(vectors: list[list[float]], low: int, high: int) -> int:
    """k in [low, high] with the best silhouette over an even sample of the vectors."""
    high = min(high, len(vectors) // 2)
    if high <= low:
        return max(1, min(low, len(vectors)))
    step = max(len(vectors) / SILHOUETTE_POINTS, 1)
    sample = [vectors[int(i * step)] for i in range(min(SILHOUETTE_POINTS, len(vectors)))]
    best_k, best = low, -2.0
    for k in range(low, high + 1):
        assign, _ = kmeans(sample, k)
        score = silhouette(sample, assign, k)
        if min(assign.count(c) for c in range(k)) < max(2, MIN_SHARE * len(sample)):
            score -= 1.0  # tiny clusters: prefer a coarser split
        if score > best:
            best_k, best = k, score
    return best_k


@dataclass
class Topic:
    index: int
    size: int
    members: set[UUID]  # ids of every chunk in the topic
    picks: list[
        Chunk
    ]  # representative chunks (nearest to the centroid, then diverse), in reading order


def plan_topics(
    items: list[tuple[Chunk, list[float]]], low: int, high: int, slots: int
) -> list[Topic]:
    """Clusters the (chunk, text vector) pairs into topics. Each gets `slots // k` (>= 2)
    representatives: the nearest to its centroid plus the chunk farthest from those picked."""
    if len(items) < MIN_POINTS:
        return []
    chunks = [c for c, _ in items]
    raw = [unit(v) for _, v in items]
    common = _mean(raw) if len(raw) > 1 else raw[0]
    # Embeddings of one subject share a large common component: centre them so topics separate.
    vectors = [unit([x - y * 0.9 for x, y in zip(v, common, strict=True)]) for v in raw]
    step = max(len(vectors) / FIT_POINTS, 1)
    fit_idx = sorted({int(i * step) for i in range(min(FIT_POINTS, len(vectors)))})
    fit = [vectors[i] for i in fit_idx]
    k = choose_k(fit, low, high)
    _, centroids = kmeans(fit, k)
    groups: list[list[int]] = [[] for _ in centroids]
    for i, v in enumerate(vectors):
        groups[max(range(len(centroids)), key=lambda c: dot(v, centroids[c]))].append(i)
    per = max(2, slots // max(sum(1 for g in groups if g), 1))
    topics: list[Topic] = []
    for members in groups:
        if not members:
            continue
        centre = _mean([vectors[i] for i in members])
        ranked = sorted(members, key=lambda i: -dot(vectors[i], centre))
        near = ranked[: max(per - 1, 1)]
        rest = [i for i in members if i not in near]
        picks = list(near)
        if rest and len(picks) < per:  # one diverse chunk: farthest from what is already picked
            picks.append(min(rest, key=lambda i: max(dot(vectors[i], vectors[p]) for p in picks)))
        picked = sorted((chunks[i] for i in picks), key=lambda c: c.ordinal)
        topics.append(
            Topic(
                len(topics) + 1,
                len(members),
                {chunks[i].id for i in members if chunks[i].id},
                picked,
            )
        )
    topics.sort(key=lambda t: min((c.ordinal for c in t.picks), default=0))
    for n, t in enumerate(topics, start=1):
        t.index = n
    return topics


def close_pairs(
    vectors: dict[str, list[float]], threshold: float, groups: dict[str, Any] | None = None
) -> list[tuple[float, str, str]]:
    """Pairs of keys with cosine >= threshold, best first. With `groups`, only pairs whose
    group differs (cross-branch)."""
    keys = list(vectors)
    unit_v = {k: unit(v) for k, v in vectors.items()}
    out = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if groups is not None and groups.get(a) == groups.get(b):
                continue
            s = dot(unit_v[a], unit_v[b])
            if s >= threshold:
                out.append((s, a, b))
    out.sort(reverse=True)
    return out


def similarity_report(vectors: dict[str, list[float]]) -> tuple[float, float, float]:
    """(max, mean, p90) of all pairwise cosines: the numbers logged to calibrate thresholds."""
    keys = list(vectors)
    unit_v = [unit(vectors[k]) for k in keys]
    sims = sorted(
        dot(unit_v[i], unit_v[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))
    )
    if not sims:
        return 0.0, 0.0, 0.0
    return sims[-1], sum(sims) / len(sims), sims[int(0.9 * (len(sims) - 1))]


PARAGRAPH_MIN, PARAGRAPH_MAX = 60, 350
SMALL_SOURCE = 30  # fewer chunks than this: cluster paragraphs instead (finer topics)


def paragraphs(text: str) -> list[str]:
    """Paragraph-sized pieces of a chunk: blank-line blocks, short headings glued to the next
    block, long blocks cut at sentence ends."""
    pieces: list[str] = []
    carry = ""
    for block in (b.strip() for b in text.split("\n\n")):
        if not block:
            continue
        block = f"{carry} {block}".strip()
        carry = ""
        if len(block) < PARAGRAPH_MIN:
            carry = block
            continue
        while len(block) > PARAGRAPH_MAX:
            cut = block.rfind(". ", PARAGRAPH_MAX // 2, PARAGRAPH_MAX)
            cut = cut + 1 if cut > 0 else PARAGRAPH_MAX
            pieces.append(block[:cut].strip())
            block = block[cut:].strip()
        if block:
            pieces.append(block)
    if carry and pieces:
        pieces[-1] = f"{pieces[-1]} {carry}"
    elif carry:
        pieces.append(carry)
    return pieces
