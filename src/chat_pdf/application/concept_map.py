"""Concept maps (Estudio): generation over all the active sources (<= 3 LLM calls) and the
on-demand expansion of one node (1 call) that grows the stored map, grounded in the sources."""

import asyncio
import logging
from dataclasses import replace
from typing import Any
from uuid import UUID

from chat_pdf.application.chats import require_chat
from chat_pdf.application.generation import INVALID_RESPONSE, _require_chat_profile
from chat_pdf.application.ports import (
    ChatRepository,
    ChunkStore,
    DocumentRepository,
    ProviderFactory,
    ProviderRepository,
)
from chat_pdf.application.sources import readable_chunks, readable_documents
from chat_pdf.domain.catalog import _line
from chat_pdf.domain.concept_cluster import (
    CROSS_COSINE,
    MAX_CROSS,
    MAX_TOPICS,
    MIN_TOPICS,
    SMALL_SOURCE,
    SYNONYM_COSINE,
    Topic,
    close_pairs,
    paragraphs,
    plan_topics,
    similarity_report,
)
from chat_pdf.domain.concept_map import (
    CALL_BUDGET,
    MAX_CALLS,
    MAX_TOTAL_NODES,
    expand_messages,
    extract_batches,
    merge_expansion,
    merge_messages,
    node_by_id,
    parse_extraction,
    parse_map,
    parse_merge,
)
from chat_pdf.domain.concept_topics import (
    assign_topics,
    build_topic_map,
    cross_props,
    merge_messages_topics,
    topic_batches,
    topic_of_chunks,
)
from chat_pdf.domain.config import RagConfig
from chat_pdf.domain.errors import Conflict, InvalidInput, NotFound, ProviderError
from chat_pdf.domain.models import Chunk, Document
from chat_pdf.domain.retrieval import reciprocal_rank_fusion
from chat_pdf.domain.slides import EXCERPT
from chat_pdf.domain.sources import assign_aliases

log = logging.getLogger(__name__)
if not log.handlers:  # uvicorn only configures its own loggers: show where the embeddings are used
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False

MAP_KIND = "concept_map"
EXPAND_CHUNKS = 12
EXPAND_OWN = 2
MAX_UNITS = 150


class Semantics:
    """The embedding model at work for concept maps: stored chunk vectors (clustering), one
    batched embedding of concept labels, and query embedding + hybrid retrieval (expansion), all
    through the embedding profile the sources were indexed with."""

    def __init__(self, store, providers: ProviderRepository, factory: ProviderFactory, rag=None):
        self._store = store
        self._providers = providers
        self._factory = factory
        self._rag = rag or RagConfig()
        self.profile = None
        self._embedder = None
        self.ids: list[UUID] = []

    async def bind(self, sources: list[Document]) -> bool:
        if not sources:
            return False
        profile = await self._providers.get(sources[0].embedding_profile_id)
        if profile is None:
            return False
        self.profile = profile
        self._embedder = self._factory.embedder(profile)
        self.ids = [d.id for d in sources if d.embedding_profile_id == profile.id]
        return True

    async def vectors(self) -> dict[UUID, list[float]]:
        return await self._store.text_vectors(self.ids)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefix = (self.profile.document_prefix or "").replace("{title}", "none")
        return await self._embedder.embed([prefix + t for t in texts])

    async def search(self, query: str, n: int) -> list[Chunk]:
        """Hybrid search (text vector + full text, RRF) of the active sources for `query`."""
        [vector] = await self._embedder.embed([(self.profile.query_prefix or "") + query])
        found = await self._store.candidates(self.ids, query, vector, n, None, ("text", "audio"))
        return reciprocal_rank_fusion(
            [found.by_vector, found.by_text], self._rag.rrf_k, [1.0, self._rag.text_weight]
        )


async def _plan(semantics: Semantics, chunks: list[Chunk], options: dict[str, Any]) -> list[Topic]:
    """Topics of the sources from the stored vectors (no LLM); [] when it cannot be done."""
    vectors = await semantics.vectors()
    items = [(c, vectors[c.id]) for c in chunks if c.id in vectors and c.content.strip()]
    if 0 < len(items) < SMALL_SOURCE:  # few chunks: embed their paragraphs for finer topics
        units = [(c, t) for c, _ in items for t in paragraphs(c.content)][:MAX_UNITS]
        if len(units) > len(items):
            embedded = await semantics.embed_documents([t for _, t in units])
            log.info(
                "concept_map: small source (%d chunks): embedded %d paragraphs (profile %s)",
                len(items), len(units), semantics.profile.name,
            )  # fmt: skip
            items = [(replace(c, content=t), v) for (c, t), v in zip(units, embedded, strict=True)]
    high = min(MAX_TOPICS, max(MIN_TOPICS, options.get("max_nodes", 30) // 4))
    slots = CALL_BUDGET * MAX_CALLS // (EXCERPT + 2)
    topics = await asyncio.to_thread(plan_topics, items, MIN_TOPICS, high, slots)
    log.info(
        "concept_map: clustered %d chunks into %d topics (profile %s, sizes %s)",
        len(items), len(topics), semantics.profile.name, [t.size for t in topics],
    )  # fmt: skip
    return topics


async def _label_semantics(semantics: Semantics, concepts: dict[str, dict]):
    """One batched embedding call for every concept: label only (synonyms) and label + summary
    (cross-links). Returns `(synonym groups, cross-branch candidate key pairs)`."""
    keys = list(concepts)
    texts = [concepts[k]["label"] for k in keys] + [
        f"{concepts[k]['label']}: {concepts[k]['summary']}" if concepts[k]["summary"]
        else concepts[k]["label"]
        for k in keys
    ]  # fmt: skip
    vecs = await semantics.embed_documents(texts)
    labels = {k: vecs[i] for i, k in enumerate(keys)}
    full = {k: vecs[len(keys) + i] for i, k in enumerate(keys)}
    top, mean, p90 = similarity_report(labels)
    log.info(
        "concept_map: embedded %d labels (profile %s); label cosine max %.3f mean %.3f p90 %.3f",
        len(keys), semantics.profile.name, top, mean, p90,
    )  # fmt: skip
    ranked = close_pairs(labels, -1.0)[:8]
    log.info(
        "concept_map: closest label pairs: %s",
        "; ".join(
            f"{concepts[a]['label']} ~ {concepts[b]['label']} = {s:.3f}" for s, a, b in ranked
        ),
    )
    groups = []
    merged_keys: set[str] = set()
    for score, a, b in close_pairs(labels, SYNONYM_COSINE):
        if concepts[a]["kind"] != concepts[b]["kind"] or {a, b} & merged_keys:
            continue
        merged_keys.add(b)
        groups.append((concepts[a]["label"], [concepts[b]["label"]]))
        log.info(
            "concept_map: synonym merge %r + %r (cosine %.3f >= %.2f)",
            concepts[a]["label"], concepts[b]["label"], score, SYNONYM_COSINE,
        )  # fmt: skip
    topic = {k: concepts[k]["topic"] for k in keys}
    cross = [
        (a, b)
        for s, a, b in close_pairs(full, CROSS_COSINE, topic)
        if b not in merged_keys and a not in merged_keys
    ][:MAX_CROSS]
    best = close_pairs(full, -1.0, topic)[:3]
    log.info(
        "concept_map: cross-branch candidates %d (>= %.2f); top cosines %s",
        len(cross), CROSS_COSINE, [round(s, 3) for s, _, _ in best],
    )  # fmt: skip
    return groups, cross


async def _generate_topics(
    semantics: Semantics, chunks: list[Chunk], aliases, catalog, options, chat
) -> dict | None:
    try:
        topics = await _plan(semantics, chunks, options)
    except Exception:
        log.exception("concept_map: clustering failed; falling back to positional sampling")
        return None
    if len(topics) < 2:
        return None
    answers = [
        (await chat.complete(messages)).content or ""
        for messages in topic_batches(options, topics, aliases, catalog)
    ]
    branches: dict[int, str] = {}
    concepts, props = parse_extraction(answers, chunks, aliases, branches)
    if not concepts:
        return None
    assign_topics(concepts, topic_of_chunks(topics), {t.index for t in topics})
    groups, cross = [], []
    try:
        groups, cross = await _label_semantics(semantics, concepts)
    except Exception:
        log.exception("concept_map: label embeddings failed; merging by the model only")
    merge_text = (
        await chat.complete(merge_messages_topics(concepts, props, branches, cross, options))
    ).content
    merge = parse_merge(merge_text or "")
    merge["groups"] = [*merge["groups"], *groups]
    extra = cross_props(merge.get("cross", []), concepts, chunks, aliases)
    log.info(
        "concept_map: model confirmed %d/%d cross-link proposals with a verified quote",
        len(extra),
        len(cross),
    )
    built = build_topic_map(concepts, [*props, *extra], branches, merge, options)
    return built or parse_map(answers, merge_text, chunks, aliases, options)


async def _context(
    chats: ChatRepository | None, docs: DocumentRepository, chunks: list[Chunk], chat_id: UUID
):
    sources = await readable_documents(chats, docs, chat_id)
    aliases = assign_aliases(await docs.list_for_chat(chat_id))
    by_doc: dict[UUID, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.document_id, []).append(c)
    catalog = "\n".join(_line(aliases[d.id], d, by_doc.get(d.id, [])) for d in sources)
    return aliases, catalog


async def generate_concept_map(
    chats: ChatRepository | None,
    docs: DocumentRepository,
    chunks: list[Chunk],
    chat_id: UUID,
    options: dict[str, Any],
    chat,
    semantics: Semantics | None = None,
) -> dict | None:
    """The verified map: 1-2 extraction calls + 1 merge call. With embeddings the topics come
    from clustering the chunk vectors; without them, from positional sampling."""
    aliases, catalog = await _context(chats, docs, chunks, chat_id)
    if semantics is not None and await semantics.bind(
        await readable_documents(chats, docs, chat_id)
    ):
        built = await _generate_topics(semantics, chunks, aliases, catalog, options, chat)
        if built is not None:
            return built
    answers = [
        (await chat.complete(messages)).content or ""
        for messages in extract_batches(options, chunks, aliases, catalog)
    ]
    concepts, props = parse_extraction(answers, chunks, aliases)
    merge_text = None
    if concepts and props:
        merge_text = (await chat.complete(merge_messages(concepts, props, options))).content
    return parse_map(answers, merge_text, chunks, aliases, options)


class ExpandConceptMap:
    """Grows a stored concept map around one node (1 LLM call) and persists the result."""

    def __init__(
        self,
        chats: ChatRepository,
        docs: DocumentRepository,
        chunks: ChunkStore,
        providers: ProviderRepository,
        factory: ProviderFactory,
        artifacts,
        rag=None,
    ):
        self._semantics = Semantics(chunks, providers, factory, rag)
        self._chats = chats
        self._docs = docs
        self._chunks = chunks
        self._providers = providers
        self._factory = factory
        self._artifacts = artifacts

    async def execute(self, chat_id: UUID, artifact_id: UUID, node_id: str):
        await require_chat(self._chats, chat_id)
        artifact = await self._artifacts.get(artifact_id)
        if artifact is None or artifact.chat_id != chat_id:
            raise NotFound("artifact_not_found", "Material no encontrado")
        if artifact.kind != MAP_KIND or artifact.status != "ready" or not artifact.data:
            raise Conflict("not_expandable", "Este material no es un mapa conceptual listo")
        node = node_by_id(artifact.data, node_id)
        if node is None:
            raise InvalidInput("node_not_found", "El concepto no existe en el mapa")
        profile = await _require_chat_profile(self._providers)
        chunks = await readable_chunks(self._chats, self._docs, self._chunks, chat_id)
        if not chunks:
            raise ProviderError("El documento no tiene fragmentos", "document_empty")
        if len(artifact.data["nodes"]) >= MAX_TOTAL_NODES:
            raise Conflict("map_full", f"El mapa alcanzó el máximo de {MAX_TOTAL_NODES} conceptos")
        aliases, catalog = await _context(self._chats, self._docs, chunks, chat_id)
        picked = await self._related(chat_id, artifact.data, node, chunks)
        answer = await self._factory.chat(profile).complete(
            expand_messages(artifact.data, node, chunks, aliases, catalog, picked)
        )
        grown = merge_expansion(artifact.data, node_id, [answer.content or ""], chunks, aliases)
        if grown is None:
            log.warning("concept_map: expand gave nothing verifiable: %.500s", answer.content)
            raise ProviderError("No se encontraron subconceptos verificables", INVALID_RESPONSE)
        data, added = grown
        await self._artifacts.update_data(artifact_id, data)
        return await self._artifacts.get(artifact_id), added

    async def _related(
        self, chat_id: UUID, data: dict, node: dict, chunks: list[Chunk]
    ) -> list[Chunk] | None:
        """Chunks the embedding retriever finds for the node (label + summary), minus the ones
        its branch already rests on; None (lexical fallback) when embeddings are unavailable."""
        try:
            sem = self._semantics
            if not await sem.bind(await readable_documents(self._chats, self._docs, chat_id)):
                return None
            query = f"{node['label']}. {node['summary']}".strip()
            found = await sem.search(query, 40)
            by_id = {c.id: c for c in chunks}
            used = _branch_chunks(data, node["id"])
            fresh = [by_id[c.id] for c in found if c.id in by_id and str(c.id) not in used]
            seen = [by_id[c.id] for c in found if c.id in by_id and str(c.id) in used]
            # new ground first, but the branch's own best chunks stay as context (2 of the 12)
            picked = [*fresh[: EXPAND_CHUNKS - EXPAND_OWN], *seen[:EXPAND_OWN]]
            picked += [c for c in [*fresh, *seen] if c not in picked][: EXPAND_CHUNKS - len(picked)]
            log.info(
                "concept_map: embedded query for expand %r (profile %s): %d candidates, "
                "%d fresh, %d already used by the branch -> %d chunks",
                node["label"], sem.profile.name, len(found), len(fresh), len(seen), len(picked),
            )  # fmt: skip
            return picked or None
        except Exception:
            log.exception("concept_map: semantic retrieval failed; using keyword match")
            return None


def _branch_chunks(data: dict, node_id: str) -> set[str]:
    """Chunk ids cited by the node and everything hanging from it (hierarchy links)."""
    children: dict[str, list[str]] = {}
    for lk in data["links"]:
        if lk.get("kind") == "jerarquia":
            children.setdefault(lk["source"], []).append(lk["target"])
    nodes = {n["id"]: n for n in data["nodes"]}
    seen, stack, used = set(), [node_id], set()
    while stack:
        k = stack.pop()
        if k in seen or k not in nodes:
            continue
        seen.add(k)
        used |= {r["chunk_id"] for r in nodes[k]["refs"] if r.get("chunk_id")}
        stack += children.get(k, [])
    return used
