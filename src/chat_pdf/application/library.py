"""The library: all documents, global concept search, asking a selection of them."""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from uuid import UUID

from chat_pdf.application.ports import (
    ChatRepository,
    FileStorage,
    ProviderFactory,
    ProviderRepository,
    Reranker,
    Retriever,
)
from chat_pdf.application.ports_library import LibraryRepository
from chat_pdf.application.projects import ProjectService, clean_name
from chat_pdf.domain.calibration import (
    above_threshold,
    calibration_table,
    chunk_relevance,
    modality_relevance,
)
from chat_pdf.domain.config import RagConfig
from chat_pdf.domain.errors import Conflict, InvalidInput, NotFound
from chat_pdf.domain.library import LibraryDoc, blend
from chat_pdf.domain.models import CITATION_KIND, Chunk
from chat_pdf.domain.retrieval import plan_rankings, reciprocal_rank_fusion
from chat_pdf.domain.sources import MAX_SOURCES, chat_title

log = logging.getLogger(__name__)

MAX_RESULTS = 50
HITS_PER_DOC = 3
SNIPPET_CHARS = 220
BLOCK = 1024 * 1024


@dataclass(frozen=True)
class DocHit:
    """One passage of a document that answers the query."""

    chunk_id: UUID | None
    kind: str  # text | image | audio | video
    page: int | None
    page_end: int | None
    t_start: float | None
    t_end: float | None
    bbox: list[float] | None
    has_thumb: bool
    snippet: str
    score: float | None


@dataclass(frozen=True)
class SearchResult:
    doc: LibraryDoc
    score: float  # blend of the best passage and the document vector, in [0, 1]
    relevant: bool  # a passage / the document clears the profile's evidence threshold
    doc_score: float | None = None
    hits: list[DocHit] = field(default_factory=list)


@dataclass(frozen=True)
class SearchResponse:
    query: str
    results: list[SearchResult]
    searched: int  # ready documents that were searched
    skipped_unready: int = 0  # documents of the filter that are not `ready` yet
    skipped_profiles: int = 0  # documents embedded by a profile that is gone / unreachable
    notes: list[str] = field(default_factory=list)  # why some documents were not searched


def _hit(chunk: Chunk, score: float | None) -> DocHit:
    snippet = " ".join(chunk.content.split())[:SNIPPET_CHARS]
    return DocHit(
        chunk.id, CITATION_KIND[chunk.modality], chunk.page_start, chunk.page_end, chunk.t_start,
        chunk.t_end, chunk.bbox, chunk.has_thumb, snippet, score,
    )  # fmt: skip


class LibraryService:
    def __init__(self, lib: LibraryRepository):
        self._lib = lib

    async def documents(self, **filters) -> list[LibraryDoc]:
        await self._lib.refresh_vectors()  # lazily: documents ingested since the last visit
        return await self._lib.library_documents(**filters)

    async def backfill(self) -> int:
        """Computes the document vector of every ready document that lacks a current one."""
        return len(await self._lib.refresh_vectors())


class GlobalSearch:
    """Concept search over every document: the query is embedded with EACH document's own
    profile (vectors of different models never meet), documents are ranked by the cosine of
    their vector blended with their best passages (hybrid retrieval over all of them), and the
    top passages of each document come back for the viewer to open."""

    def __init__(
        self,
        lib: LibraryRepository,
        providers: ProviderRepository,
        retriever: Retriever,
        reranker: Reranker,
        factory: ProviderFactory,
        config: RagConfig,
    ):
        self._lib = lib
        self._providers = providers
        self._retriever = retriever
        self._reranker = reranker
        self._factory = factory
        self._cfg = config

    async def execute(
        self,
        query: str,
        limit: int = 20,
        kind: str | None = None,
        category_id: UUID | None = None,
        project_id: UUID | None = None,
        document_ids: list[UUID] | None = None,
    ) -> SearchResponse:
        query = " ".join(query.split())
        if not query:
            raise InvalidInput("empty_query", "Escribí qué querés buscar")
        await self._lib.refresh_vectors()
        docs = await self._lib.library_documents(
            kind=kind, category_id=category_id, project_id=project_id, ids=document_ids
        )
        ready = [d for d in docs if d.status == "ready"]
        by_profile: dict[UUID, list[LibraryDoc]] = {}
        for d in ready:
            by_profile.setdefault(d.embedding_profile_id, []).append(d)
        results: list[SearchResult] = []
        skipped = 0
        notes: list[str] = []
        for profile_id, group in by_profile.items():
            profile = await self._providers.get(profile_id)
            if profile is None:
                skipped += len(group)
                notes.append(f"{len(group)} documento(s) con un perfil de embeddings borrado")
                continue
            try:
                results += await self._search_group(profile, group, query)
            except Exception:
                log.exception("global search failed for profile %s", profile_id)
                skipped += len(group)
                notes.append(
                    f"{len(group)} documento(s) omitidos: el perfil «{profile.name}» no responde "
                    "(cada documento se busca con su propio perfil de embeddings)"
                )
        results.sort(key=lambda r: -r.score)
        return SearchResponse(
            query, results[: min(max(limit, 1), MAX_RESULTS)], len(ready) - skipped,
            len(docs) - len(ready), skipped, notes,
        )  # fmt: skip

    async def _search_group(
        self, profile, group: list[LibraryDoc], query: str
    ) -> list[SearchResult]:
        ids = [d.id for d in group]
        table = calibration_table(profile.calibration, self._cfg.weak_evidence_threshold)
        [vector] = await self._factory.embedder(profile).embed(
            [(profile.query_prefix or "") + query]
        )
        doc_cos = await self._lib.document_scores(ids, vector)
        plan = plan_rankings(
            "auto", self._cfg.text_weight, self._cfg.native_weight, self._cfg.native_hint_weight
        )
        n = min(200, max(40, 5 * len(ids)))
        found = await self._retriever.candidates(ids, query, vector, n, None, plan.modalities)
        fused = reciprocal_rank_fusion(
            [found.by_vector, found.by_text, found.by_image, found.by_audio],
            self._cfg.rrf_k,
            list(plan.weights),
        )
        by_doc: dict[UUID, list[Chunk]] = {}
        for chunk in fused:
            if chunk.document_id is not None:
                by_doc.setdefault(chunk.document_id, []).append(chunk)
        out = []
        for doc in group:
            chunks = by_doc.get(doc.id, [])[:HITS_PER_DOC]
            scores = [chunk_relevance(c, table) for c in chunks]
            best = max((s for s in scores if s is not None), default=None)
            cos = doc_cos.get(doc.id)
            doc_rel = None if cos is None else modality_relevance(cos, "text", table)
            relevant = bool(
                (cos is not None and above_threshold(cos, "text", table))
                or any(
                    (c.similarity is not None and above_threshold(c.similarity, "text", table))
                    or (
                        c.native_similarity is not None
                        and above_threshold(c.native_similarity, c.modality, table)
                    )
                    for c in chunks
                )
            )
            score = blend(doc_rel, best)
            if not chunks and not relevant:
                continue  # nothing surfaced and the document vector says it is unrelated
            out.append(
                SearchResult(
                    doc,
                    round(score, 4),
                    relevant,
                    None if doc_rel is None else round(doc_rel, 3),
                    [_hit(c, s) for c, s in zip(chunks, scores, strict=True)],
                )  # fmt: skip
            )
        return out


@dataclass(frozen=True)
class AskGroupResult:
    chat_id: UUID
    project_id: UUID | None
    document_ids: list[UUID]


async def _file_blocks(path) -> AsyncIterator[bytes]:
    import asyncio

    with path.open("rb") as handle:
        while block := await asyncio.to_thread(handle.read, BLOCK):
            yield block


class AskDocuments:
    """ "Preguntar a estos documentos": puts COPIES of the selected documents into a new chat or a
    new project and starts a chat there (a source belongs to one collection; sharing one across
    collections is deferred, and moving would break the original chat). The copy takes the
    stored file and is re-ingested in the background with the original's embedding profile."""

    def __init__(
        self,
        lib: LibraryRepository,
        chats: ChatRepository,
        projects: ProjectService,
        storage: FileStorage,
    ):
        self._lib = lib
        self._chats = chats
        self._projects = projects
        self._storage = storage

    async def execute(
        self,
        document_ids: list[UUID],
        mode: str = "chat",
        name: str | None = None,
        project_id: UUID | None = None,
    ) -> AskGroupResult:
        ids = list(dict.fromkeys(document_ids))
        if not ids:
            raise InvalidInput("no_documents", "Elegí al menos un documento")
        if len(ids) > MAX_SOURCES:
            raise Conflict("too_many_sources", f"Hasta {MAX_SOURCES} documentos por chat")
        sources = []
        for document_id in ids:
            src = await self._lib.document_source(document_id)
            if src is None:
                raise NotFound("document_not_found", "Documento no encontrado")
            if src["status"] != "ready" or not self._storage.path(document_id).is_file():
                raise Conflict("document_not_ready", f"«{src['filename']}» no está disponible")
            sources.append(src)
        if len({s["embedding_profile_id"] for s in sources}) > 1:
            raise Conflict(
                "embedding_profile_mismatch",
                "Los documentos usan perfiles de embeddings distintos: elegí del mismo perfil",
            )
        chat_id: UUID
        target: UUID
        if project_id is not None:
            await self._projects.require(project_id)
            existing = await self._lib.collection_documents(project_id)
            if len(existing) + len(sources) > MAX_SOURCES:
                raise Conflict("too_many_sources", "El proyecto superaría el máximo de fuentes")
            if any(
                d.embedding_profile_id != sources[0]["embedding_profile_id"]
                for d in existing
                if d.status != "failed"
            ):
                raise Conflict(
                    "embedding_profile_mismatch", "El proyecto usa otro perfil de embeddings"
                )
            target = project_id
            chat = await self._lib.create_chat_in(
                target, (await self._projects.require(target)).name
            )
            chat_id = chat.id
        elif mode == "project":
            title = clean_name(name or f"Selección de {len(sources)} documentos")
            project = await self._projects.create(title, None, None, None)
            target = project.id
            chat_id = (await self._lib.create_chat_in(target, project.name)).id
        else:
            chat = await self._chats.create()
            located = await self._lib.collection_of_chat(chat.id)
            assert located is not None
            target, chat_id = located[0], chat.id
        await self._chats.set_title(chat_id, clean_name(name) if name else _title(sources, mode))
        new_ids = []
        for src in sources:
            new_id = await self._copy(src, target)
            new_ids.append(new_id)
        return AskGroupResult(
            chat_id, target if (project_id or mode == "project") else None, new_ids
        )

    async def _copy(self, src: dict, collection_id: UUID) -> UUID:
        from uuid import uuid4

        new_id = uuid4()
        origin = self._storage.path(src["id"])
        try:
            await self._storage.save(new_id, _file_blocks(origin), origin.suffix)
            await self._lib.add_document(
                new_id, collection_id, src["filename"], src["mime"], src["sha256"],
                src["embedding_profile_id"], src["kind"],
            )  # fmt: skip
        except BaseException:
            await self._storage.delete(new_id)
            raise
        return new_id


def _title(sources: list[dict], mode: str) -> str:
    return chat_title([s["filename"] for s in sources])
