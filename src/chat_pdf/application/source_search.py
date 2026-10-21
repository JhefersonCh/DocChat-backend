"""One answer's retrieval over the readable sources of a chat (one or many)."""

from dataclasses import replace
from uuid import UUID

from chat_pdf.application.find_all import ExhaustiveFinder
from chat_pdf.application.ports import Embedder, Reranker, Retriever
from chat_pdf.domain.calibration import CalibrationTable, calibration_table, chunk_relevance
from chat_pdf.domain.catalog import collection_catalog, has_media
from chat_pdf.domain.config import RagConfig
from chat_pdf.domain.find_all import FindAllResult
from chat_pdf.domain.models import CITATION_KIND, Chunk, Document, Search, Source, SourceRef
from chat_pdf.domain.panorama import (
    collection_index,
    has_low_coverage,
    panorama_chunks,
)
from chat_pdf.domain.prompting import format_find_all
from chat_pdf.domain.prompts import FIND_ALL_NO_MEDIA, PANORAMA_QUERY, TOOL_UNKNOWN_SOURCE
from chat_pdf.domain.providers import ProviderProfile
from chat_pdf.domain.retrieval import SourceSet, parse_hint, plan_rankings, reciprocal_rank_fusion
from chat_pdf.domain.segments import with_parent_text
from chat_pdf.domain.sources import (
    ALL,
    assign_aliases,
    balance_sources,
    parse_source_filter,
)

PAGE_CONTEXT_CHARS = 300
Event = tuple[str, object]  # (SSE event name, payload); the HTTP layer serializes it


def _found(sources: list[Source], table: CalibrationTable | None = None) -> list[dict]:
    """Where each retrieved chunk sits, for the viewer's X-ray minimap (JSON-ready)."""
    return [
        {
            "n": s.n,
            "chunk_id": str(s.chunk.id) if s.chunk.id else None,
            "page_start": s.chunk.page_start,
            "page_end": s.chunk.page_end,
            "section": s.chunk.section,
            "score": chunk_relevance(s.chunk, table),
            "kind": CITATION_KIND[s.chunk.modality],
            "bbox": s.chunk.bbox,
            "t_start": s.chunk.t_start,
            "t_end": s.chunk.t_end,
            "has_thumb": s.chunk.has_thumb,
            "document_id": str(s.document_id) if s.document_id else None,
            "source_alias": s.alias,
            "region_label": s.chunk.region_label,
        }
        for s in sources
    ]


class DocumentSearch:
    """Searches the readable sources of a chat and accumulates the answer's SourceSet and the
    searches made. `doc` is one Document or the list of them (S1..Sn follow their order).

    Hybrid retrieval: vector and full-text rankings fused by RRF, then the Reranker, then top-k
    (balanced so that no source starves the others). Every search may be limited to some sources.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        embedder: Embedder,
        profile: ProviderProfile,
        doc: Document | list[Document],
        config: RagConfig,
        chunks: list[Chunk] | None = None,
        max_page: int | None = None,
        finder: ExhaustiveFinder | None = None,
        aliases: dict[UUID, str] | None = None,
        labeled: bool | None = None,
    ):
        self._retriever = retriever
        self._reranker = reranker
        self.max_page = max_page  # spoiler-safe reading: nothing past this page is retrieved
        self._chunks = [c for c in chunks or [] if self._readable(c)]
        self._embedder = embedder
        self._cfg = config
        self._prefix = profile.query_prefix or ""
        self._docs = [doc] if isinstance(doc, Document) else list(doc)
        self._aliases = aliases or assign_aliases(self._docs)
        refs = {d.id: SourceRef(d.id, self._aliases[d.id], d.filename, d.kind) for d in self._docs}
        self._by_alias = {r.alias: r.document_id for r in refs.values()}
        self._labeled = len(self._docs) > 1 if labeled is None else labeled
        self._sources = SourceSet(refs, self._labeled)
        self.searches: list[Search] = []
        self._hits: list[int] = []
        self._announced = 0
        self._finder = finder
        self.table = calibration_table(profile.calibration, config.weak_evidence_threshold)
        self.has_media = has_media(self._chunks)

    def _readable(self, chunk: Chunk) -> bool:
        page = chunk.page_start
        return self.max_page is None or page is None or page <= self.max_page

    @property
    def sources(self) -> list[Source]:
        return self._sources.sources

    @property
    def multi(self) -> bool:
        """The planner must choose among several readable sources."""
        return len(self._docs) > 1

    @property
    def aliases(self) -> list[str]:
        return [self._aliases[d.id] for d in self._docs]

    def resolve(self, raw: object) -> tuple[list[UUID] | None, str | None]:
        """The model's `source` argument -> (document ids or None for all, error text or None)."""
        found = parse_source_filter(raw, self._by_alias)
        if found == []:
            return None, TOOL_UNKNOWN_SOURCE.format(source=raw, available=", ".join(self.aliases))
        return found, None

    def _ids(self, only: list[UUID] | None) -> list[UUID]:
        return [d.id for d in self._docs if only is None or d.id in only]

    def _label(self, only: list[UUID] | None) -> str:
        ids = self._ids(only)
        if len(ids) == len(self._docs):
            return ALL
        return ",".join(self._aliases[i] for i in ids)

    async def run(
        self, query: str, k: int, modality: str = "auto", only: list[UUID] | None = None
    ) -> list[Source]:
        """Hybrid search. Four rankings, each a top-n inside its own modality (text vector, full
        text, native image, native audio), fused by weighted RRF; `modality` re-weights them."""
        hint = parse_hint(modality)
        plan = plan_rankings(
            hint, self._cfg.text_weight, self._cfg.native_weight, self._cfg.native_hint_weight
        )
        ids = self._ids(only)
        [vector] = await self._embedder.embed([self._prefix + query])
        n = max(self._cfg.search_candidates, k)
        found = await self._retriever.candidates(
            ids, query, vector, n, self.max_page, plan.modalities
        )
        rankings = [found.by_vector, found.by_text, found.by_image, found.by_audio]
        fused = reciprocal_rank_fusion(rankings, self._cfg.rrf_k, list(plan.weights))
        ranked = await self._reranker.rerank(query, fused)
        chunks = balance_sources(ranked, k, ids[0] if len(ids) == 1 else None)
        chunks = with_parent_text(chunks, self._chunks)
        sources = self._sources.add(chunks)
        counts = dict(zip(("vector", "text", "image", "audio"), map(len, rankings), strict=True))
        self.searches.append(
            Search(
                query, k, _found(sources, self.table), modality=hint, rankings=counts,
                source=self._label(only),
            )
        )  # fmt: skip
        self._hits.append(len(chunks))
        return sources

    async def find_all(
        self, query: str, modality: str = "imagen", only: list[UUID] | None = None
    ) -> str:
        """Exhaustive search of the media: every verified occurrence, as the tool's text."""
        if self._finder is None or not self.has_media:
            return FIND_ALL_NO_MEDIA
        hint = parse_hint(modality)
        [vector] = await self._embedder.embed([self._prefix + query])
        found = await self._finder.find(self._ids(only), query, vector, hint, self.max_page)
        sources = self._sources.add(with_parent_text(found.chunks, self._chunks))
        labels = {
            s.n: found.confidence[s.chunk.id] for s in sources if s.chunk.id in found.confidence
        }
        result = FindAllResult(sources, labels, found.checked, found.discarded, found.extra)
        counts = {"checked": found.checked, "discarded": found.discarded, "asked": found.asked}
        self.searches.append(
            Search(
                query, len(sources), _found(sources, self.table), False, hint, True, counts,
                self._label(only),
            )
        )  # fmt: skip
        self._hits.append(len(sources))
        return format_find_all(query, result)

    def _with_parents(self, chunks: list[Chunk]) -> list[Chunk]:
        """Small-to-big: a retrieved region / tile also carries its parent's description and the
        page text around it, so the model sees the whole picture. A parent retrieved in the same
        batch is already shown on its own: the child then keeps just its own text."""
        by_id = {c.id: c for c in self._chunks if c.id}
        present = {c.id for c in chunks}
        out = []
        for chunk in chunks:
            parent = by_id.get(chunk.parent_id) if chunk.parent_id else None
            if parent is None or parent.id in present or chunk.modality != "image":
                out.append(chunk)
                continue
            page = " ".join(
                c.content for c in self._chunks
                if c.modality == "text" and c.page_start == parent.page_start is not None
            )[:PAGE_CONTEXT_CHARS]  # fmt: skip
            text = parent.content.strip()
            if page:
                text = f"{text}\nTexto de la página: {page}".strip()
            out.append(replace(chunk, parent_text=text or None))
        return out

    def _entries(self) -> list[tuple[str, Document, list[Chunk]]]:
        sole = self._docs[0].id if len(self._docs) == 1 else None
        return [
            (
                self._aliases[d.id],
                d,
                [c for c in self._chunks if (c.document_id or sole) == d.id],
            )
            for d in self._docs
        ]

    @property
    def index(self) -> str:
        return collection_index(self._entries())

    @property
    def catalog(self) -> str:
        """Media counts of the sources for the planner ("" for one text-only document)."""
        return collection_catalog(self._entries())

    def add_panorama(self) -> int:
        """Add chunks spread over each source (one per uncovered section), the budget split
        evenly between sources; returns how many."""
        entries = self._entries()
        budget = self._cfg.max_context_chars // len(entries)
        chosen: list[Chunk] = []
        for _, doc, chunks in entries:
            own = [s for s in self.sources if s.document_id == doc.id]
            chosen += panorama_chunks(chunks, own, budget)
        if chosen:
            added = self._sources.add(chosen)
            self.searches.append(
                Search(PANORAMA_QUERY, len(chosen), _found(added, self.table), True)
            )
            self._hits.append(len(chosen))
        return len(chosen)

    def needs_panorama(self) -> bool:
        return has_low_coverage(self.sources, len(self.searches))

    def announce(self) -> list[Event]:
        """`search` events for the searches run since the last call (query/k as persisted)."""
        events = [
            (
                "search",
                {
                    "query": s.query,
                    "k": s.k,
                    "hits": h,
                    "found": s.found,
                    "panorama": s.panorama,
                    "modality": s.modality,
                    "exhaustive": s.exhaustive,
                    "rankings": s.rankings,
                    "source": s.source,
                },
            )
            for s, h in zip(
                self.searches[self._announced :], self._hits[self._announced :], strict=True
            )
        ]
        self._announced = len(self.searches)
        return events
