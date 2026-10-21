from dataclasses import dataclass
from uuid import UUID

from chat_pdf.application.ask import DocumentSearch
from chat_pdf.application.ports import (
    DocumentRepository,
    ProviderFactory,
    ProviderRepository,
    Reranker,
    Retriever,
)
from chat_pdf.domain.config import RagConfig
from chat_pdf.domain.errors import Conflict, NotFound
from chat_pdf.domain.models import Citation

MAX_HITS = 20


@dataclass(frozen=True)
class SearchHit:
    """A passage of the document that matches the query (no LLM involved)."""

    chunk_id: UUID | None
    page_start: int | None
    page_end: int | None
    section: str | None
    snippet: str
    content: str | None
    score: float | None  # relevance in [0, 1]
    kind: str = "text"  # text | image | audio | video (same contract as Citation)
    bbox: list[float] | None = None
    t_start: float | None = None
    t_end: float | None = None
    has_thumb: bool = False

    @classmethod
    def of(cls, c: Citation) -> "SearchHit":
        return cls(
            c.chunk_id, c.page, c.page_end, c.section, c.snippet, c.content, c.score,
            c.kind, c.bbox, c.t_start, c.t_end, c.has_thumb,
        )  # fmt: skip


class SearchDocument:
    """Hybrid retrieval over one document, embedding the query with the document's own profile."""

    def __init__(
        self,
        docs: DocumentRepository,
        providers: ProviderRepository,
        retriever: Retriever,
        reranker: Reranker,
        factory: ProviderFactory,
        config: RagConfig,
    ):
        self._docs = docs
        self._providers = providers
        self._retriever = retriever
        self._reranker = reranker
        self._factory = factory
        self._cfg = config

    async def execute(self, document_id: UUID, query: str, k: int) -> list[SearchHit]:
        doc = await self._docs.get(document_id)
        if doc is None:
            raise NotFound("document_not_found", "Documento no encontrado")
        if doc.status != "ready":
            raise Conflict("document_not_ready", f"El documento está en estado {doc.status}")
        profile = await self._providers.get(doc.embedding_profile_id)
        if profile is None:
            raise Conflict(
                "embedding_profile_missing", "El perfil de embeddings del documento ya no existe"
            )
        search = DocumentSearch(
            self._retriever,
            self._reranker,
            self._factory.embedder(profile),
            profile,
            doc,
            self._cfg,
        )
        sources = await search.run(query, min(max(k, 1), MAX_HITS))
        return [SearchHit.of(s.cite(self._cfg.snippet_chars, search.table)) for s in sources]
