"""Chunk inspector: how a document was split (debug aid)."""

from uuid import UUID

from chat_pdf.application.ports import ChunkStore, DocumentRepository
from chat_pdf.domain.errors import NotFound
from chat_pdf.domain.models import Chunk


class ListChunks:
    def __init__(self, docs: DocumentRepository, chunks: ChunkStore):
        self._docs = docs
        self._chunks = chunks

    async def execute(
        self, document_id: UUID, page: int | None, offset: int, limit: int
    ) -> tuple[int, list[Chunk]]:
        """(total matching, the `offset`/`limit` slice) in reading order; `page` keeps the
        chunks whose page range contains it."""
        if await self._docs.get(document_id) is None:
            raise NotFound("document_not_found", "Documento no encontrado")
        found = await self._chunks.for_document(document_id)
        if page is not None:
            found = [c for c in found if _covers(c, page)]
        return len(found), found[offset : offset + limit]


def _covers(chunk: Chunk, page: int) -> bool:
    if chunk.page_start is None:
        return False
    return chunk.page_start <= page <= (chunk.page_end or chunk.page_start)
