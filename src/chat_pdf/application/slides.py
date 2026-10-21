"""Slide decks: suitability pre-check and the generation of a deck (map over sections)."""

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
from chat_pdf.domain.errors import Conflict, ProviderError
from chat_pdf.domain.models import Chunk
from chat_pdf.domain.slides import (
    assess_messages,
    default_slide_count,
    parse_assessment,
    parse_deck,
    slide_batches,
)
from chat_pdf.domain.sources import assign_aliases


class AssessSlides:
    """One cheap LLM call over the catalog, the overview and chunks sampled across ALL the
    active sources: is this material worth a deck?"""

    def __init__(
        self,
        chats: ChatRepository,
        docs: DocumentRepository,
        chunks: ChunkStore,
        providers: ProviderRepository,
        factory: ProviderFactory,
    ):
        self._chats = chats
        self._docs = docs
        self._chunks = chunks
        self._providers = providers
        self._factory = factory

    async def execute(self, chat_id: UUID) -> dict[str, Any]:
        await require_chat(self._chats, chat_id)
        sources = await readable_documents(self._chats, self._docs, chat_id)
        if not sources:
            raise Conflict("document_not_ready", "El documento no está listo")
        profile = await _require_chat_profile(self._providers)
        chunks = await readable_chunks(self._chats, self._docs, self._chunks, chat_id)
        if not chunks:
            raise ProviderError("El documento no tiene fragmentos", "document_empty")
        aliases = assign_aliases(await self._docs.list_for_chat(chat_id))
        by_doc: dict[UUID, list[Chunk]] = {}
        for c in chunks:
            by_doc.setdefault(c.document_id, []).append(c)
        entries = [(aliases[d.id], d, by_doc.get(d.id, [])) for d in sources]
        answer = await self._factory.chat(profile).complete(assess_messages(entries, chunks))
        result = parse_assessment(answer.content or "", default_slide_count(chunks))
        if result is None:
            raise ProviderError("El modelo no devolvió contenido válido", INVALID_RESPONSE)
        return result


async def generate_deck(
    chats: ChatRepository | None,
    docs: DocumentRepository,
    chunks: list[Chunk],
    chat_id: UUID,
    options: dict[str, Any],
    chat,
) -> dict | None:
    """The verified deck (<= 3 LLM calls). `chat` is the provider's chat client."""
    sources = await readable_documents(chats, docs, chat_id)
    aliases = assign_aliases(await docs.list_for_chat(chat_id))
    by_doc: dict[UUID, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.document_id, []).append(c)
    catalog = "\n".join(_line(aliases[d.id], d, by_doc.get(d.id, [])) for d in sources)
    answers = [
        (await chat.complete(messages)).content or ""
        for messages in slide_batches(options, chunks, aliases, catalog)
    ]
    return parse_deck(answers, chunks, aliases, options)
