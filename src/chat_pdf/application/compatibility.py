"""Source compatibility alerts: inform (never block) when the sources of a chat / project / a
library selection do not seem to be about related things. See domain/compatibility.py."""

import asyncio
import logging
from dataclasses import replace
from uuid import UUID

from chat_pdf.application.chats import require_chat
from chat_pdf.application.ports import (
    ChatRepository,
    DocumentRepository,
    ProviderFactory,
    ProviderRepository,
)
from chat_pdf.application.ports_library import LibraryRepository
from chat_pdf.domain.compatibility import (
    Compatibility,
    CompatItem,
    analyze,
    cache_key,
    clean_reason,
    flagged,
    reason_messages,
    topic_vectors,
)
from chat_pdf.domain.errors import InvalidInput, NotFound
from chat_pdf.domain.library import LibraryDoc
from chat_pdf.domain.models import Document
from chat_pdf.domain.sources import MAX_SOURCES, assign_aliases

log = logging.getLogger(__name__)

REASON_TIMEOUT_S = 20
CACHE_SIZE = 128


class CompatibilityService:
    def __init__(
        self,
        lib: LibraryRepository,
        chats: ChatRepository,
        docs: DocumentRepository,
        providers: ProviderRepository,
        factory: ProviderFactory,
    ):
        self._lib = lib
        self._chats = chats
        self._docs = docs
        self._providers = providers
        self._factory = factory
        self._reasons: dict[frozenset, str | None] = {}  # one-line reason by source set

    async def for_chat(self, chat_id: UUID) -> Compatibility:
        """Active (not muted) sources of the chat; aliases are the chat's S1..Sn."""
        chat = await require_chat(self._chats, chat_id)
        docs = await self._docs.list_for_chat(chat_id)
        aliases = assign_aliases(docs)
        active = [d for d in docs if d.id not in chat.disabled_sources]
        return await self._check([(d, aliases[d.id]) for d in active])

    async def for_project(self, project_id: UUID) -> Compatibility:
        if await self._lib.get_project(project_id) is None:
            raise NotFound("project_not_found", "Proyecto no encontrado")
        docs = await self._lib.collection_documents(project_id)
        aliases = assign_aliases(docs)
        return await self._check([(d, aliases[d.id]) for d in docs])

    async def for_documents(self, ids: list[UUID]) -> Compatibility:
        """A library selection (before a chat exists): aliases follow the selection order."""
        ids = list(dict.fromkeys(ids))
        if not ids or len(ids) > MAX_SOURCES:
            raise InvalidInput("invalid_selection", f"Elegí entre 1 y {MAX_SOURCES} documentos")
        found = {d.id: d for d in await self._lib.library_documents(ids=ids)}
        if any(i not in found for i in ids):
            raise NotFound("document_not_found", "Documento no encontrado")
        picked = [(found[i], f"S{n}") for n, i in enumerate(ids, start=1)]
        return await self._check(picked)

    async def _check(self, pairs: list[tuple[Document | LibraryDoc, str]]) -> Compatibility:
        pairs = [(d, a) for d, a in pairs if d.status != "failed"]
        ready = [d.id for d, _ in pairs if d.status == "ready"]
        if len(pairs) < 2 or not ready:
            return analyze(_items(pairs, {}))
        items = _items(pairs, await self._lib.compat_vectors(ready))
        result = analyze(items)
        marked = flagged(result)
        if not marked:
            return result
        reason = await self._reason(items, marked)
        if reason is None:
            return result
        return replace(
            result,
            sources=[replace(s, reason=reason) if s in marked else s for s in result.sources],
        )

    async def _reason(self, items: list[CompatItem], marked) -> str | None:
        key = cache_key([i for i in items if i.vectors]) | {
            ("marked", m.document_id) for m in marked
        }
        if key in self._reasons:
            return self._reasons[key]
        reason = None
        profile = await self._providers.active("chat")
        if profile is not None:
            try:
                reply = await asyncio.wait_for(
                    self._factory.chat(profile).complete(reason_messages(items, marked)),
                    REASON_TIMEOUT_S,
                )
                reason = clean_reason(reply.content)
            except Exception:
                log.warning("compatibility reason failed", exc_info=True)
        if len(self._reasons) >= CACHE_SIZE:
            self._reasons.pop(next(iter(self._reasons)))
        self._reasons[key] = reason
        return reason


def _items(pairs: list[tuple[Document | LibraryDoc, str]], state: dict) -> list[CompatItem]:
    out = []
    for doc, alias in pairs:
        sha, n, rows = state.get(doc.id, (None, None, []))
        out.append(
            CompatItem(
                doc.id,
                alias,
                doc.filename,
                doc.embedding_profile_id,
                topic_vectors(rows),
                n,
                doc.summary,
                sha,
            )
        )
    return out
