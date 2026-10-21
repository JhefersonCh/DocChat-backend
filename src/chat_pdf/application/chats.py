from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from chat_pdf.application.ports import (
    ChatRepository,
    DocumentRepository,
    FileStorage,
    UnitOfWorkFactory,
)
from chat_pdf.domain.errors import InvalidInput, NotFound
from chat_pdf.domain.models import Chat, Document, Message
from chat_pdf.domain.sources import CollectionOverview, chat_title, collection_overview, ordered


async def require_chat(chats: ChatRepository, chat_id: UUID) -> Chat:
    chat = await chats.get(chat_id)
    if chat is None:
        raise NotFound("chat_not_found", "Chat no encontrado")
    return chat


@dataclass(frozen=True)
class ChatDetail:
    chat: Chat
    sources: list[Document] = field(default_factory=list)  # in the order added: S1..Sn
    overview: CollectionOverview | None = None  # aggregate of the sources (2 or more)

    @property
    def document(self) -> Document | None:
        """Primary source (S1), kept for clients written when a chat had one document."""
        return self.sources[0] if self.sources else None


class CreateChat:
    def __init__(self, uow: UnitOfWorkFactory):
        self._uow = uow

    async def execute(self) -> Chat:
        async with self._uow() as uow:
            return await uow.chats.create()


class DeleteChat:
    def __init__(self, uow: UnitOfWorkFactory, storage: FileStorage):
        self._uow = uow
        self._storage = storage

    async def execute(self, chat_id: UUID) -> None:
        async with self._uow() as uow:
            await require_chat(uow.chats, chat_id)
            document_ids = await uow.chats.delete(chat_id)
        for document_id in document_ids:  # after the commit: a rollback must not lose files
            await self._storage.delete(document_id)


class UpdateChatSettings:
    def __init__(self, chats: ChatRepository, docs: DocumentRepository | None = None):
        self._chats = chats
        self._docs = docs

    async def execute(self, chat_id: UUID, changes: dict[str, Any]) -> Chat:
        """Applies the given style / custom_instructions / length; absent keys stay as they are.

        `active_source_ids` (the sources the chat reads) is stored as the muted complement."""
        await require_chat(self._chats, chat_id)
        changes = dict(changes)
        active = changes.pop("active_source_ids", None)
        if active is not None and self._docs is not None:
            docs = await self._docs.list_for_chat(chat_id)
            unknown = set(active) - {d.id for d in docs}
            if unknown:
                raise InvalidInput("unknown_source", "Alguna de las fuentes no es de este chat")
            changes["disabled_sources"] = [d.id for d in ordered(docs) if d.id not in active]
        return await self._chats.update_settings(chat_id, changes)


class RemoveSource:
    """Deletes one source of a chat (document, chunks, assets); the title follows the rest."""

    def __init__(self, uow: UnitOfWorkFactory, storage: FileStorage):
        self._uow = uow
        self._storage = storage

    async def execute(self, chat_id: UUID, document_id: UUID) -> None:
        async with self._uow() as uow:
            await uow.chats.lock(chat_id)
            chat = await require_chat(uow.chats, chat_id)
            docs = ordered(await uow.docs.list_for_chat(chat_id))
            if document_id not in {d.id for d in docs}:
                raise NotFound("source_not_found", "La fuente no es de este chat")
            await uow.docs.delete(document_id)
            rest = [d for d in docs if d.id != document_id]
            await uow.chats.set_title(chat_id, chat_title([d.filename for d in rest]))
            if document_id in chat.disabled_sources:
                muted = [i for i in chat.disabled_sources if i != document_id]
                await uow.chats.update_settings(chat_id, {"disabled_sources": muted})
        await self._storage.delete(document_id)  # after the commit: a rollback must not lose files


class ListChats:
    def __init__(self, chats: ChatRepository):
        self._chats = chats

    async def execute(self) -> list[Chat]:
        return await self._chats.list()


class GetChat:
    def __init__(self, chats: ChatRepository, docs: DocumentRepository):
        self._chats = chats
        self._docs = docs

    async def execute(self, chat_id: UUID) -> ChatDetail:
        chat = await require_chat(self._chats, chat_id)
        docs = ordered(await self._docs.list_for_chat(chat_id))
        return ChatDetail(chat, docs, collection_overview(docs))


class ListMessages:
    def __init__(self, chats: ChatRepository):
        self._chats = chats

    async def execute(self, chat_id: UUID) -> list[Message]:
        await require_chat(self._chats, chat_id)
        return await self._chats.messages(chat_id)
