"""Notes: an assistant answer saved per chat, with a snapshot of its citations."""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from chat_pdf.application.chats import require_chat
from chat_pdf.application.ports import ChatRepository
from chat_pdf.domain.errors import NotFound

TITLE_MAX = 200


@dataclass(frozen=True)
class Note:
    id: UUID
    chat_id: UUID
    message_id: UUID | None
    title: str
    content: str
    citations: list[dict[str, Any]]  # snapshot: survives reindex / re-ask
    created_at: datetime


class NoteRepository(Protocol):
    async def create(
        self,
        chat_id: UUID,
        message_id: UUID | None,
        title: str,
        content: str,
        citations: list[dict[str, Any]],
    ) -> Note: ...

    async def for_chat(self, chat_id: UUID) -> list[Note]:
        """Newest first."""
        ...

    async def for_message(self, chat_id: UUID, message_id: UUID) -> Note | None: ...

    async def set_title(self, note_id: UUID, chat_id: UUID, title: str) -> Note | None: ...

    async def delete(self, note_id: UUID, chat_id: UUID) -> bool: ...


def clean_title(title: str | None) -> str:
    return " ".join((title or "").split())[:TITLE_MAX]


def _not_found() -> NotFound:
    return NotFound("note_not_found", "Nota no encontrada")


class SaveNote:
    """Idempotent per answer: saving the same message twice returns the existing note."""

    def __init__(self, chats: ChatRepository, notes: NoteRepository):
        self._chats = chats
        self._notes = notes

    async def execute(self, chat_id: UUID, message_id: UUID, title: str | None) -> Note:
        await require_chat(self._chats, chat_id)
        message = next((m for m in await self._chats.messages(chat_id) if m.id == message_id), None)
        if message is None or message.role != "assistant":
            raise NotFound("message_not_found", "Respuesta no encontrada")
        existing = await self._notes.for_message(chat_id, message_id)
        if existing is not None:
            return existing
        citations = [asdict(c) for c in message.citations or []]
        return await self._notes.create(
            chat_id, message_id, clean_title(title), message.content, citations
        )


class ListNotes:
    def __init__(self, chats: ChatRepository, notes: NoteRepository):
        self._chats = chats
        self._notes = notes

    async def execute(self, chat_id: UUID) -> list[Note]:
        await require_chat(self._chats, chat_id)
        return await self._notes.for_chat(chat_id)


class RenameNote:
    def __init__(self, notes: NoteRepository):
        self._notes = notes

    async def execute(self, chat_id: UUID, note_id: UUID, title: str) -> Note:
        note = await self._notes.set_title(note_id, chat_id, clean_title(title))
        if note is None:
            raise _not_found()
        return note


class DeleteNote:
    def __init__(self, notes: NoteRepository):
        self._notes = notes

    async def execute(self, chat_id: UUID, note_id: UUID) -> None:
        if not await self._notes.delete(note_id, chat_id):
            raise _not_found()
