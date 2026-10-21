import json
from typing import Any
from uuid import UUID

from psycopg.rows import class_row
from psycopg.types.json import Jsonb

from chat_pdf.adapters.persistence.connection import ConnectionSource
from chat_pdf.application.notes import Note

COLUMNS = "id, chat_id, message_id, title, content, citations, created_at"


class PgNoteRepository:
    """notes table. Implements the NoteRepository port."""

    def __init__(self, source: ConnectionSource):
        self._source = source

    async def create(
        self,
        chat_id: UUID,
        message_id: UUID | None,
        title: str,
        content: str,
        citations: list[dict[str, Any]],
    ) -> Note:
        snapshot = Jsonb(citations, dumps=lambda obj: json.dumps(obj, default=str))
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Note))
            await cur.execute(
                f"insert into notes (chat_id, message_id, title, content, citations) "
                f"values (%s, %s, %s, %s, %s) returning {COLUMNS}",
                (chat_id, message_id, title, content, snapshot),
            )
            return await cur.fetchone()

    async def for_chat(self, chat_id: UUID) -> list[Note]:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Note))
            await cur.execute(
                f"select {COLUMNS} from notes where chat_id = %s order by created_at desc",
                (chat_id,),
            )
            return await cur.fetchall()

    async def for_message(self, chat_id: UUID, message_id: UUID) -> Note | None:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Note))
            await cur.execute(
                f"select {COLUMNS} from notes where chat_id = %s and message_id = %s limit 1",
                (chat_id, message_id),
            )
            return await cur.fetchone()

    async def set_title(self, note_id: UUID, chat_id: UUID, title: str) -> Note | None:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Note))
            await cur.execute(
                f"update notes set title = %s where id = %s and chat_id = %s returning {COLUMNS}",
                (title, note_id, chat_id),
            )
            return await cur.fetchone()

    async def delete(self, note_id: UUID, chat_id: UUID) -> bool:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                "delete from notes where id = %s and chat_id = %s", (note_id, chat_id)
            )
            return cur.rowcount > 0
