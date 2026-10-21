import json
from dataclasses import asdict
from typing import Any
from uuid import UUID

from psycopg.rows import class_row, dict_row
from psycopg.types.json import Jsonb

from chat_pdf.adapters.persistence.connection import ConnectionSource
from chat_pdf.domain.models import Chat, Citation, Message, Search

MESSAGE_COLUMNS = (
    "id, role, content, citations, searches, suggestions, weak_evidence, verification, persona,"
    " created_at"
)
CHAT_COLUMNS = (
    "id, title, created_at, style, custom_instructions, length, persona, disabled_sources, "
    "(select count(*)::int from documents d where d.collection_id = chats.collection_id)"
    " as source_count, "
    "(select k.id from collections k where k.id = chats.collection_id and k.kind = 'project')"
    " as project_id, pinned_at"
)
SETTINGS_FIELDS = ("style", "custom_instructions", "length", "persona", "disabled_sources")


def _jsonb(items: list[Citation] | list[Search]) -> Jsonb:
    """Domain objects -> JSON column (the domain itself knows nothing about JSON)."""
    return Jsonb([asdict(i) for i in items], dumps=lambda obj: json.dumps(obj, default=str))


def _uuid(value: str | None) -> UUID | None:
    return None if value is None else UUID(value)


def _message(row: dict[str, Any]) -> Message:
    citations, searches = row.pop("citations"), row.pop("searches")
    return Message(
        **row,
        citations=None if citations is None else [
            Citation(**{
                **c,
                "document_id": UUID(c["document_id"]),
                "chunk_id": _uuid(c.get("chunk_id")),
            })
            for c in citations
        ],
        searches=None if searches is None else [Search(**s) for s in searches],
    )  # fmt: skip


class PgChatRepository:
    """chats + messages. Implements the ChatRepository port."""

    def __init__(self, source: ConnectionSource):
        self._source = source

    async def create(self) -> Chat:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute("insert into collections default values returning id")
            collection = await cur.fetchone()
            cur = conn.cursor(row_factory=class_row(Chat))
            await cur.execute(
                f"""insert into chats (collection_id) values (%s)
                    returning {CHAT_COLUMNS}""",
                (collection["id"],),
            )
            return await cur.fetchone()

    async def get(self, chat_id: UUID) -> Chat | None:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Chat))
            await cur.execute(f"select {CHAT_COLUMNS} from chats where id = %s", (chat_id,))
            return await cur.fetchone()

    async def delete(self, chat_id: UUID) -> list[UUID]:
        """Delete the chat. A private collection (kind 'chat') goes with it, sources included:
        returns the document ids whose files to remove. A shared one ('project') stays."""
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(
                """select c.collection_id, k.kind,
                          coalesce(array_agg(d.id) filter (where d.id is not null), '{}') as doc_ids
                   from chats c join collections k on k.id = c.collection_id
                        left join documents d on d.collection_id = c.collection_id
                   where c.id = %s group by c.collection_id, k.kind""",
                (chat_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return []
            if row["kind"] != "chat":
                await cur.execute("delete from chats where id = %s", (chat_id,))
                return []
            await cur.execute("delete from collections where id = %s", (row["collection_id"],))
            return row["doc_ids"]

    async def lock(self, chat_id: UUID) -> None:
        async with self._source.connection() as conn:
            await conn.execute("select 1 from chats where id = %s for update", (chat_id,))

    async def set_title(self, chat_id: UUID, title: str) -> None:
        async with self._source.connection() as conn:
            await conn.execute("update chats set title = %s where id = %s", (title, chat_id))

    async def messages(self, chat_id: UUID) -> list[Message]:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(
                f"""select {MESSAGE_COLUMNS} from messages
                    where chat_id = %s order by created_at, id""",
                (chat_id,),
            )
            return [_message(r) for r in await cur.fetchall()]

    async def history(self, chat_id: UUID, limit: int) -> list[Message]:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(
                f"""select {MESSAGE_COLUMNS} from messages where chat_id = %s
                    order by created_at desc, id limit %s""",
                (chat_id, limit),
            )
            return [_message(r) for r in reversed(await cur.fetchall())]

    async def set_suggestions(self, message_id: UUID, questions: list[str]) -> None:
        async with self._source.connection() as conn:
            await conn.execute(
                "update messages set suggestions = %s where id = %s", (Jsonb(questions), message_id)
            )

    async def set_verification(self, message_id: UUID, verification: dict) -> None:
        async with self._source.connection() as conn:
            await conn.execute(
                "update messages set verification = %s where id = %s",
                (Jsonb(verification), message_id),
            )

    async def add_exchange(
        self,
        chat_id: UUID,
        question: str,
        answer: str,
        citations: list[Citation],
        searches: list[Search],
        weak_evidence: bool = False,
        persona: str | None = None,
    ) -> UUID:
        async with self._source.connection() as conn:
            # clock_timestamp(): now() is constant inside a transaction and would tie the rows
            await conn.execute(
                """insert into messages (chat_id, role, content, created_at)
                   values (%s, 'user', %s, clock_timestamp())""",
                (chat_id, question),
            )
            cur = await conn.execute(
                """insert into messages
                       (chat_id, role, content, citations, searches, weak_evidence, persona,
                        created_at)
                   values (%s, 'assistant', %s, %s, %s, %s, %s, clock_timestamp())
                   returning id""",
                (chat_id, answer, _jsonb(citations), _jsonb(searches), weak_evidence, persona),
            )
            return (await cur.fetchone())[0]

    async def update_settings(self, chat_id: UUID, changes: dict[str, Any]) -> Chat:
        fields = [f for f in SETTINGS_FIELDS if f in changes]  # whitelist: names go into the SQL
        casts = {"disabled_sources": "::uuid[]"}  # an empty list has no type of its own
        sets = [f"{f} = %({f})s{casts.get(f, '')}" for f in fields]
        if "pinned" in changes:
            sets.append("pinned_at = now()" if changes["pinned"] else "pinned_at = null")
        assignments = ", ".join(sets) or "title = title"
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Chat))
            await cur.execute(
                f"update chats set {assignments} where id = %(id)s returning {CHAT_COLUMNS}",
                {**{f: changes[f] for f in fields}, "id": chat_id},
            )
            return await cur.fetchone()

    # Last on purpose: a `list` method shadows the builtin in annotations declared after it.
    async def list(self) -> list[Chat]:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Chat))
            await cur.execute(f"select {CHAT_COLUMNS} from chats order by created_at desc")
            return await cur.fetchall()
