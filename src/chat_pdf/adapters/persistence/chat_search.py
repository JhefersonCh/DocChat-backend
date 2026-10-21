"""Keyword matching in SQL: `search_fold` (lower + unaccent, migration 018) LIKE every pattern.
Implements the ChatSearchRepository port. Read-only."""

from typing import Any

from psycopg.rows import dict_row

from chat_pdf.adapters.persistence.connection import ConnectionSource

_PROJECT = (
    "case when k.kind = 'project' then k.id end as project_id,"
    " case when k.kind = 'project' then k.name end as project_name"
)

MESSAGES_SQL = f"""
select m.id, m.chat_id, c.title as chat_title, {_PROJECT}, m.role, m.created_at, m.content
from messages m join chats c on c.id = m.chat_id join collections k on k.id = c.collection_id
where search_fold(m.content) like all (%(p)s::text[])
order by m.created_at desc limit %(n)s"""

CHATS_SQL = f"""
select c.id, c.title, c.created_at, {_PROJECT}
from chats c join collections k on k.id = c.collection_id
where search_fold(c.title) like all (%(p)s::text[])
order by c.created_at desc limit %(n)s"""

PROJECTS_SQL = """
select k.id, k.name, k.created_at from collections k
where k.kind = 'project'
  and search_fold(coalesce(k.name, '') || ' ' || coalesce(k.description, ''))
      like all (%(p)s::text[])
order by k.created_at desc limit %(n)s"""

SOURCES_SQL = """
select d.id, d.filename,
       (select c.id from chats c where c.collection_id = d.collection_id
         order by c.created_at limit 1) as chat_id,
       case when k.kind = 'project' then k.id end as project_id
from documents d join collections k on k.id = d.collection_id
where search_fold(d.filename) like all (%(p)s::text[])
order by d.created_at desc limit %(n)s"""


class PgChatSearchRepository:
    def __init__(self, source: ConnectionSource):
        self._source = source

    async def _run(self, sql: str, patterns: list[str], limit: int) -> list[dict[str, Any]]:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(sql, {"p": patterns, "n": limit})  # type: ignore[arg-type]
            return await cur.fetchall()

    async def messages(self, patterns: list[str], limit: int) -> list[dict[str, Any]]:
        return await self._run(MESSAGES_SQL, patterns, limit)

    async def chats(self, patterns: list[str], limit: int) -> list[dict[str, Any]]:
        return await self._run(CHATS_SQL, patterns, limit)

    async def projects(self, patterns: list[str], limit: int) -> list[dict[str, Any]]:
        return await self._run(PROJECTS_SQL, patterns, limit)

    async def sources(self, patterns: list[str], limit: int) -> list[dict[str, Any]]:
        return await self._run(SOURCES_SQL, patterns, limit)
