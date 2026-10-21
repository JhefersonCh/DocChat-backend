import json
from typing import Any
from uuid import UUID

from psycopg.rows import class_row
from psycopg.types.json import Jsonb

from chat_pdf.adapters.persistence.connection import ConnectionSource
from chat_pdf.application.artifacts import Artifact

COLUMNS = "id, chat_id, kind, status, options, data, error, created_at"


def _jsonb(value: Any) -> Jsonb:
    return Jsonb(value, dumps=lambda obj: json.dumps(obj, default=str))


class PgArtifactRepository:
    """artifacts table. Implements the ArtifactRepository port."""

    def __init__(self, source: ConnectionSource):
        self._source = source

    async def create(self, chat_id: UUID, kind: str, options: dict[str, Any]) -> Artifact:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Artifact))
            await cur.execute(
                f"insert into artifacts (chat_id, kind, options) values (%s, %s, %s) "
                f"returning {COLUMNS}",
                (chat_id, kind, _jsonb(options)),
            )
            return await cur.fetchone()

    async def get(self, artifact_id: UUID) -> Artifact | None:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Artifact))
            await cur.execute(f"select {COLUMNS} from artifacts where id = %s", (artifact_id,))
            return await cur.fetchone()

    async def for_chat(self, chat_id: UUID) -> list[Artifact]:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Artifact))
            await cur.execute(
                f"select {COLUMNS} from artifacts where chat_id = %s order by created_at desc",
                (chat_id,),
            )
            return await cur.fetchall()

    async def finish(
        self, artifact_id: UUID, data: dict[str, Any] | None, error: str | None
    ) -> None:
        async with self._source.connection() as conn:
            await conn.execute(
                "update artifacts set status = %s, data = %s, error = %s "
                "where id = %s and status = 'pending'",
                (
                    "ready" if error is None else "failed",
                    None if data is None else _jsonb(data),
                    error,
                    artifact_id,
                ),
            )

    async def update_data(self, artifact_id: UUID, data: dict[str, Any]) -> None:
        async with self._source.connection() as conn:
            await conn.execute(
                "update artifacts set data = %s where id = %s and status = 'ready'",
                (_jsonb(data), artifact_id),
            )

    async def delete(self, artifact_id: UUID, chat_id: UUID) -> bool:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                "delete from artifacts where id = %s and chat_id = %s", (artifact_id, chat_id)
            )
            return cur.rowcount > 0

    async def fail_pending(self, error: str) -> None:
        async with self._source.connection() as conn:
            await conn.execute(
                "update artifacts set status = 'failed', error = %s where status = 'pending'",
                (error,),
            )
