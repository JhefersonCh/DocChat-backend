from uuid import UUID

from psycopg.rows import class_row
from psycopg.types.json import Jsonb

from chat_pdf.adapters.persistence.connection import ConnectionSource
from chat_pdf.domain.models import Document

DOC_COLUMNS = (
    "id, collection_id, filename, mime, status, error, page_count, embedding_profile_id, "
    "created_at, summary, suggested_questions, summary_status, "
    "progress_done, progress_total, cancel_requested, "
    "kind, duration_s, width, height, stage, enrich_status, enrich_error"
)


class PgDocumentRepository:
    """documents table. Implements the DocumentRepository port."""

    def __init__(self, source: ConnectionSource):
        self._source = source

    async def create(
        self,
        document_id: UUID,
        chat_id: UUID,
        filename: str,
        mime: str,
        sha256: str,
        embedding_profile_id: UUID,
        kind: str = "pdf",
    ) -> None:
        async with self._source.connection() as conn:
            await conn.execute(
                """insert into documents (id, collection_id, filename, mime, sha256,
                                          embedding_profile_id, kind)
                   select %s, collection_id, %s, %s, %s, %s, %s from chats where id = %s""",
                (document_id, filename, mime, sha256, embedding_profile_id, kind, chat_id),
            )

    async def get(self, document_id: UUID) -> Document | None:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Document))
            await cur.execute(f"select {DOC_COLUMNS} from documents where id = %s", (document_id,))
            return await cur.fetchone()

    async def list_for_chat(self, chat_id: UUID) -> list[Document]:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Document))
            await cur.execute(
                f"""select {DOC_COLUMNS} from documents
                    where collection_id = (select collection_id from chats where id = %s)
                    order by created_at, id""",
                (chat_id,),
            )
            return await cur.fetchall()

    async def delete(self, document_id: UUID) -> None:
        async with self._source.connection() as conn:
            await conn.execute("delete from documents where id = %s", (document_id,))

    async def mark_processing(self, document_id: UUID) -> bool:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                """update documents set status = 'processing', error = null, summary = null,
                          suggested_questions = '[]', summary_status = 'skipped',
                          progress_done = null, progress_total = null, cancel_requested = false,
                          stage = null, enrich_status = 'skipped', enrich_error = null
                   where id = %s and status in ('ready', 'failed')
                         and enrich_status <> 'running' returning id""",
                (document_id,),
            )
            return await cur.fetchone() is not None

    async def fail_unfinished(self, error: str) -> None:
        async with self._source.connection() as conn:
            await conn.execute(
                "update documents set status = 'failed', error = %s "
                "where status in ('pending', 'processing')",
                (error,),
            )
            await conn.execute(
                "update documents set summary_status = 'failed' where summary_status = 'pending'"
            )
            await conn.execute(  # the segments keep their own text_status: the stage can resume
                "update documents set enrich_status = 'partial', enrich_error = %s, stage = null, "
                "progress_done = null, progress_total = null "
                "where enrich_status in ('pending', 'running')",
                (error,),
            )

    async def set_summary(
        self,
        document_id: UUID,
        status: str,
        summary: str | None = None,
        questions: list[str] | None = None,
    ) -> None:
        async with self._source.connection() as conn:
            await conn.execute(
                """update documents set summary_status = %s, summary = coalesce(%s, summary),
                       suggested_questions = coalesce(%s, suggested_questions) where id = %s""",
                (status, summary, None if questions is None else Jsonb(questions), document_id),
            )

    async def mark_summary_pending(self, document_id: UUID) -> bool:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                """update documents set summary_status = 'pending'
                   where id = %s and status = 'ready' and summary_status <> 'pending'
                   returning id""",
                (document_id,),
            )
            return await cur.fetchone() is not None

    async def set_status(
        self,
        document_id: UUID,
        status: str,
        error: str | None = None,
        page_count: int | None = None,
        summary_status: str | None = None,
        enrich_status: str | None = None,
        width: int | None = None,
        height: int | None = None,
        duration_s: float | None = None,
    ) -> None:
        async with self._source.connection() as conn:
            await conn.execute(
                """update documents set status = %s, error = %s,
                       page_count = coalesce(%s, page_count),
                       summary_status = coalesce(%s, summary_status),
                       enrich_status = coalesce(%s, enrich_status),
                       width = coalesce(%s, width), height = coalesce(%s, height),
                       duration_s = coalesce(%s, duration_s),
                       stage = null, progress_done = null, progress_total = null,
                       cancel_requested = false
                   where id = %s""",
                (
                    status, error, page_count, summary_status, enrich_status, width, height,
                    duration_s, document_id,
                ),
            )  # fmt: skip

    async def set_progress(
        self, document_id: UUID, done: int, total: int, stage: str | None = None
    ) -> bool:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                """update documents set progress_done = %s, progress_total = %s,
                          stage = coalesce(%s, stage)
                   where id = %s returning cancel_requested""",
                (done, total, stage, document_id),
            )
            row = await cur.fetchone()
            return bool(row and row[0])

    async def request_cancel(self, document_id: UUID) -> bool:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                """update documents set cancel_requested = true
                   where id = %s and (status = 'processing' or enrich_status = 'running')
                   returning id""",
                (document_id,),
            )
            return await cur.fetchone() is not None

    async def set_enrich(
        self, document_id: UUID, status: str, error: str | None = None, stage: str = "describing"
    ) -> None:
        running = status == "running"
        async with self._source.connection() as conn:
            await conn.execute(
                """update documents set enrich_status = %s, enrich_error = %s,
                       stage = case when %s then %s end,
                       progress_done = null, progress_total = null, cancel_requested = false
                   where id = %s""",
                (status, error, running, stage, document_id),
            )

    async def mark_enrich_pending(self, document_id: UUID) -> bool:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                """update documents set enrich_status = 'pending', enrich_error = null
                   where id = %s and status = 'ready'
                         and enrich_status in ('partial', 'failed', 'skipped') returning id""",
                (document_id,),
            )
            return await cur.fetchone() is not None
