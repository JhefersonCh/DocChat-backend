from types import TracebackType
from typing import Self

from psycopg_pool import AsyncConnectionPool

from chat_pdf.adapters.persistence.chats import PgChatRepository
from chat_pdf.adapters.persistence.chunks import PgChunkStore
from chat_pdf.adapters.persistence.connection import BoundConnection
from chat_pdf.adapters.persistence.documents import PgDocumentRepository


class PgUnitOfWork:
    """Implements the UnitOfWork port: one pooled connection, one transaction.

    The pool's connection context commits on clean exit and rolls back on exception.
    """

    def __init__(self, pool: AsyncConnectionPool):
        self._pool = pool

    async def __aenter__(self) -> Self:
        self._connection = self._pool.connection()
        source = BoundConnection(await self._connection.__aenter__())
        self.chats = PgChatRepository(source)
        self.docs = PgDocumentRepository(source)
        self.chunks = PgChunkStore(source)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._connection.__aexit__(exc_type, exc, tb)
