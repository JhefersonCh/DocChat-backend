from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Protocol

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool


class ConnectionSource(Protocol):
    """Where a repository gets its connection: the pool (a transaction per call) or a bound one."""

    def connection(self) -> AbstractAsyncContextManager[AsyncConnection]: ...


class BoundConnection:
    """Hands out one connection whose transaction is owned by someone else (the UnitOfWork)."""

    def __init__(self, conn: AsyncConnection):
        self._conn = conn

    @asynccontextmanager
    async def connection(self):
        yield self._conn


class PgDatabase:
    """Implements the Database port."""

    def __init__(self, pool: AsyncConnectionPool):
        self._pool = pool

    async def ping(self) -> None:
        async with self._pool.connection() as conn:
            await conn.execute("select 1")
