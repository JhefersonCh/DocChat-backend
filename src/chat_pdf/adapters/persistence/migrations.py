from pathlib import Path

from psycopg import AsyncConnection

LOCK_ID = 727274
DEFAULT_DIR = Path(__file__).resolve().parents[4] / "migrations"


async def run_migrations(conn: AsyncConnection, directory: Path = DEFAULT_DIR) -> list[str]:
    """Apply pending `*.sql` files in name order, one transaction each. Returns applied names."""
    # Session-level lock held for the whole run (a transaction-level one would be released before
    # the migrations are applied), so concurrent startups apply them one at a time.
    await conn.execute("select pg_advisory_lock(%s)", (LOCK_ID,))
    await conn.commit()  # end the implicit transaction: each block below is a real transaction
    try:
        async with conn.transaction():
            await conn.execute(
                "create table if not exists schema_migrations ("
                "name text primary key, applied_at timestamptz not null default now())"
            )
            cur = await conn.execute("select name from schema_migrations")
            done = {row[0] for row in await cur.fetchall()}

        applied = []
        for path in sorted(directory.glob("*.sql")):
            if path.name in done:
                continue
            async with conn.transaction():
                await conn.execute(path.read_text())  # type: ignore[arg-type]
                await conn.execute("insert into schema_migrations (name) values (%s)", (path.name,))
            applied.append(path.name)
        return applied
    finally:
        await conn.execute("select pg_advisory_unlock(%s)", (LOCK_ID,))
