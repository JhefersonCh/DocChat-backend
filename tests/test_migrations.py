import asyncio
from pathlib import Path

import psycopg

from chat_pdf.adapters.persistence.migrations import run_migrations

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def test_migrations_apply_once_and_are_idempotent(empty_db_url):
    async def run():
        async with await psycopg.AsyncConnection.connect(empty_db_url) as conn:
            first = await run_migrations(conn)
            second = await run_migrations(conn)
            cur = await conn.execute("select to_regclass('public.chunks') is not null")
            return first, second, (await cur.fetchone())[0]

    first, second, has_chunks = asyncio.run(run())

    expected = sorted(f.name for f in MIGRATIONS_DIR.glob("*.sql"))
    assert first == expected
    assert len(expected) >= 6
    assert second == []
    assert has_chunks


def test_011_is_additive_and_backfills_document_kind(empty_db_url, tmp_path):
    for old in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if old.name < "011":
            (tmp_path / old.name).write_text(old.read_text())

    async def run():
        async with await psycopg.AsyncConnection.connect(empty_db_url) as conn:
            await run_migrations(conn, tmp_path)  # the schema before multimodal
            await conn.execute(
                "insert into provider_profiles (kind, name, base_url, model) "
                "values ('embedding', 'e', 'http://x', 'm')"
            )
            await conn.execute("insert into collections default values")
            for name, mime in (("a.pdf", "application/pdf"), ("b.md", "text/markdown")):
                await conn.execute(
                    "insert into documents (collection_id, filename, mime, sha256, "
                    "embedding_profile_id) select c.id, %s, %s, 'x', p.id "
                    "from collections c, provider_profiles p",
                    (name, mime),
                )
            doc = await (await conn.execute("select id from documents limit 1")).fetchone()
            await conn.execute(
                "insert into chunks (document_id, ordinal, content, char_start, char_end) "
                "values (%s, 0, 'old chunk', 0, 9)",
                (doc[0],),
            )
            await conn.commit()
            await run_migrations(conn, MIGRATIONS_DIR)  # now 011
            kinds = await (await conn.execute("select filename, kind from documents")).fetchall()
            chunk = await (
                await conn.execute("select modality, bbox, asset_key from chunks")
            ).fetchone()
            profile = await (
                await conn.execute("select supports_vision, modalities from provider_profiles")
            ).fetchone()
            await conn.execute(  # media segments have no char offsets
                "insert into chunks (document_id, ordinal, content, modality, origin, bbox) "
                "values (%s, 1, '', 'image', 'whole_image', '{0,0,1,1}')",
                (doc[0],),
            )
            return dict(kinds), chunk, profile

    kinds, chunk, profile = asyncio.run(run())

    assert kinds == {"a.pdf": "pdf", "b.md": "text"}
    assert chunk == ("text", None, None)
    assert profile == (False, ["text"])


def test_012_admits_vision_profiles_and_an_active_vision_setting(empty_db_url):
    async def run():
        async with await psycopg.AsyncConnection.connect(empty_db_url) as conn:
            await run_migrations(conn)
            await conn.execute(
                "insert into provider_profiles (kind, name, base_url, model) "
                "values ('vision', 'v', 'http://x', 'm')"
            )
            await conn.execute(
                "insert into app_settings (id, active_vision_profile_id) "
                "select true, id from provider_profiles"
            )
            row = await (
                await conn.execute("select active_vision_profile_id is not null from app_settings")
            ).fetchone()
            await conn.execute("select enrich_error from documents")
            return row[0]

    assert asyncio.run(run())
