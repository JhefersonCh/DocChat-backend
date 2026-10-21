"""The Postgres retriever on real pgvector: four rankings each inside its own modality, native
vs text cosines kept apart, the exhaustive scan and the verdict cache (needs the compose db)."""

import asyncio
from uuid import UUID

import psycopg
from psycopg_pool import AsyncConnectionPool

from chat_pdf.adapters.persistence.chunks import PgChunkStore
from chat_pdf.adapters.persistence.migrations import run_migrations
from chat_pdf.domain.models import Chunk

QUERY = [1.0, 0.0, 0.0]


def _chunk(ordinal, content, modality="text", **kw):
    char = (0, len(content)) if modality == "text" else (None, None)
    return Chunk(ordinal, content, 1, 1, *char, modality=modality, **kw)


async def _seed(url) -> tuple[UUID, PgChunkStore, AsyncConnectionPool]:
    async with await psycopg.AsyncConnection.connect(url) as conn:
        await run_migrations(conn)
        await conn.execute(
            "insert into provider_profiles (kind, name, base_url, model) "
            "values ('embedding', 'e', 'http://x', 'm')"
        )
        await conn.execute("insert into collections default values")
        cur = await conn.execute(
            "insert into documents (collection_id, filename, mime, sha256, embedding_profile_id) "
            "select c.id, 'a.pdf', 'application/pdf', 'x', p.id from collections c, "
            "provider_profiles p returning id"
        )
        doc_id = (await cur.fetchone())[0]
        await conn.commit()
    pool = AsyncConnectionPool(url, open=False, min_size=1, max_size=2)
    await pool.open()
    try:
        return doc_id, await _fill(PgChunkStore(pool), doc_id), pool
    except BaseException:
        await pool.close()
        raise


async def _fill(store: PgChunkStore, doc_id: UUID) -> PgChunkStore:
    chunks = [
        _chunk(0, "El informe habla de mascotas"),  # text, vector [1, 0, 0]
        _chunk(
            1, "", "image", asset_key="1.webp", bbox=[0.0, 0.0, 0.5, 0.5]
        ),  # dog, no caption yet
        _chunk(
            2, "", "image", asset_key="2.webp", bbox=[0.0, 0.0, 1.0, 1.0]
        ),  # cat, captioned below
        _chunk(3, "", "audio", t_start=0.0, t_end=20.0),
    ]
    await store.replace(
        doc_id,
        chunks,
        [[1.0, 0, 0], [0.9, 0.1, 0], [0, 1.0, 0], [0.6, 0.8, 0]],  # native vectors
    )
    stored = await store.for_document(doc_id)
    await store.save_caption(stored[2].id, "Un gato sobre el sofá", "gato sofa", [0.8, 0.6, 0])
    return store


def test_rankings_stay_inside_their_modality_and_cosines_stay_apart(empty_db_url):
    async def run():
        doc_id, store, pool = await _seed(empty_db_url)
        try:
            return (
                await store.candidates([doc_id], "mascotas", QUERY, 10),
                await store.candidates([doc_id], "mascotas", QUERY, 10, None, ("audio",)),
                await store.candidates([doc_id], "mascotas", QUERY, 10, None, ("text",)),
            )
        finally:
            await pool.close()

    everything, audio_only, text_only = asyncio.run(run())

    by_ordinal = lambda chunks: [c.ordinal for c in chunks]  # noqa: E731
    assert by_ordinal(everything.by_vector) == [0, 2]  # text + the captioned image (text space)
    assert by_ordinal(everything.by_text) == [0]  # FTS over content
    assert by_ordinal(everything.by_image) == [1, 2]  # native: dog (0.99) before cat (0.0)
    assert by_ordinal(everything.by_audio) == [3]
    dog = everything.by_image[0]
    assert dog.similarity is None and round(dog.native_similarity, 2) == 0.99  # no mixing
    cat = everything.by_image[1]
    assert round(cat.similarity, 2) == 0.8 and round(cat.native_similarity, 2) == 0.0
    assert everything.by_vector[0].native_similarity is None  # plain text has no native cosine
    assert by_ordinal(audio_only.by_image) == [] and by_ordinal(audio_only.by_audio) == [3]
    assert by_ordinal(audio_only.by_vector) == []  # the filter also narrows the text rankings
    assert by_ordinal(text_only.by_vector) == [0]
    assert text_only.by_image == [] and text_only.by_audio == []


def test_media_matches_returns_every_segment_with_both_cosines_and_keyword_flag(empty_db_url):
    async def run():
        doc_id, store, pool = await _seed(empty_db_url)
        try:
            return await store.media_matches([doc_id], "un gato", QUERY, ("image", "frame"))
        finally:
            await pool.close()

    matches = asyncio.run(run())

    assert [m.chunk.ordinal for m in matches] == [1, 2]  # no top-n cut, no audio / text
    assert [m.fts for m in matches] == [False, True]  # "gato" is in the cat's caption only
    assert round(matches[0].chunk.native_similarity, 2) == 0.99
    assert matches[0].chunk.similarity is None  # not captioned: nothing in text space


def test_verdicts_are_cached_per_segment_and_query_and_die_with_the_segment(empty_db_url):
    async def run():
        doc_id, store, pool = await _seed(empty_db_url)
        try:
            dog, cat = (c.id for c in (await store.for_document(doc_id))[1:3])
            await store.save_verdict(dog, "perro", True)
            await store.save_verdict(cat, "perro", False)
            await store.save_verdict(dog, "perro", True)  # idempotent
            await store.save_verdict(cat, "gato", True)
            got = await store.verdicts([dog, cat], "perro")
            other = await store.verdicts([dog, cat], "gato")
            none = await store.verdicts([], "perro")
            await store.replace(doc_id, [_chunk(0, "nuevo")], [[1.0, 0, 0]])  # re-index
            return dog, cat, got, other, none, await store.verdicts([dog, cat], "perro")
        finally:
            await pool.close()

    dog, cat, got, other, none, after = asyncio.run(run())

    assert got == {dog: True, cat: False} and other == {cat: True} and none == {}
    assert after == {}


def test_media_citations_and_search_details_survive_persistence(empty_db_url):
    from uuid import uuid4

    from chat_pdf.adapters.persistence.chats import PgChatRepository
    from chat_pdf.domain.models import Citation, Search

    async def run():
        async with await psycopg.AsyncConnection.connect(empty_db_url) as conn:
            await run_migrations(conn)
        pool = AsyncConnectionPool(empty_db_url, open=False, min_size=1, max_size=2)
        await pool.open()
        try:
            chats = PgChatRepository(pool)
            chat = await chats.create()
            cite = Citation(
                1, uuid4(), "a.pdf", 1, "perro", chunk_id=uuid4(), content="Un perro", kind="image",
                bbox=[0.1, 0.5, 0.4, 0.9], has_thumb=True, source_kind="pdf",
            )  # fmt: skip
            found = [{"n": 1, "kind": "image", "bbox": cite.bbox, "has_thumb": True}]
            search = Search("perro", 1, found, False, "imagen", True, {"checked": 1})
            await chats.add_exchange(chat.id, "¿dónde?", "Aquí [1]", [cite], [search])
            return cite, (await chats.messages(chat.id))[1]
        finally:
            await pool.close()

    cite, message = asyncio.run(run())

    assert message.citations == [cite]
    [search] = message.searches
    assert (search.modality, search.exhaustive, search.rankings) == ("imagen", True, {"checked": 1})
    assert search.found[0]["bbox"] == cite.bbox and search.found[0]["kind"] == "image"
