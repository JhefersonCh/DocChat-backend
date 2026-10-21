import json
from uuid import UUID

from psycopg.rows import dict_row

from chat_pdf.adapters.persistence.connection import ConnectionSource
from chat_pdf.application.ports import Candidates
from chat_pdf.domain.find_all import MediaMatch
from chat_pdf.domain.models import Chunk

CHUNK_COLUMNS = (
    "id, ordinal, content, page_start, page_end, char_start, char_end, section, "
    "modality, origin, bbox, t_start, t_end, asset_key, media_sha, text_status, document_id, "
    "parent_id"
)

# ts_rank weights {D, C, B, A}: the section title (A) counts like body text (B). With the default
# A = 1.0 a title word ("negocio" in "modelo de negocio") lifts every chunk of that section.
TS_WEIGHTS = "{0.1, 0.2, 0.4, 0.4}"

# Spoiler-safe reading: chunks without a page (TXT/MD) are never filtered out.
PAGE_FILTER = "(%(max_page)s::int is null or page_start is null or page_start <= %(max_page)s)"
PAGE_FILTER_C = PAGE_FILTER.replace("page_start", "c.page_start")

# Text-space vector of a segment: its own embedding for text chunks, the embedding of the
# description for media (whose native `embedding` lives in another space; plan D6 / phase 3).
TEXT_VECTOR = "coalesce(text_embedding, embedding)"
TEXT_SPACE = "(modality = 'text' or text_embedding is not null)"

# Cosine of the query to a segment in each space it has (NULL when it has none): `similarity` in
# text space (text chunks, descriptions / transcripts), `native_similarity` in the segment's own
# modality (image / audio vectors). They are never compared with each other.
SIMILARITIES = f"""
    case when {TEXT_SPACE} then 1 - ({TEXT_VECTOR} <=> %(vector)s::vector) end as similarity,
    case when modality <> 'text' and embedding is not null
         then 1 - (embedding <=> %(vector)s::vector) end as native_similarity"""

# Segments that are never captioned / transcribed / listed on their own: a video frame or clip under
# a representative frame inherits its caption; an audio "context" parent only concatenates its
# children's transcripts (granularity pass).
UNCAPTIONED = "(modality <> 'frame' or parent_id is null) and origin is distinct from 'context'"

MODALITY_FILTER = "(%(modalities)s::text[] is null or modality = any(%(modalities)s))"

# One round trip, four rankings, each a top-n WITHIN its modality. The full-text query ORs the
# question's words (plainto_tsquery ANDs them, and a natural-language question rarely contains
# every word of the passage). The native rankings reuse the text query's embedding: the model
# embeds text, images and audio into one space.
CANDIDATES_SQL = f"""
with query as (
    select nullif(replace(plainto_tsquery('es_unaccent', %(query)s)::text, ' & ', ' | '), '')
           ::tsquery as q
), by_vector as (
    select id, row_number() over (order by {TEXT_VECTOR} <=> %(vector)s::vector) as rank
    from chunks where document_id = any(%(document_ids)s) and {TEXT_SPACE} and {PAGE_FILTER}
      and {MODALITY_FILTER}
    order by {TEXT_VECTOR} <=> %(vector)s::vector limit %(n)s
), by_text as (
    select c.id,
           row_number() over (order by ts_rank('{TS_WEIGHTS}', c.tsv, query.q, 1) desc, c.ordinal)
           as rank
    from chunks c, query
    where c.document_id = any(%(document_ids)s) and c.tsv @@ query.q and {PAGE_FILTER_C}
      and {MODALITY_FILTER.replace("modality", "c.modality")}
    order by rank limit %(n)s
), by_image as (
    select id, row_number() over (order by embedding <=> %(vector)s::vector) as rank
    from chunks where %(use_image)s and document_id = any(%(document_ids)s)
      and modality in ('image', 'frame') and embedding is not null and {PAGE_FILTER}
    order by embedding <=> %(vector)s::vector limit %(n)s
), by_audio as (
    select id, row_number() over (order by embedding <=> %(vector)s::vector) as rank
    from chunks where %(use_audio)s and document_id = any(%(document_ids)s)
      and modality = 'audio' and embedding is not null and {PAGE_FILTER}
    order by embedding <=> %(vector)s::vector limit %(n)s
)
select {CHUNK_COLUMNS}, v.rank as vector_rank, t.rank as text_rank, i.rank as image_rank,
       a.rank as audio_rank, {SIMILARITIES}
from chunks
left join by_vector v using (id)
left join by_text t using (id)
left join by_image i using (id)
left join by_audio a using (id)
where v.rank is not null or t.rank is not null or i.rank is not null or a.rank is not null
"""

# Exhaustive search: every media segment of the asked modalities, no top-n cut.
MEDIA_MATCHES_SQL = f"""
with query as (
    select nullif(replace(plainto_tsquery('es_unaccent', %(query)s)::text, ' & ', ' | '), '')
           ::tsquery as q
)
select {CHUNK_COLUMNS}, {SIMILARITIES}, coalesce(tsv @@ query.q, false) as fts
from chunks, query
where document_id = any(%(document_ids)s) and modality = any(%(modalities)s) and {PAGE_FILTER}
  and origin is distinct from 'context'
order by ordinal
"""

NATIVE_IMAGE = {"image", "frame"}


def _vector(values: list[float]) -> str:
    return "[" + ",".join(map(str, values)) + "]"  # pgvector text format


def _only(modalities: tuple[str, ...] | None) -> list[str] | None:
    return list(modalities) if modalities else None


def _ranking(rows: list[dict], column: str) -> list[Chunk]:
    ranked = sorted((r for r in rows if r[column] is not None), key=lambda r: r[column])
    return [_chunk(r) for r in ranked]


def _chunk(row: dict) -> Chunk:
    columns = [*CHUNK_COLUMNS.split(", "), "similarity", "native_similarity"]
    return Chunk(**{k: row[k] for k in columns})


class PgChunkStore:
    """pgvector + Postgres full-text storage. Implements the ChunkStore and Retriever ports."""

    def __init__(self, source: ConnectionSource):
        self._source = source

    async def replace(
        self, document_id: UUID, chunks: list[Chunk], vectors: list[list[float] | None]
    ) -> None:
        async with self._source.connection() as conn, conn.cursor() as cur:
            await cur.execute("delete from chunks where document_id = %s", (document_id,))
            await cur.executemany(
                """insert into chunks (id, document_id, ordinal, content, page_start, page_end,
                                       char_start, char_end, section, index_text, embedding,
                                       modality, origin, bbox, t_start, t_end, asset_key,
                                       media_sha, text_status, parent_id, region_label)
                   values (coalesce(%s::uuid, gen_random_uuid()), %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    (
                        c.id,
                        document_id,
                        c.ordinal,
                        c.content,
                        c.page_start,
                        c.page_end,
                        c.char_start,
                        c.char_end,
                        c.section,
                        c.index_text,
                        None if v is None else _vector(v),
                        c.modality,
                        c.origin,
                        c.bbox,
                        c.t_start,
                        c.t_end,
                        c.asset_key,
                        c.media_sha,
                        c.text_status,
                        c.parent_id,
                        c.region_label,
                    )
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
            links = [
                (document_id, c.ordinal, c.parent_ordinal)
                for c in chunks
                if c.parent_ordinal is not None
            ]
            if links:  # children were inserted with their parent's ordinal: resolve it to the id
                await cur.executemany(
                    """update chunks set parent_id = p.id from chunks p
                       where chunks.document_id = %s and chunks.ordinal = %s
                         and p.document_id = chunks.document_id and p.ordinal = %s""",
                    links,
                )

    async def next_ordinal(self, document_id: UUID) -> int:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                "select coalesce(max(ordinal) + 1, 0) from chunks where document_id = %s",
                (document_id,),
            )
            return (await cur.fetchone())[0]

    async def add_segment(
        self,
        document_id: UUID,
        chunk: Chunk,
        vector: list[float] | None,
        text_vector: list[float] | None = None,
    ) -> None:
        """Append one segment after `replace` (stage 2 children: regions of a described image),
        with its description already indexed when `text_vector` is given."""
        async with self._source.connection() as conn:
            await conn.execute(
                """insert into chunks (document_id, ordinal, content, index_text, page_start,
                                       page_end, embedding, text_embedding, modality, origin,
                                       bbox, asset_key, media_sha, text_status, parent_id,
                                       region_label)
                   values (%s, %s, %s, %s, %s, %s, %s::vector, %s::vector, %s, %s, %s, %s, %s,
                           %s, %s, %s)""",
                (
                    document_id, chunk.ordinal, chunk.content, chunk.index_text,
                    chunk.page_start, chunk.page_end,
                    None if vector is None else _vector(vector),
                    None if text_vector is None else _vector(text_vector),
                    chunk.modality, chunk.origin, chunk.bbox, chunk.asset_key, chunk.media_sha,
                    chunk.text_status, chunk.parent_id, chunk.region_label,
                ),
            )  # fmt: skip

    async def delete_children(self, parent_id: UUID, origin: str) -> list[str]:
        """Drop the children of `origin` (grid tiles replaced by regions); their asset names."""
        async with self._source.connection() as conn:
            cur = await conn.execute(
                "delete from chunks where parent_id = %s and origin = %s returning asset_key",
                (parent_id, origin),
            )
            return [r[0] for r in await cur.fetchall() if r[0]]

    async def for_document(self, document_id: UUID) -> list[Chunk]:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(
                f"select {CHUNK_COLUMNS} from chunks where document_id = %s order by ordinal",
                (document_id,),
            )
            return [Chunk(**r) for r in await cur.fetchall()]

    async def text_vectors(self, document_ids: list[UUID]) -> dict[UUID, list[float]]:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                f"""select id, {TEXT_VECTOR}::text from chunks
                    where document_id = any(%s) and {TEXT_SPACE} and {TEXT_VECTOR} is not null
                      and origin is distinct from 'context'""",
                (document_ids,),
            )
            return {row[0]: json.loads(row[1]) for row in await cur.fetchall()}

    async def pending_captions(
        self, document_id: UUID, modalities: tuple[str, ...] | None = None
    ) -> list[Chunk]:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(
                f"""select {CHUNK_COLUMNS} from chunks
                    where document_id = %s and modality <> 'text' and text_status = 'pending'
                      and {UNCAPTIONED}
                      and (%s::text[] is null or modality = any(%s))
                    order by ordinal""",
                (document_id, _only(modalities), _only(modalities)),
            )
            return [Chunk(**r) for r in await cur.fetchall()]

    async def requeue_failed_captions(
        self, document_id: UUID, modalities: tuple[str, ...] | None = None
    ) -> int:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                "update chunks set text_status = 'pending' "
                "where document_id = %s and modality <> 'text' and text_status = 'failed' "
                "and (%s::text[] is null or modality = any(%s))",
                (document_id, _only(modalities), _only(modalities)),
            )
            return cur.rowcount

    async def save_caption(
        self, chunk_id: UUID, content: str, index_text: str, vector: list[float]
    ) -> None:
        async with self._source.connection() as conn:
            await conn.execute(
                """update chunks set content = %s, index_text = %s,
                       text_embedding = %s::vector, text_status = 'ready' where id = %s""",
                (content, index_text, _vector(vector), chunk_id),
            )
            # The frames / clips under a representative inherit its caption as TEXT (no vision
            # call): shown and quoted, but not text-indexed (index_text '' keeps full-text silent,
            # no text vector), so only their native vectors compete in search.
            await conn.execute(
                """update chunks set content = %s, index_text = '', text_status = 'ready'
                   where parent_id = %s and modality = 'frame'""",
                (content, chunk_id),
            )

    async def set_context(self, chunk_id: UUID, text: str) -> None:
        """Text of a context parent (concatenated child transcripts). Never searchable itself."""
        async with self._source.connection() as conn:
            await conn.execute(
                "update chunks set content = %s, index_text = '', text_status = 'ready' "
                "where id = %s",
                (text, chunk_id),
            )

    async def fail_caption(self, chunk_id: UUID) -> None:
        async with self._source.connection() as conn:
            await conn.execute(
                "update chunks set text_status = 'failed' where id = %s", (chunk_id,)
            )

    async def skip_caption(self, chunk_id: UUID) -> None:
        async with self._source.connection() as conn:
            await conn.execute(
                "update chunks set text_status = 'skipped' where id = %s", (chunk_id,)
            )

    async def count_media(
        self, document_id: UUID, modalities: tuple[str, ...] | None = None
    ) -> int:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                f"select count(*) from chunks where document_id = %s and modality <> 'text' "
                f"and {UNCAPTIONED} and (%s::text[] is null or modality = any(%s))",
                (document_id, _only(modalities), _only(modalities)),
            )
            return (await cur.fetchone())[0]

    async def asset_of(self, chunk_id: UUID) -> tuple[UUID, str | None] | None:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                "select document_id, asset_key from chunks where id = %s", (chunk_id,)
            )
            row = await cur.fetchone()
        return None if row is None else (row[0], row[1])

    async def verdicts(self, chunk_ids: list[UUID], query_key: str) -> dict[UUID, bool]:
        if not chunk_ids:
            return {}
        async with self._source.connection() as conn:
            cur = await conn.execute(
                "select chunk_id, verdict from segment_verdicts "
                "where chunk_id = any(%s) and query_key = %s",
                (chunk_ids, query_key),
            )
            return {row[0]: row[1] for row in await cur.fetchall()}

    async def save_verdict(self, chunk_id: UUID, query_key: str, verdict: bool) -> None:
        async with self._source.connection() as conn:
            await conn.execute(
                """insert into segment_verdicts (chunk_id, query_key, verdict)
                   values (%s, %s, %s)
                   on conflict (chunk_id, query_key) do update set verdict = excluded.verdict""",
                (chunk_id, query_key, verdict),
            )

    async def media_matches(
        self,
        document_ids: list[UUID],
        query: str,
        vector: list[float],
        modalities: tuple[str, ...],
        max_page: int | None = None,
    ) -> list[MediaMatch]:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(
                MEDIA_MATCHES_SQL,
                {
                    "document_ids": document_ids,
                    "query": query,
                    "vector": _vector(vector),
                    "modalities": list(modalities),
                    "max_page": max_page,
                },
            )
            return [MediaMatch(_chunk(r), r["fts"]) for r in await cur.fetchall()]

    async def candidates(
        self,
        document_ids: list[UUID],
        query: str,
        vector: list[float],
        n: int,
        max_page: int | None = None,
        modalities: tuple[str, ...] | None = None,
    ) -> Candidates:
        """Exact cosine scans + GIN full-text match, filtered by documents (no ANN index)."""
        wanted = set(modalities) if modalities else None
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(
                CANDIDATES_SQL,
                {
                    "document_ids": document_ids,
                    "query": query,
                    "vector": _vector(vector),
                    "n": n,
                    "max_page": max_page,
                    "modalities": list(modalities) if modalities else None,
                    "use_image": wanted is None or bool(wanted & NATIVE_IMAGE),
                    "use_audio": wanted is None or "audio" in wanted,
                },
            )
            rows = await cur.fetchall()
        return Candidates(
            _ranking(rows, "vector_rank"),
            _ranking(rows, "text_rank"),
            _ranking(rows, "image_rank"),
            _ranking(rows, "audio_rank"),
        )
