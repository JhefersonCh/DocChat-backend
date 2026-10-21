"""Projects, categories and the library of documents. Implements the LibraryRepository port."""

from typing import Any
from uuid import UUID

from psycopg.rows import class_row, dict_row

from chat_pdf.adapters.persistence.chats import CHAT_COLUMNS
from chat_pdf.adapters.persistence.connection import ConnectionSource
from chat_pdf.adapters.persistence.documents import DOC_COLUMNS
from chat_pdf.domain.library import Category, CategoryRef, LibraryDoc, Project
from chat_pdf.domain.models import Chat, Document

PROJECT_SQL = """
select k.id, k.name, k.created_at, k.description, k.color, k.icon, k.pinned_at,
       (select count(*)::int from documents d where d.collection_id = k.id) as source_count,
       (select count(*)::int from documents d
         where d.collection_id = k.id and d.status = 'ready') as ready_count,
       (select count(*)::int from chats c where c.collection_id = k.id) as chat_count
from collections k where k.kind = 'project'"""

CATEGORY_SQL = """
select c.id, c.name, c.origin, c.created_at, c.color,
       (select count(*)::int from document_categories dc where dc.category_id = c.id) as doc_count
from categories c"""

# A document with where it lives and its categories (a JSON array, built in one round trip).
LIBRARY_SQL = """
select d.id, d.filename, d.kind, d.status, d.mime, d.created_at, d.collection_id,
       d.embedding_profile_id, d.page_count, d.duration_s, d.summary,
       case when k.kind = 'project' then k.id end as project_id,
       case when k.kind = 'project' then k.name end as project_name,
       (select c.id from chats c where c.collection_id = d.collection_id
         order by c.created_at limit 1) as chat_id,
       (select c.title from chats c where c.collection_id = d.collection_id
         order by c.created_at limit 1) as chat_title,
       d.doc_embedding is not null as has_vector,
       coalesce((select jsonb_agg(jsonb_build_object(
                     'id', cat.id, 'name', cat.name, 'origin', dc.origin, 'color', cat.color)
                     order by cat.name)
                 from document_categories dc join categories cat on cat.id = dc.category_id
                 where dc.document_id = d.id), '[]'::jsonb) as categories
from documents d join collections k on k.id = d.collection_id"""

# Mean of the text-space vectors of a document's chunks; recomputed when the chunk count changed
# (re-index, descriptions that arrived later). Only `ready` documents.
REFRESH_VECTORS_SQL = """
with counts as (
    select c.document_id, count(*)::int as n from chunks c
    where c.modality = 'text' or c.text_embedding is not null group by c.document_id
), stale as (
    select d.id, counts.n from documents d join counts on counts.document_id = d.id
    where d.status = 'ready' and d.doc_embedding_n is distinct from counts.n
      and (%(ids)s::uuid[] is null or d.id = any(%(ids)s))
)
update documents d
   set doc_embedding = (select avg(coalesce(c.text_embedding, c.embedding)) from chunks c
                        where c.document_id = d.id
                          and (c.modality = 'text' or c.text_embedding is not null)),
       doc_embedding_n = stale.n
from stale where d.id = stale.id returning d.id"""


COMPAT_SAMPLE = 40
COMPAT_VECTORS_SQL = """
with t as (
    select c.document_id, c.modality, c.ordinal, coalesce(c.text_embedding, c.embedding) as v,
           count(*) over (partition by c.document_id)::int as total,
           ntile(%(n)s) over (partition by c.document_id, c.modality = 'text'
                              order by c.ordinal) as tile
    from chunks c where c.document_id = any(%(ids)s)
      and (c.modality = 'text' or c.text_embedding is not null)
      and (c.modality <> 'frame' or c.parent_id is null)
)
select distinct on (t.document_id, t.modality = 'text', t.tile)
       t.document_id, d.sha256, t.total, t.modality, t.v::text
from t join documents d on d.id = t.document_id
order by t.document_id, t.modality = 'text', t.tile, t.ordinal"""


def _library_doc(row: dict[str, Any]) -> LibraryDoc:
    cats = row.pop("categories")
    return LibraryDoc(
        **row,
        categories=[CategoryRef(UUID(c["id"]), c["name"], c["origin"], c["color"]) for c in cats],
    )


def _vector(values: list[float]) -> str:
    return "[" + ",".join(map(str, values)) + "]"


class PgLibraryRepository:
    def __init__(self, source: ConnectionSource):
        self._source = source

    # --- projects --------------------------------------------------------------------------

    async def create_project(
        self, name: str, description: str | None, color: str | None, icon: str | None
    ) -> UUID:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                """insert into collections (kind, name, description, color, icon)
                   values ('project', %s, %s, %s, %s) returning id""",
                (name, description, color, icon),
            )
            return (await cur.fetchone())[0]

    async def get_project(self, project_id: UUID) -> Project | None:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Project))
            await cur.execute(f"{PROJECT_SQL} and k.id = %s", (project_id,))
            return await cur.fetchone()

    async def list_projects(self) -> list[Project]:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Project))
            await cur.execute(f"{PROJECT_SQL} order by k.created_at desc")
            return await cur.fetchall()

    async def update_project(self, project_id: UUID, changes: dict[str, Any]) -> None:
        fields = [f for f in ("name", "description", "color", "icon") if f in changes]
        sets = [f"{f} = %({f})s" for f in fields]  # whitelisted names
        if "pinned" in changes:
            sets.append("pinned_at = now()" if changes["pinned"] else "pinned_at = null")
        if not sets:
            return
        assignments = ", ".join(sets)
        async with self._source.connection() as conn:
            await conn.execute(
                f"update collections set {assignments} where id = %(id)s and kind = 'project'",
                {**{f: changes[f] for f in fields}, "id": project_id},
            )

    async def delete_project(self, project_id: UUID) -> list[UUID]:
        """Deletes the project's collection (chats, messages, documents and chunks cascade).
        Returns the document ids whose stored files to remove."""
        async with self._source.connection() as conn:
            cur = await conn.execute(
                "select id from documents where collection_id = %s", (project_id,)
            )
            ids = [r[0] for r in await cur.fetchall()]
            await conn.execute(
                "delete from collections where id = %s and kind = 'project'", (project_id,)
            )
            return ids

    async def project_chats(self, project_id: UUID) -> list[Chat]:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Chat))
            await cur.execute(
                f"select {CHAT_COLUMNS} from chats where collection_id = %s "
                "order by created_at desc",
                (project_id,),
            )
            return await cur.fetchall()

    async def create_chat_in(self, collection_id: UUID, title: str = "") -> Chat:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Chat))
            await cur.execute(
                "insert into chats (collection_id, title) values (%s, %s) "
                f"returning {CHAT_COLUMNS}",
                (collection_id, title),
            )
            return await cur.fetchone()

    async def collection_documents(self, collection_id: UUID) -> list[Document]:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Document))
            await cur.execute(
                f"select {DOC_COLUMNS} from documents where collection_id = %s "
                "order by created_at, id",
                (collection_id,),
            )
            return await cur.fetchall()

    async def lock_collection(self, collection_id: UUID) -> None:
        async with self._source.connection() as conn:
            await conn.execute(
                "select 1 from collections where id = %s for update", (collection_id,)
            )

    async def add_document(
        self,
        document_id: UUID,
        collection_id: UUID,
        filename: str,
        mime: str,
        sha256: str,
        profile_id: UUID,
        kind: str,
    ) -> None:
        async with self._source.connection() as conn:
            await conn.execute(
                """insert into documents (id, collection_id, filename, mime, sha256,
                                          embedding_profile_id, kind)
                   values (%s, %s, %s, %s, %s, %s, %s)""",
                (document_id, collection_id, filename, mime, sha256, profile_id, kind),
            )

    async def delete_document(self, collection_id: UUID, document_id: UUID) -> bool:
        """Deletes one source of a collection; chats of the collection stop muting it."""
        async with self._source.connection() as conn:
            cur = await conn.execute(
                "delete from documents where id = %s and collection_id = %s",
                (document_id, collection_id),
            )
            if not cur.rowcount:
                return False
            await conn.execute(
                "update chats set disabled_sources = array_remove(disabled_sources, %s) "
                "where collection_id = %s",
                (document_id, collection_id),
            )
            return True

    async def document_source(self, document_id: UUID) -> dict[str, Any] | None:
        """What a copy of a document needs (the file is copied by the caller)."""
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(
                """select id, filename, mime, sha256, embedding_profile_id, kind, status
                   from documents where id = %s""",
                (document_id,),
            )
            return await cur.fetchone()

    async def document_in(self, collection_id: UUID, document_id: UUID) -> bool:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                "select 1 from documents where id = %s and collection_id = %s",
                (document_id, collection_id),
            )
            return await cur.fetchone() is not None

    async def collection_of_chat(self, chat_id: UUID) -> tuple[UUID, str] | None:
        """(collection id, kind) of a chat."""
        async with self._source.connection() as conn:
            cur = await conn.execute(
                """select c.collection_id, k.kind from chats c
                   join collections k on k.id = c.collection_id where c.id = %s""",
                (chat_id,),
            )
            row = await cur.fetchone()
            return (row[0], row[1]) if row else None

    async def adopt_chat(self, chat_id: UUID, old_collection: UUID, project_id: UUID) -> None:
        """Moves the chat's private collection into the project: its documents change
        collection, the chat points to the project (muted sources are kept), and the emptied
        private collection goes away."""
        async with self._source.connection() as conn:
            await conn.execute(
                "update documents set collection_id = %s where collection_id = %s",
                (project_id, old_collection),
            )
            await conn.execute(
                "update chats set collection_id = %s where id = %s", (project_id, chat_id)
            )
            await conn.execute(
                "delete from collections where id = %s and kind = 'chat'", (old_collection,)
            )

    # --- categories ------------------------------------------------------------------------

    async def list_categories(self) -> list[Category]:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Category))
            await cur.execute(f"{CATEGORY_SQL} order by lower(c.name)")
            return await cur.fetchall()

    async def get_category(self, category_id: UUID) -> Category | None:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Category))
            await cur.execute(f"{CATEGORY_SQL} where c.id = %s", (category_id,))
            return await cur.fetchone()

    async def category_by_name(self, name: str) -> Category | None:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(Category))
            await cur.execute(f"{CATEGORY_SQL} where lower(c.name) = lower(%s)", (name,))
            return await cur.fetchone()

    async def create_category(self, name: str, color: str | None, origin: str = "user") -> UUID:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                "insert into categories (name, color, origin) values (%s, %s, %s) returning id",
                (name, color, origin),
            )
            return (await cur.fetchone())[0]

    async def update_category(self, category_id: UUID, changes: dict[str, Any]) -> None:
        """Rename / recolor. The user touching a category makes it theirs (origin 'user')."""
        fields = [f for f in ("name", "color") if f in changes]
        assignments = ", ".join([*(f"{f} = %({f})s" for f in fields), "origin = 'user'"])
        async with self._source.connection() as conn:
            await conn.execute(
                f"update categories set {assignments} where id = %(id)s",
                {**{f: changes[f] for f in fields}, "id": category_id},
            )

    async def delete_category(self, category_id: UUID) -> None:
        async with self._source.connection() as conn:
            await conn.execute("delete from categories where id = %s", (category_id,))

    async def assign(self, category_id: UUID, document_ids: list[UUID], origin: str) -> int:
        """Adds assignments (existing ones are kept as they are). A user assignment to an auto
        category makes the category the user's."""
        async with self._source.connection() as conn:
            cur = await conn.execute(
                """insert into document_categories (document_id, category_id, origin)
                   select d.id, %s, %s from documents d where d.id = any(%s)
                   on conflict (document_id, category_id) do nothing""",
                (category_id, origin, document_ids),
            )
            if origin == "user":
                await conn.execute(
                    "update categories set origin = 'user' where id = %s", (category_id,)
                )
            return cur.rowcount

    async def unassign(self, category_id: UUID, document_ids: list[UUID]) -> int:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                "delete from document_categories where category_id = %s and document_id = any(%s)",
                (category_id, document_ids),
            )
            return cur.rowcount

    async def replace_auto_categories(
        self, groups: list[tuple[str, list[UUID]]], reserved: set[str]
    ) -> list[tuple[UUID, str, int]]:
        """The automatic categorization's only write. Drops every auto ASSIGNMENT and every auto
        category (an auto category with user assignments is the user's by then, so none is left
        with origin 'auto'), then creates one auto category per group. A name clashing with a
        user category gets a numeric suffix. Never touches `origin = 'user'` rows."""
        async with self._source.connection() as conn:
            await conn.execute("delete from document_categories where origin = 'auto'")
            await conn.execute("delete from categories where origin = 'auto'")
            taken = {n.lower() for n in reserved}
            made: list[tuple[UUID, str, int]] = []
            for name, doc_ids in groups:
                final, i = name, 2
                while final.lower() in taken:
                    final, i = f"{name} {i}", i + 1
                taken.add(final.lower())
                cur = await conn.execute(
                    "insert into categories (name, origin) values (%s, 'auto') returning id",
                    (final,),
                )
                category_id = (await cur.fetchone())[0]
                await conn.cursor().executemany(
                    """insert into document_categories (document_id, category_id, origin)
                       values (%s, %s, 'auto') on conflict do nothing""",
                    [(d, category_id) for d in doc_ids],
                )
                made.append((category_id, final, len(doc_ids)))
            return made

    # --- library ---------------------------------------------------------------------------

    async def library_documents(
        self,
        kind: str | None = None,
        category_id: UUID | None = None,
        project_id: UUID | None = None,
        status: str | None = None,
        ids: list[UUID] | None = None,
        uncategorized: bool = False,
        text: str | None = None,
    ) -> list[LibraryDoc]:
        where = [
            "(%(kind)s::text is null or d.kind = %(kind)s)",
            "(%(status)s::text is null or d.status = %(status)s)",
            "(%(project)s::uuid is null or d.collection_id = %(project)s)",
            "(%(ids)s::uuid[] is null or d.id = any(%(ids)s))",
            """(%(category)s::uuid is null or exists (
                 select 1 from document_categories x
                 where x.document_id = d.id and x.category_id = %(category)s))""",
            "(not %(uncat)s or not exists (select 1 from document_categories y "
            "where y.document_id = d.id))",
            "(%(text)s::text is null or d.filename ilike '%%' || %(text)s || '%%')",
        ]
        params = {
            "kind": kind, "status": status, "project": project_id, "ids": ids,
            "category": category_id, "uncat": uncategorized, "text": text,
        }  # fmt: skip
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(
                f"{LIBRARY_SQL} where {' and '.join(where)} order by d.created_at desc, d.id",
                params,
            )
            return [_library_doc(r) for r in await cur.fetchall()]

    async def refresh_vectors(self, ids: list[UUID] | None = None) -> list[UUID]:
        """Computes (or recomputes when stale) the document vector of the ready documents."""
        async with self._source.connection() as conn:
            cur = await conn.execute(REFRESH_VECTORS_SQL, {"ids": ids})
            return [r[0] for r in await cur.fetchall()]

    async def compat_vectors(
        self, ids: list[UUID]
    ) -> dict[UUID, tuple[str, int, list[tuple[str, list[float]]]]]:
        """Text-space chunk vectors (text chunks and the captions / transcripts of media, never a
        native image / audio vector), at most COMPAT_SAMPLE spread over the document per kind."""
        async with self._source.connection() as conn:
            cur = await conn.execute(COMPAT_VECTORS_SQL, {"ids": ids, "n": COMPAT_SAMPLE})
            out: dict[UUID, tuple[str, int, list[tuple[str, list[float]]]]] = {}
            for doc_id, sha, total, modality, vec in await cur.fetchall():
                entry = out.setdefault(doc_id, (sha, total, []))
                entry[2].append((modality, [float(x) for x in vec.strip("[]").split(",")]))
            return out

    async def vectors(self, ids: list[UUID]) -> dict[UUID, list[float]]:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                "select id, doc_embedding::text from documents "
                "where id = any(%s) and doc_embedding is not null",
                (ids,),
            )
            return {
                r[0]: [float(x) for x in r[1].strip("[]").split(",")] for r in await cur.fetchall()
            }

    async def vector_state(self, ids: list[UUID]) -> dict[UUID, tuple[int | None, list[float]]]:
        """Document vector and the chunk count it was computed from, per document."""
        async with self._source.connection() as conn:
            cur = await conn.execute(
                "select id, doc_embedding_n, doc_embedding::text from documents "
                "where id = any(%s) and doc_embedding is not null",
                (ids,),
            )
            return {
                r[0]: (r[1], [float(x) for x in r[2].strip("[]").split(",")])
                for r in await cur.fetchall()
            }

    async def document_scores(self, ids: list[UUID], vector: list[float]) -> dict[UUID, float]:
        """Cosine of the query vector to each document's vector (documents of the profile the
        query was embedded with; a dimension mismatch is skipped)."""
        async with self._source.connection() as conn:
            cur = await conn.execute(
                """select id, 1 - (doc_embedding <=> %s::vector) from documents
                   where id = any(%s) and doc_embedding is not null
                     and vector_dims(doc_embedding) = %s""",
                (_vector(vector), ids, len(vector)),
            )
            return {r[0]: float(r[1]) for r in await cur.fetchall()}
