"""Projects, categories and the library (global concept search). See docs/frontend-api.md."""

import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request, UploadFile
from pydantic import BaseModel, Field

from chat_pdf.adapters.http.schemas import (
    ChatOut,
    CollectionOverviewOut,
    DocumentOut,
    SourceOut,
    UploadOut,
)
from chat_pdf.application.categories import AutoCategorizeResult
from chat_pdf.application.documents import IngestDocument
from chat_pdf.application.library import SearchResponse, SearchResult
from chat_pdf.application.projects import ProjectDetail
from chat_pdf.bootstrap import LibraryUseCases
from chat_pdf.domain.compatibility import Compatibility
from chat_pdf.domain.library import Category, LibraryDoc, Project
from chat_pdf.domain.sources import assign_aliases

router = APIRouter(prefix="/api/v1")
UPLOAD_READ_BYTES = 1024 * 1024
Kind = Literal["pdf", "text", "image", "audio", "video"]


def library(request: Request) -> LibraryUseCases:
    return request.app.state.use_cases.library


Lib = Annotated[LibraryUseCases, Depends(library)]


def ingest_use_case(request: Request) -> IngestDocument:
    return request.app.state.use_cases.ingest_document


Ingest = Annotated[IngestDocument, Depends(ingest_use_case)]

# --- Models ------------------------------------------------------------------------------------


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    color: str | None = Field(default=None, max_length=32)
    icon: str | None = Field(default=None, max_length=40)


class ProjectPatch(BaseModel):
    """Absent field = untouched; `description: null` clears it."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    color: str | None = Field(default=None, max_length=32)
    icon: str | None = Field(default=None, max_length=40)
    pinned: bool | None = None  # pin / unpin in the sidebar

    def changes(self) -> dict:
        given = self.model_dump(exclude_unset=True)
        return {k: v for k, v in given.items() if k not in ("name", "pinned") or v is not None}


class ProjectOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None = None
    color: str | None = None
    icon: str | None = None
    created_at: datetime
    source_count: int = 0
    ready_count: int = 0
    chat_count: int = 0
    pinned_at: datetime | None = None

    @classmethod
    def of(cls, p: Project) -> "ProjectOut":
        return cls(**vars(p))


class ProjectDetailOut(ProjectOut):
    sources: list[SourceOut] = []  # S1..Sn, the same aliases every chat of the project uses
    chats: list[ChatOut] = []
    overview: CollectionOverviewOut | None = None

    @classmethod
    def of(cls, detail: ProjectDetail) -> "ProjectDetailOut":  # type: ignore[override]
        aliases = assign_aliases(detail.sources)
        return cls(
            **ProjectOut.of(detail.project).model_dump(),
            sources=[
                SourceOut(**DocumentOut.of(d).model_dump(), alias=aliases[d.id], active=True)
                for d in detail.sources
            ],
            chats=[ChatOut.of(c) for c in detail.chats],
            overview=(
                CollectionOverviewOut.model_validate(detail.overview, from_attributes=True)
                if detail.overview
                else None
            ),
        )


class AdoptIn(BaseModel):
    chat_id: uuid.UUID


class CategoryIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    color: str | None = Field(default=None, max_length=32)


class CategoryPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    color: str | None = Field(default=None, max_length=32)

    def changes(self) -> dict:
        given = self.model_dump(exclude_unset=True)
        return {k: v for k, v in given.items() if k != "name" or v is not None}


class CategoryOut(BaseModel):
    id: uuid.UUID
    name: str
    origin: Literal["user", "auto"]
    color: str | None = None
    doc_count: int = 0
    created_at: datetime

    @classmethod
    def of(cls, c: Category) -> "CategoryOut":
        return cls(**vars(c))


class DocumentIds(BaseModel):
    document_ids: list[uuid.UUID] = Field(min_length=1, max_length=500)


class CategoryRefOut(BaseModel):
    id: uuid.UUID
    name: str
    origin: Literal["user", "auto"]  # of the assignment
    color: str | None = None


class LibraryDocOut(BaseModel):
    id: uuid.UUID
    filename: str
    kind: Kind
    status: str
    mime: str
    created_at: datetime
    page_count: int | None = None
    duration_s: float | None = None
    summary: str | None = None
    project_id: uuid.UUID | None = None  # set when the document lives in a project
    project_name: str | None = None
    chat_id: uuid.UUID | None = None  # chat to open (the document's own chat, or a project's first)
    chat_title: str | None = None
    has_vector: bool = False
    categories: list[CategoryRefOut] = []

    @classmethod
    def of(cls, d: LibraryDoc) -> "LibraryDocOut":
        data = {
            k: v for k, v in vars(d).items() if k not in ("collection_id", "embedding_profile_id")
        }
        data["categories"] = [CategoryRefOut(**vars(c)) for c in d.categories]
        return cls(**data)


class DocHitOut(BaseModel):
    chunk_id: uuid.UUID | None
    kind: Literal["text", "image", "audio", "video"]
    page: int | None = None
    page_end: int | None = None
    t_start: float | None = None
    t_end: float | None = None
    bbox: list[float] | None = None
    has_thumb: bool = False  # GET /api/v1/segments/{chunk_id}/thumb
    snippet: str
    score: float | None = None


class SearchResultOut(BaseModel):
    document: LibraryDocOut
    score: float  # [0, 1]: best passage blended with the document vector
    relevant: bool  # clears the embedding profile's evidence threshold
    doc_score: float | None = None
    hits: list[DocHitOut] = []

    @classmethod
    def of(cls, r: SearchResult) -> "SearchResultOut":
        return cls(
            document=LibraryDocOut.of(r.doc), score=r.score, relevant=r.relevant,
            doc_score=r.doc_score, hits=[DocHitOut(**vars(h)) for h in r.hits],
        )  # fmt: skip


class LibrarySearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=20, ge=1, le=50)
    kind: Kind | None = None
    category_id: uuid.UUID | None = None
    project_id: uuid.UUID | None = None
    document_ids: list[uuid.UUID] | None = None


class LibrarySearchOut(BaseModel):
    query: str
    results: list[SearchResultOut]
    searched: int
    skipped_unready: int = 0
    skipped_profiles: int = 0  # documents whose embedding profile did not answer
    notes: list[str] = []

    @classmethod
    def of(cls, r: SearchResponse) -> "LibrarySearchOut":
        return cls(
            query=r.query, results=[SearchResultOut.of(x) for x in r.results], searched=r.searched,
            skipped_unready=r.skipped_unready, skipped_profiles=r.skipped_profiles, notes=r.notes,
        )  # fmt: skip


class AutoCategoryOut(BaseModel):
    id: uuid.UUID
    name: str
    doc_count: int


class AutoCategorizeOut(BaseModel):
    categories: list[AutoCategoryOut]
    clustered: int
    skipped_incompatible: int  # documents embedded by another model: not comparable
    skipped_without_vector: int
    named_by_model: bool
    note: str | None = None

    @classmethod
    def of(cls, r: AutoCategorizeResult) -> "AutoCategorizeOut":
        return cls(
            categories=[AutoCategoryOut(id=i, name=n, doc_count=c) for i, n, c in r.categories],
            clustered=r.clustered, skipped_incompatible=r.skipped_incompatible,
            skipped_without_vector=r.skipped_without_vector, named_by_model=r.named_by_model,
            note=r.note,
        )  # fmt: skip


class BestMatchOut(BaseModel):
    document_id: uuid.UUID
    alias: str
    similarity: float


class SourceCompatOut(BaseModel):
    document_id: uuid.UUID
    alias: str
    filename: str
    status: Literal["ok", "weak", "unrelated", "duplicate", "unknown"]
    best_match: BestMatchOut | None = None
    reason: str | None = (
        None  # one Spanish line (weak / unrelated), the copy note of a duplicate, or why "unknown"
    )


class CompatibilityOut(BaseModel):
    coherent: bool  # false when the sources are not one related group
    sources: list[SourceCompatOut]

    @classmethod
    def of(cls, c: Compatibility) -> "CompatibilityOut":
        return cls.model_validate(c, from_attributes=True)


class CompatibilityIn(BaseModel):
    document_ids: list[uuid.UUID] = Field(min_length=1, max_length=12)


class AskGroupIn(BaseModel):
    document_ids: list[uuid.UUID] = Field(min_length=1, max_length=12)
    mode: Literal["chat", "project"] = "chat"
    name: str | None = Field(default=None, max_length=200)
    project_id: uuid.UUID | None = None  # add the copies to this project instead


class AskGroupOut(BaseModel):
    chat_id: uuid.UUID
    project_id: uuid.UUID | None = None
    document_ids: list[uuid.UUID]  # the new sources (copies), being indexed


# --- Projects ----------------------------------------------------------------------------------


@router.post("/projects", status_code=201, response_model=ProjectOut)
async def create_project(body: ProjectIn, lib: Lib):
    return ProjectOut.of(
        await lib.projects.create(body.name, body.description, body.color, body.icon)
    )


@router.get("/projects", response_model=list[ProjectOut])
async def list_projects(lib: Lib):
    return [ProjectOut.of(p) for p in await lib.projects.list_all()]


@router.get("/projects/{project_id}", response_model=ProjectDetailOut)
async def get_project(project_id: uuid.UUID, lib: Lib):
    return ProjectDetailOut.of(await lib.projects.detail(project_id))


@router.patch("/projects/{project_id}", response_model=ProjectOut)
async def update_project(project_id: uuid.UUID, body: ProjectPatch, lib: Lib):
    return ProjectOut.of(await lib.projects.update(project_id, body.changes()))


@router.delete("/projects/{project_id}", status_code=204)
async def delete_project(project_id: uuid.UUID, lib: Lib):
    """Deletes the project, its chats (with messages) and its sources (files included)."""
    await lib.projects.delete(project_id)


async def _blocks(file: UploadFile) -> AsyncIterator[bytes]:
    while block := await file.read(UPLOAD_READ_BYTES):
        yield block


@router.post("/projects/{project_id}/sources", status_code=202, response_model=UploadOut)
async def upload_project_source(
    project_id: uuid.UUID, file: UploadFile, background: BackgroundTasks, lib: Lib, ingest: Ingest
):
    document_id = await lib.projects.upload(
        project_id, file.filename or "", file.content_type, _blocks(file)
    )
    background.add_task(ingest.execute, document_id)
    return UploadOut(document_id=document_id)


@router.delete("/projects/{project_id}/sources/{document_id}", status_code=204)
async def remove_project_source(project_id: uuid.UUID, document_id: uuid.UUID, lib: Lib):
    await lib.projects.remove_source(project_id, document_id)


@router.post("/projects/{project_id}/chats", status_code=201, response_model=ChatOut)
async def create_project_chat(project_id: uuid.UUID, lib: Lib):
    """A new chat that reads every source of the project (mute some with PATCH /chats/{id})."""
    return ChatOut.of(await lib.projects.create_chat(project_id))


@router.post("/projects/{project_id}/adopt", response_model=ProjectDetailOut)
async def adopt_chat(project_id: uuid.UUID, body: AdoptIn, lib: Lib):
    """Moves a loose chat into the project: its documents become the project's sources."""
    return ProjectDetailOut.of(await lib.projects.adopt_chat(project_id, body.chat_id))


@router.get("/projects/{project_id}/compatibility", response_model=CompatibilityOut)
async def project_compatibility(project_id: uuid.UUID, lib: Lib):
    """Do the project's sources talk about related things? Informative: never blocks."""
    return CompatibilityOut.of(await lib.compatibility.for_project(project_id))


@router.get("/chats/{chat_id}/compatibility", response_model=CompatibilityOut)
async def chat_compatibility(chat_id: uuid.UUID, lib: Lib):
    """The same for the ACTIVE (not muted) sources of a chat; aliases are the chat's S1..Sn."""
    return CompatibilityOut.of(await lib.compatibility.for_chat(chat_id))


@router.post("/library/compatibility", response_model=CompatibilityOut)
async def selection_compatibility(body: CompatibilityIn, lib: Lib):
    """The same for a library selection, before asking it (aliases follow the selection order)."""
    return CompatibilityOut.of(await lib.compatibility.for_documents(body.document_ids))


# --- Categories --------------------------------------------------------------------------------


@router.get("/categories", response_model=list[CategoryOut])
async def list_categories(lib: Lib):
    return [CategoryOut.of(c) for c in await lib.categories.list_all()]


@router.post("/categories", status_code=201, response_model=CategoryOut)
async def create_category(body: CategoryIn, lib: Lib):
    return CategoryOut.of(await lib.categories.create(body.name, body.color))


@router.patch("/categories/{category_id}", response_model=CategoryOut)
async def update_category(category_id: uuid.UUID, body: CategoryPatch, lib: Lib):
    return CategoryOut.of(await lib.categories.update(category_id, body.changes()))


@router.delete("/categories/{category_id}", status_code=204)
async def delete_category(category_id: uuid.UUID, lib: Lib):
    await lib.categories.delete(category_id)


@router.post("/categories/{category_id}/assign", response_model=CategoryOut)
async def assign_documents(category_id: uuid.UUID, body: DocumentIds, lib: Lib):
    """Adds documents to the category (user assignments: automatic runs never touch them)."""
    return CategoryOut.of(await lib.categories.assign(category_id, body.document_ids))


@router.post("/categories/{category_id}/unassign", response_model=CategoryOut)
async def unassign_documents(category_id: uuid.UUID, body: DocumentIds, lib: Lib):
    return CategoryOut.of(await lib.categories.unassign(category_id, body.document_ids))


@router.post("/library/categorize", response_model=AutoCategorizeOut)
async def auto_categorize(lib: Lib):
    """Clusters the documents and names the groups; writes only `origin = 'auto'` rows."""
    return AutoCategorizeOut.of(await lib.categories.auto_categorize())


# --- Library -----------------------------------------------------------------------------------


@router.get("/library/documents", response_model=list[LibraryDocOut])
async def library_documents(
    lib: Lib,
    kind: Annotated[Kind | None, Query()] = None,
    category_id: Annotated[uuid.UUID | None, Query()] = None,
    project_id: Annotated[uuid.UUID | None, Query()] = None,
    status: Annotated[str | None, Query()] = None,
    uncategorized: Annotated[bool, Query()] = False,
    q: Annotated[str | None, Query(max_length=200)] = None,
):
    """Every document of every chat and project (newest first), with where it lives and its
    categories. `q` filters by file name; use POST /library/search for concepts."""
    docs = await lib.library.documents(
        kind=kind, category_id=category_id, project_id=project_id, status=status,
        uncategorized=uncategorized, text=(q or "").strip() or None,
    )  # fmt: skip
    return [LibraryDocOut.of(d) for d in docs]


@router.post("/library/search", response_model=LibrarySearchOut)
async def library_search(body: LibrarySearchIn, lib: Lib):
    return LibrarySearchOut.of(
        await lib.search.execute(
            body.query,
            body.limit,
            body.kind,
            body.category_id,
            body.project_id,
            body.document_ids,
        )  # fmt: skip
    )


@router.post("/library/backfill", response_model=dict[str, int])
async def backfill_vectors(lib: Lib):
    """Computes the document vector of every ready document that lacks a current one."""
    return {"updated": await lib.library.backfill()}


@router.post("/library/ask", status_code=201, response_model=AskGroupOut)
async def ask_documents(body: AskGroupIn, background: BackgroundTasks, lib: Lib, ingest: Ingest):
    """ "Preguntar a estos documentos": copies the selection into a new chat (mode "chat"), a new
    project (mode "project") or an existing one (`project_id`) and opens a chat there. The copies
    are indexed in the background; the chat can be opened right away."""
    result = await lib.ask.execute(body.document_ids, body.mode, body.name, body.project_id)
    for document_id in result.document_ids:
        background.add_task(ingest.execute, document_id)
    return AskGroupOut(
        chat_id=result.chat_id, project_id=result.project_id, document_ids=result.document_ids
    )
