"""Projects: a collection of kind 'project' with many sources and many chats."""

import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import PurePath
from uuid import UUID, uuid4

from chat_pdf.application.ports import FileStorage, ProviderRepository
from chat_pdf.application.ports_library import LibraryRepository
from chat_pdf.domain.errors import Conflict, InvalidInput, NotFound
from chat_pdf.domain.library import Project
from chat_pdf.domain.models import Chat, Document
from chat_pdf.domain.providers import ProviderProfile
from chat_pdf.domain.rules import SourceLimit, kind_for_mime, resolve_upload_mime
from chat_pdf.domain.sources import MAX_SOURCES, CollectionOverview, collection_overview, ordered

MAX_NAME = 80


def clean_name(name: str) -> str:
    name = " ".join((name or "").split())
    if not name:
        raise InvalidInput("invalid_name", "El nombre no puede estar vacío")
    return name[:MAX_NAME]


@dataclass(frozen=True)
class ProjectDetail:
    project: Project
    sources: list[Document] = field(default_factory=list)  # S1..Sn in the order added
    chats: list[Chat] = field(default_factory=list)
    overview: CollectionOverview | None = None


class ProjectService:
    def __init__(self, lib: LibraryRepository, providers: ProviderRepository, storage: FileStorage):
        self._lib = lib
        self._providers = providers
        self._storage = storage

    async def require(self, project_id: UUID) -> Project:
        project = await self._lib.get_project(project_id)
        if project is None:
            raise NotFound("project_not_found", "Proyecto no encontrado")
        return project

    async def create(
        self, name: str, description: str | None, color: str | None, icon: str | None
    ) -> Project:
        project_id = await self._lib.create_project(
            clean_name(name), (description or "").strip() or None, color, icon
        )
        return await self.require(project_id)

    async def list_all(self) -> list[Project]:
        return await self._lib.list_projects()

    async def detail(self, project_id: UUID) -> ProjectDetail:
        project = await self.require(project_id)
        docs = ordered(await self._lib.collection_documents(project_id))
        chats = await self._lib.project_chats(project_id)
        return ProjectDetail(project, docs, chats, collection_overview(docs))

    async def update(self, project_id: UUID, changes: dict) -> Project:
        await self.require(project_id)
        if "name" in changes:
            changes = {**changes, "name": clean_name(changes["name"])}
        await self._lib.update_project(project_id, changes)
        return await self.require(project_id)

    async def delete(self, project_id: UUID) -> None:
        """Deletes the project with its chats and sources (database first, files after)."""
        await self.require(project_id)
        for document_id in await self._lib.delete_project(project_id):
            await self._storage.delete(document_id)

    async def upload(
        self,
        project_id: UUID,
        filename: str,
        content_type: str | None,
        content: AsyncIterator[bytes],
    ) -> UUID:
        """Stores the file and adds it as a source (status `pending`); ingestion is separate."""
        await self.require(project_id)
        mime = resolve_upload_mime(filename, content_type)
        document_id = uuid4()
        digest = hashlib.sha256()

        async def hashed() -> AsyncIterator[bytes]:
            async for block in content:
                digest.update(block)
                yield block

        try:
            await self._storage.save(document_id, hashed(), PurePath(filename).suffix)
            existing = await self._lib.collection_documents(project_id)
            SourceLimit.check(existing)
            profile = await self._profile_for(existing)
            await self._lib.add_document(
                document_id, project_id, filename, mime, digest.hexdigest(), profile.id,
                kind_for_mime(mime),
            )  # fmt: skip
        except BaseException:
            await self._storage.delete(document_id)
            raise
        return document_id

    async def _profile_for(self, existing: list[Document]) -> ProviderProfile:
        """The sources of a project share one embedding profile (chats search them with one
        query vector): a new source joins its siblings', else the active one."""
        for doc in existing:
            if doc.status != "failed":
                profile = await self._providers.get(doc.embedding_profile_id)
                if profile is not None:
                    return profile
        profile = await self._providers.active("embedding")
        if profile is None:
            raise Conflict("no_embedding_profile", "No hay un proveedor de embeddings activo")
        return profile

    async def remove_source(self, project_id: UUID, document_id: UUID) -> None:
        await self.require(project_id)
        if not await self._lib.delete_document(project_id, document_id):
            raise NotFound("source_not_found", "La fuente no es de este proyecto")
        await self._storage.delete(document_id)

    async def create_chat(self, project_id: UUID) -> Chat:
        project = await self.require(project_id)
        return await self._lib.create_chat_in(project_id, project.name)

    async def adopt_chat(self, project_id: UUID, chat_id: UUID) -> ProjectDetail:
        """Moves an existing chat into the project: its sources become the project's (the
        documents move, they are not copied) and the chat then reads the whole project; the
        sources it had muted stay muted."""
        await self.require(project_id)
        located = await self._lib.collection_of_chat(chat_id)
        if located is None:
            raise NotFound("chat_not_found", "Chat no encontrado")
        old_collection, kind = located
        if kind != "chat":
            raise Conflict("chat_in_project", "El chat ya pertenece a un proyecto")
        incoming = await self._lib.collection_documents(old_collection)
        current = await self._lib.collection_documents(project_id)
        profiles = {d.embedding_profile_id for d in [*incoming, *current] if d.status != "failed"}
        if len(profiles) > 1:
            raise Conflict(
                "embedding_profile_mismatch",
                "Las fuentes del chat y del proyecto usan perfiles de embeddings distintos",
            )
        alive = [d for d in [*current, *incoming] if d.status != "failed"]
        if len(alive) > MAX_SOURCES:
            raise Conflict("too_many_sources", "El proyecto superaría el máximo de fuentes")
        await self._lib.adopt_chat(chat_id, old_collection, project_id)
        return await self.detail(project_id)
