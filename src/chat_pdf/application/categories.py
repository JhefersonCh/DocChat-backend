"""Categories of documents: manual (user) and automatic (clustering + one LLM naming call)."""

import logging
from dataclasses import dataclass, field
from uuid import UUID

from chat_pdf.application.ports import ProviderFactory, ProviderRepository
from chat_pdf.application.ports_library import LibraryRepository
from chat_pdf.application.projects import clean_name
from chat_pdf.domain.errors import Conflict, InvalidInput, NotFound
from chat_pdf.domain.library import (
    MIN_DOCS_TO_CLUSTER,
    Category,
    NamingDoc,
    cluster_documents,
    naming_messages,
    parse_names,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoCategorizeResult:
    categories: list[tuple[UUID, str, int]] = field(default_factory=list)  # id, name, documents
    clustered: int = 0  # documents that were clustered
    skipped_incompatible: int = 0  # ready documents embedded by another profile / model
    skipped_without_vector: int = 0  # ready documents with no text vector (nothing to compare)
    named_by_model: bool = False
    note: str | None = None


class CategoryService:
    def __init__(
        self,
        lib: LibraryRepository,
        providers: ProviderRepository,
        factory: ProviderFactory,
    ):
        self._lib = lib
        self._providers = providers
        self._factory = factory

    async def _require(self, category_id: UUID) -> Category:
        category = await self._lib.get_category(category_id)
        if category is None:
            raise NotFound("category_not_found", "Categoría no encontrada")
        return category

    async def list_all(self) -> list[Category]:
        return await self._lib.list_categories()

    async def create(self, name: str, color: str | None) -> Category:
        name = clean_name(name)
        if await self._lib.category_by_name(name):
            raise Conflict("category_exists", "Ya existe una categoría con ese nombre")
        return await self._require(await self._lib.create_category(name, color))

    async def update(self, category_id: UUID, changes: dict) -> Category:
        await self._require(category_id)
        if "name" in changes:
            name = clean_name(changes["name"])
            other = await self._lib.category_by_name(name)
            if other and other.id != category_id:
                raise Conflict("category_exists", "Ya existe una categoría con ese nombre")
            changes = {**changes, "name": name}
        await self._lib.update_category(category_id, changes)
        return await self._require(category_id)

    async def delete(self, category_id: UUID) -> None:
        await self._require(category_id)
        await self._lib.delete_category(category_id)

    async def assign(self, category_id: UUID, document_ids: list[UUID]) -> Category:
        await self._require(category_id)
        if not document_ids:
            raise InvalidInput("no_documents", "Elegí al menos un documento")
        await self._lib.assign(category_id, document_ids, "user")
        return await self._require(category_id)

    async def unassign(self, category_id: UUID, document_ids: list[UUID]) -> Category:
        await self._require(category_id)
        await self._lib.unassign(category_id, document_ids)
        return await self._require(category_id)

    async def auto_categorize(self) -> AutoCategorizeResult:
        """Clusters the ready documents by their vector and names each cluster with ONE call.

        Writes only `origin = 'auto'` categories and assignments (replacing the previous auto
        ones); user categories and assignments are never read as targets nor modified. Documents
        embedded by another profile live in another vector space: each profile group is clustered
        on its own and groups too small to cluster are reported as skipped."""
        docs = await self._lib.library_documents(status="ready")
        await self._lib.refresh_vectors()
        vectors = await self._lib.vectors([d.id for d in docs])
        with_vector = [d for d in docs if d.id in vectors]
        by_profile: dict[UUID, list] = {}
        for d in with_vector:
            by_profile.setdefault(d.embedding_profile_id, []).append(d)
        # Each embedding profile is its own vector space: documents are clustered within their
        # group, never across; a group too small to cluster is reported as skipped.
        clusters: list[list[UUID]] = []
        group = []
        for members in by_profile.values():
            if len(members) >= MIN_DOCS_TO_CLUSTER:
                group += members
                clusters += cluster_documents({d.id: vectors[d.id] for d in members})
        result = AutoCategorizeResult(
            clustered=len(group),
            skipped_incompatible=len(with_vector) - len(group),
            skipped_without_vector=len(docs) - len(with_vector),
        )
        if not group:
            return _with(result, note=f"Hacen falta al menos {MIN_DOCS_TO_CLUSTER} documentos")
        if not clusters:
            return _with(result, note="No se encontraron grupos claros")
        clusters.sort(key=lambda ids: -len(ids))
        by_id = {d.id: d for d in group}
        names, by_model = await self._names(
            [[NamingDoc(by_id[i].filename, by_id[i].summary) for i in ids] for ids in clusters]
        )
        reserved = {c.name for c in await self._lib.list_categories() if c.origin == "user"}
        made = await self._lib.replace_auto_categories(
            list(zip(names, clusters, strict=True)), reserved
        )
        return _with(result, categories=made, named_by_model=by_model)

    async def _names(self, groups: list[list[NamingDoc]]) -> tuple[list[str], bool]:
        """One chat call for every group; any failure falls back to "Grupo N"."""
        profile = await self._providers.active("chat")
        if profile is None:
            return parse_names(None, len(groups)), False
        try:
            reply = await self._factory.chat(profile).complete(naming_messages(groups))
        except Exception:
            log.exception("naming categories failed; using generic names")
            return parse_names(None, len(groups)), False
        names = parse_names(reply.content, len(groups))
        return names, any(not n.startswith("Grupo ") for n in names)


def _with(result: AutoCategorizeResult, **changes) -> AutoCategorizeResult:
    from dataclasses import replace

    return replace(result, **changes)
