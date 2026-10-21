"""Study artifacts (FAQ, quiz, mind map, ...): request -> pending row -> background generation."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from chat_pdf.application.chats import require_chat
from chat_pdf.application.concept_map import Semantics, generate_concept_map
from chat_pdf.application.generation import INVALID_RESPONSE, _require_chat_profile
from chat_pdf.application.ports import (
    ChatRepository,
    ChunkStore,
    DocumentRepository,
    ProviderFactory,
    ProviderRepository,
)
from chat_pdf.application.slides import generate_deck
from chat_pdf.application.sources import readable_chunks, readable_documents
from chat_pdf.domain.artifact_batches import artifact_batches
from chat_pdf.domain.artifact_parsers import parse_artifact
from chat_pdf.domain.artifacts import clean_options
from chat_pdf.domain.errors import Conflict, NotFound, ProviderError

log = logging.getLogger(__name__)

INTERRUPTED_ARTIFACT = "La generación se interrumpió (el servidor se reinició)"
GENERIC_FAILURE = "No se pudo generar el material"


@dataclass(frozen=True)
class Artifact:
    id: UUID
    chat_id: UUID
    kind: str
    status: str  # pending | ready | failed
    options: dict[str, Any]
    data: dict[str, Any] | None
    error: str | None
    created_at: datetime


class ArtifactRepository(Protocol):
    async def create(self, chat_id: UUID, kind: str, options: dict[str, Any]) -> Artifact: ...

    async def get(self, artifact_id: UUID) -> Artifact | None: ...

    async def for_chat(self, chat_id: UUID) -> list[Artifact]:
        """Newest first."""
        ...

    async def finish(
        self, artifact_id: UUID, data: dict[str, Any] | None, error: str | None
    ) -> None:
        """`ready` with `data`, or `failed` with `error`."""
        ...

    async def update_data(self, artifact_id: UUID, data: dict[str, Any]) -> None:
        """Replace the data of a `ready` artifact (a concept map grown by expansion)."""
        ...

    async def delete(self, artifact_id: UUID, chat_id: UUID) -> bool: ...

    async def fail_pending(self, error: str) -> None: ...


class RequestArtifact:
    """Validates, stores a `pending` artifact; the caller then runs GenerateArtifact."""

    def __init__(
        self,
        chats: ChatRepository,
        docs: DocumentRepository,
        providers: ProviderRepository,
        artifacts: ArtifactRepository,
    ):
        self._chats = chats
        self._docs = docs
        self._providers = providers
        self._artifacts = artifacts

    async def execute(self, chat_id: UUID, kind: str, options: dict | None) -> Artifact:
        clean = clean_options(kind, options)
        await require_chat(self._chats, chat_id)
        if not await readable_documents(self._chats, self._docs, chat_id):
            raise Conflict("document_not_ready", "El documento no está listo")
        await _require_chat_profile(self._providers)
        return await self._artifacts.create(chat_id, kind, clean)


class GenerateArtifact:
    """Background job. Never raises: failures are stored on the artifact."""

    def __init__(
        self,
        artifacts: ArtifactRepository,
        docs: DocumentRepository,
        chunks: ChunkStore,
        providers: ProviderRepository,
        factory: ProviderFactory,
        chats: ChatRepository | None = None,
        rag=None,
    ):
        self._rag = rag
        self._artifacts = artifacts
        self._docs = docs
        self._chunks = chunks
        self._providers = providers
        self._factory = factory
        self._chats = chats

    async def execute(self, artifact_id: UUID) -> None:
        try:
            data = await self._generate(artifact_id)
            await self._artifacts.finish(artifact_id, data, None)
        except Exception as exc:
            log.exception("artifact %s failed", artifact_id)
            message = exc.message if isinstance(exc, ProviderError) else GENERIC_FAILURE
            try:
                await self._artifacts.finish(artifact_id, None, message)
            except Exception:
                log.exception("could not mark artifact %s as failed", artifact_id)

    async def _generate(self, artifact_id: UUID) -> dict:
        artifact = await self._artifacts.get(artifact_id)
        if artifact is None:
            raise RuntimeError("artifact not found")
        profile = await _require_chat_profile(self._providers)
        chunks = await readable_chunks(self._chats, self._docs, self._chunks, artifact.chat_id)
        if not chunks:
            raise ProviderError("El documento no tiene fragmentos", "document_empty")
        chat = self._factory.chat(profile)
        if artifact.kind == "slides":
            deck = await generate_deck(
                self._chats, self._docs, chunks, artifact.chat_id, artifact.options, chat
            )
            if deck is None:
                raise ProviderError("El modelo no devolvió contenido válido", INVALID_RESPONSE)
            return deck
        if artifact.kind == "concept_map":
            cmap = await generate_concept_map(
                self._chats,
                self._docs,
                chunks,
                artifact.chat_id,
                artifact.options,
                chat,
                Semantics(self._chunks, self._providers, self._factory, self._rag),
            )
            if cmap is None:
                raise ProviderError("El modelo no devolvió contenido válido", INVALID_RESPONSE)
            return cmap
        answers = [
            (await chat.complete(messages)).content or ""
            for messages in artifact_batches(artifact.kind, artifact.options, chunks)
        ]
        data = parse_artifact(artifact.kind, answers, chunks, artifact.options)
        if data is None:
            raise ProviderError("El modelo no devolvió contenido válido", INVALID_RESPONSE)
        return data


class ListArtifacts:
    def __init__(self, chats: ChatRepository, artifacts: ArtifactRepository):
        self._chats = chats
        self._artifacts = artifacts

    async def execute(self, chat_id: UUID) -> list[Artifact]:
        await require_chat(self._chats, chat_id)
        return await self._artifacts.for_chat(chat_id)


class DeleteArtifact:
    def __init__(self, artifacts: ArtifactRepository):
        self._artifacts = artifacts

    async def execute(self, chat_id: UUID, artifact_id: UUID) -> None:
        if not await self._artifacts.delete(artifact_id, chat_id):
            raise NotFound("artifact_not_found", "Material no encontrado")


class FailInterruptedArtifacts:
    """Startup: generation runs in-process, so a restart leaves `pending` orphans."""

    def __init__(self, artifacts: ArtifactRepository):
        self._artifacts = artifacts

    async def execute(self) -> None:
        await self._artifacts.fail_pending(INTERRUPTED_ARTIFACT)
