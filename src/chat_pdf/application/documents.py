import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from chat_pdf.application.chats import require_chat
from chat_pdf.application.generation import GenerateOverview
from chat_pdf.application.ingest_audio import AudioIngest, supports_audio
from chat_pdf.application.ingest_media import MediaIngest, page_contexts, supports_images
from chat_pdf.application.ingest_video import VideoIngest
from chat_pdf.application.ports import (
    DocumentRepository,
    FileStorage,
    Loader,
    ProviderFactory,
    ProviderRepository,
    UnitOfWorkFactory,
)
from chat_pdf.domain.chunking import chunk_pages
from chat_pdf.domain.cleaning import strip_boilerplate
from chat_pdf.domain.config import RagConfig
from chat_pdf.domain.errors import Conflict, NotFound
from chat_pdf.domain.models import Chunk, Document
from chat_pdf.domain.providers import ProviderProfile
from chat_pdf.domain.rules import SourceLimit, kind_for_mime, resolve_upload_mime
from chat_pdf.domain.sources import chat_title, ordered

if TYPE_CHECKING:
    from chat_pdf.application.enrich import EnrichDocument

log = logging.getLogger(__name__)

INTERRUPTED_INGEST = "Ingesta interrumpida por un reinicio; usá Re-indexar para reintentar."
INGEST_CANCELLED = "Ingesta cancelada"
SCANNED_PDF = "El PDF no tiene texto (escaneado); OCR no disponible"
NO_IMAGE_EMBEDDINGS = (
    "El perfil de embeddings activo no admite imágenes: probá el perfil (Probar) para detectarlo"
)
NO_AUDIO_EMBEDDINGS = (
    "El perfil de embeddings activo no admite audio y no hay un perfil de transcripción: "
    "activá uno de los dos para indexar el audio"
)
NO_AUDIO_INDEX = "No se pudo indexar el audio (embedding falló) y no hay perfil de transcripción"
NO_VIDEO_EMBEDDINGS = (
    "El perfil de embeddings activo no admite imágenes, necesarias para indexar los "
    "fotogramas del video: probá el perfil (Probar) para detectarlo"
)


class IngestCancelled(Exception):
    """Raised between embedding batches when the user asked to cancel the ingest."""


def document_input(profile: ProviderProfile, chunk: Chunk) -> str:
    """What gets embedded: the profile's document prefix with `{title}` = the chunk's section
    ("none" outside any section, as EmbeddingGemma expects) + the normalized chunk text."""
    prefix = (profile.document_prefix or "").replace("{title}", chunk.section or "none")
    return prefix + (chunk.index_text or chunk.content)


class UploadDocument:
    def __init__(self, uow: UnitOfWorkFactory, providers: ProviderRepository, storage: FileStorage):
        self._uow = uow
        self._providers = providers
        self._storage = storage

    async def execute(
        self, chat_id: UUID, filename: str, content_type: str | None, content: AsyncIterator[bytes]
    ) -> UUID:
        """Store the file and add it as a source of the chat (status `pending`); ingestion is
        separate. The first upload of a chat and every later one are the same operation."""
        mime = resolve_upload_mime(filename, content_type)
        document_id = uuid4()
        digest = hashlib.sha256()

        async def hashed() -> AsyncIterator[bytes]:
            async for block in content:
                digest.update(block)
                yield block

        try:
            await self._storage.save(document_id, hashed(), PurePath(filename).suffix)
            replaced = await self._register(
                document_id, chat_id, filename, mime, digest.hexdigest()
            )
        except BaseException:
            await self._storage.delete(document_id)
            raise
        if replaced is not None:
            await self._storage.delete(replaced.id)
        return document_id

    async def _register(
        self, document_id: UUID, chat_id: UUID, filename: str, mime: str, sha256: str
    ) -> Document | None:
        """All DB writes in one transaction. Returns the failed document it replaced, if any:
        a chat whose ONLY source failed takes the next upload as its retry."""
        async with self._uow() as uow:
            await uow.chats.lock(chat_id)  # two concurrent uploads to one chat run one at a time
            await require_chat(uow.chats, chat_id)
            existing = ordered(await uow.docs.list_for_chat(chat_id))
            replaced = None
            if len(existing) == 1 and existing[0].status == "failed":
                replaced = existing.pop()
                await uow.docs.delete(replaced.id)
            SourceLimit.check(existing)
            profile = await self._profile_for(existing)
            await uow.docs.create(
                document_id, chat_id, filename, mime, sha256, profile.id, kind_for_mime(mime)
            )
            names = [d.filename for d in existing] + [filename]
            await uow.chats.set_title(chat_id, chat_title(names))
        return replaced

    async def _profile_for(self, existing: list[Document]) -> ProviderProfile:
        """The sources of a chat share one embedding profile (their vectors are searched with
        one query vector): a new source joins its siblings' profile, else the active one."""
        for doc in existing:
            if doc.status != "failed":
                profile = await self._providers.get(doc.embedding_profile_id)
                if profile is not None:
                    return profile
        profile = await self._providers.active("embedding")
        if profile is None:
            raise Conflict("no_embedding_profile", "No hay un proveedor de embeddings activo")
        return profile


class IngestDocument:
    def __init__(
        self,
        uow: UnitOfWorkFactory,
        providers: ProviderRepository,
        factory: ProviderFactory,
        loaders: Mapping[str, Loader],
        storage: FileStorage,
        config: RagConfig,
        overview: GenerateOverview | None = None,
        media: MediaIngest | None = None,
        enrich: "EnrichDocument | None" = None,
        audio: AudioIngest | None = None,
        video: VideoIngest | None = None,
    ):
        self._audio = audio
        self._video = video
        self._uow = uow
        self._providers = providers
        self._factory = factory
        self._loaders = loaders
        self._storage = storage
        self._cfg = config
        self._overview = overview
        self._media = media
        self._enrich = enrich

    async def execute(self, document_id: UUID) -> None:
        """Background job: never raises; any error ends as status=failed."""
        try:
            await self._ingest(document_id)
        except IngestCancelled:
            async with self._uow() as uow:
                if await uow.docs.get(document_id) is not None:
                    log.info("ingest of %s cancelled", document_id)
                    await uow.docs.set_status(document_id, "failed", error=INGEST_CANCELLED)
            return
        except Exception as exc:
            async with self._uow() as uow:
                if await uow.docs.get(document_id) is None:  # deleted mid-ingest: nothing to fail
                    log.info("document %s was deleted during ingest; stopping", document_id)
                    return
                log.exception("ingest failed for %s", document_id)
                message = (str(exc) or type(exc).__name__)[: self._cfg.max_error_chars]
                await uow.docs.set_status(document_id, "failed", error=message)
            return
        if self._enrich is not None:  # stage 2: descriptions; never raises, never un-readies
            await self._enrich.execute(document_id)
        if self._overview is not None:  # best effort, after `ready`: never changes the status
            async with self._uow() as uow:
                doc = await uow.docs.get(document_id)
            if doc is not None and doc.summary_status == "pending":
                await self._overview.execute(document_id)

    async def _ingest(self, document_id: UUID) -> None:
        async with self._uow() as uow:
            doc = await uow.docs.get(document_id)
            if doc is None:
                return
            await uow.docs.set_status(document_id, "processing")
        profile = await self._providers.get(doc.embedding_profile_id)
        if profile is None:
            raise RuntimeError("el perfil de embeddings ya no existe")

        path = self._storage.path(document_id)
        await self._report(document_id, 0, 0, "extracting")
        if doc.kind == "audio":
            return await self._ingest_audio(doc, profile, path)
        if doc.kind == "video":
            return await self._ingest_video(doc, profile, path)
        loader = self._loaders.get(doc.mime)  # images have none: they are all media
        pages = await asyncio.to_thread(loader.load, path) if loader else []
        pages_with_text = [p for p in strip_boilerplate(pages) if p.text.strip()]
        plan = None
        if self._media is not None and self._media.handles(doc.mime):
            if supports_images(profile):
                plan = await self._media.plan(doc.mime, path, pages)
            elif doc.kind == "image":
                raise RuntimeError(NO_IMAGE_EMBEDDINGS)
            else:
                log.info("embedding profile has no image support: %s indexed as text only", doc.id)
        regions = plan.regions if plan else []
        if not pages_with_text and not regions:
            if doc.mime == "application/pdf":
                raise RuntimeError(SCANNED_PDF)
            raise RuntimeError("el archivo no tiene texto" if loader else "no hay imagen útil")

        chunks = chunk_pages(pages_with_text, self._cfg)
        embedder = self._factory.embedder(profile)
        inputs = [document_input(profile, c) for c in chunks]
        total = len(inputs) + len(regions)  # text chunks, then image regions
        vectors: list[list[float]] = []
        step = max(1, self._cfg.embed_batch)
        for done in range(0, len(inputs), step):
            await self._report(document_id, done, total, "embedding")  # raises IngestCancelled
            vectors += await embedder.embed(inputs[done : done + step])
        await self._report(document_id, len(inputs), total, "embedding")

        if plan is not None and self._media is not None and regions:

            async def progress(done: int) -> None:
                await self._report(document_id, len(inputs) + done, total, "embedding")

            media_chunks, media_vectors = await self._media.segments(
                document_id,
                doc.mime,
                path,
                profile,
                plan,
                len(chunks),
                progress,
                page_contexts(chunks),
            )
            chunks, vectors = [*chunks, *media_chunks], [*vectors, *media_vectors]
            if not chunks:
                raise RuntimeError("no se pudo indexar ninguna imagen del archivo")

        page_count = len(pages) if doc.mime == "application/pdf" else None
        has_media = any(c.modality != "text" for c in chunks)
        # Descriptions (stage 2) need a vision profile; the overview needs a chat profile and
        # something to summarize (an image has nothing until it is described).
        vision = await self._providers.active("vision") if has_media else None
        enrich_status = "pending" if vision else "skipped"
        can_summarize = await self._providers.active("chat") and (pages_with_text or vision)
        async with self._uow() as uow:  # new chunks and `ready` commit together or not at all
            await uow.chunks.replace(document_id, chunks, vectors)  # reindex drops the old ones
            await uow.docs.set_status(
                document_id,
                "ready",
                page_count=page_count,
                summary_status="pending" if can_summarize else "skipped",
                enrich_status=enrich_status,
                width=plan.width if plan else None,
                height=plan.height if plan else None,
            )

    async def _ingest_audio(self, doc: Document, profile: ProviderProfile, path: Path) -> None:
        """Windows + native embeddings + waveform; transcripts (stage 2) need the STT profile."""
        if self._audio is None:
            raise RuntimeError("el audio no está habilitado")

        async def progress(done: int, total: int) -> None:
            await self._report(doc.id, done, total, "embedding")

        result = await self._audio.segments(doc.id, path, profile, progress)
        stt = await self._providers.active("stt")
        if stt is None and not result.embedded:
            raise RuntimeError(NO_AUDIO_INDEX if supports_audio(profile) else NO_AUDIO_EMBEDDINGS)
        chat = await self._providers.active("chat")
        async with self._uow() as uow:
            await uow.chunks.replace(doc.id, result.chunks, result.vectors)
            await uow.docs.set_status(
                doc.id,
                "ready",
                summary_status="pending" if chat and stt else "skipped",
                enrich_status="pending" if stt else "skipped",
                duration_s=result.duration_s,
            )

    async def _ingest_video(self, doc: Document, profile: ProviderProfile, path: Path) -> None:
        """Frames + audio windows with native embeddings; captions / transcripts are stage 2."""
        if self._video is None:
            raise RuntimeError("el video no está habilitado")
        if not supports_images(profile):
            raise RuntimeError(NO_VIDEO_EMBEDDINGS)

        async def progress(done: int, total: int, stage: str) -> None:
            await self._report(doc.id, done, total, stage)

        result = await self._video.segments(doc.id, path, profile, progress)
        vision, stt = await self._providers.active("vision"), await self._providers.active("stt")
        chat = await self._providers.active("chat")
        stage_two = bool(vision or (stt and result.windows))
        async with self._uow() as uow:
            await uow.chunks.replace(doc.id, result.chunks, result.vectors)
            await uow.docs.set_status(
                doc.id,
                "ready",
                summary_status="pending" if chat and stage_two else "skipped",
                enrich_status="pending" if stage_two else "skipped",
                duration_s=result.info.duration_s,
                width=result.info.width,
                height=result.info.height,
            )

    async def _report(
        self, document_id: UUID, done: int, total: int, stage: str | None = None
    ) -> None:
        async with self._uow() as uow:
            if await uow.docs.set_progress(document_id, done, total, stage):
                raise IngestCancelled


class ReindexDocument:
    """Re-chunk and re-embed a stored document (same embedding profile) without re-uploading.

    Marks it `processing` here; the caller then runs IngestDocument in the background, which
    swaps the chunks in one transaction and ends `ready` or `failed`.
    """

    def __init__(self, docs: DocumentRepository, storage: FileStorage):
        self._docs = docs
        self._storage = storage

    async def execute(self, document_id: UUID) -> None:
        if await self._docs.get(document_id) is None:
            raise NotFound("document_not_found", "Documento no encontrado")
        if not self._storage.path(document_id).is_file():
            raise NotFound("file_not_found", "El archivo del documento no está disponible")
        if not await self._docs.mark_processing(document_id):
            raise Conflict("document_busy", "El documento ya se está procesando")


class CancelIngest:
    """Ask a running ingest to stop; it notices between embedding batches and ends `failed`."""

    def __init__(self, docs: DocumentRepository):
        self._docs = docs

    async def execute(self, document_id: UUID) -> None:
        if await self._docs.get(document_id) is None:
            raise NotFound("document_not_found", "Documento no encontrado")
        if not await self._docs.request_cancel(document_id):
            raise Conflict("document_not_busy", "El documento no se está procesando")


class FailInterruptedIngests:
    """Startup: ingestion runs in-process, so a restart leaves `pending`/`processing` orphans."""

    def __init__(self, docs: DocumentRepository):
        self._docs = docs

    async def execute(self) -> None:
        await self._docs.fail_unfinished(INTERRUPTED_INGEST)


class GetDocument:
    def __init__(self, docs: DocumentRepository):
        self._docs = docs

    async def execute(self, document_id: UUID) -> Document:
        doc = await self._docs.get(document_id)
        if doc is None:
            raise NotFound("document_not_found", "Documento no encontrado")
        return doc


@dataclass(frozen=True)
class StoredFile:
    path: Path
    filename: str
    mime: str


class OpenDocumentFile:
    def __init__(self, docs: DocumentRepository, storage: FileStorage):
        self._docs = docs
        self._storage = storage

    async def execute(self, document_id: UUID, variant: str = "playable") -> StoredFile:
        """The original, or for a video whose codec browsers refuse (transcoded at ingest) its
        mp4 copy; `variant="original"` always gives the uploaded file."""
        doc = await self._docs.get(document_id)
        if doc is None:
            raise NotFound("document_not_found", "Documento no encontrado")
        if doc.kind == "video" and variant != "original":
            playable = self._storage.playable_path(document_id)
            if playable.is_file():
                return StoredFile(playable, PurePath(doc.filename).stem + ".mp4", "video/mp4")
        path = self._storage.path(document_id)
        if not path.is_file():
            raise NotFound("file_not_found", "El archivo del documento no está disponible")
        return StoredFile(path, doc.filename, doc.mime)
