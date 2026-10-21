"""Stage 2 of image ingest (plan D5, §2.6): describe each image segment with the vision profile.

Remote and fallible, so it never touches the source's `ready` status. It is resumable per
segment (`text_status`) and reports its own progress; the outcome is `enrich_status`:
ready (all described), partial (some failed / cancelled / interrupted), failed (none), or
skipped (no vision profile)."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from uuid import UUID

from chat_pdf.application.documents import document_input
from chat_pdf.application.ingest_media import page_contexts, supports_images
from chat_pdf.application.ports import (
    DocumentRepository,
    FileStorage,
    ProviderFactory,
    ProviderRepository,
    UnitOfWorkFactory,
)
from chat_pdf.domain.captions import Caption, caption_prompt, parse_caption
from chat_pdf.domain.errors import Conflict, InvalidInput, NotFound, ProviderError
from chat_pdf.domain.figures import compose_bbox, tile_boxes
from chat_pdf.domain.models import Chunk, Document

if TYPE_CHECKING:
    from chat_pdf.application.ports import RenderedMedia
    from chat_pdf.application.transcribe import TranscribeWindows

log = logging.getLogger(__name__)

ATTEMPTS = 2  # per segment: an empty answer (reasoning ate the budget) or a blip is retried once
STOP_AFTER_CONSECUTIVE_FAILURES = 3  # a dead / unauthorized provider: stop spending time
CANCELLED = "Descripciones canceladas"
NO_VISION = "No hay un proveedor de visión activo"
NO_STT = "No hay un proveedor de transcripción activo"
INTERRUPTED = "Descripciones interrumpidas"
CONTEXT_PAGE_CHARS = 600


@dataclass
class Outcome:
    described: int = 0
    failed: int = 0
    last_error: str = ""
    cancelled: bool = False
    aborted: bool = False


class RequestEnrichment:
    """Validates and marks the descriptions `pending`; the caller then runs EnrichDocument."""

    def __init__(self, docs: DocumentRepository, providers: ProviderRepository):
        self._docs = docs
        self._providers = providers

    async def execute(self, document_id: UUID) -> None:
        doc = await self._docs.get(document_id)
        if doc is None:
            raise NotFound("document_not_found", "Documento no encontrado")
        if doc.status != "ready" or doc.enrich_status in ("pending", "running"):
            raise Conflict("document_busy", "El documento no está listo o ya se está describiendo")
        if doc.kind == "audio":  # its stage 2 is the transcript, not a description
            if await self._providers.active("stt") is None:
                raise Conflict("no_stt_profile", NO_STT)
        elif doc.kind == "video":  # frames need vision, the audio track needs STT: one is enough
            if await self._providers.active("vision") is None:
                if await self._providers.active("stt") is None:
                    raise Conflict("no_vision_profile", NO_VISION)
        elif await self._providers.active("vision") is None:
            raise Conflict("no_vision_profile", NO_VISION)
        if not await self._docs.mark_enrich_pending(document_id):
            raise Conflict("document_busy", "Las descripciones ya están completas o en curso")


class EnrichDocument:
    def __init__(
        self,
        uow: UnitOfWorkFactory,
        providers: ProviderRepository,
        factory: ProviderFactory,
        storage: FileStorage,
        retry_delay_s: float = 2.0,
        audio: "TranscribeWindows | None" = None,
        crop: "Callable[[bytes, list[float]], RenderedMedia] | None" = None,
    ):
        self._crop = crop  # cuts a region out of a picture (None: no content-aware regions)
        self._audio = audio
        self._uow = uow
        self._providers = providers
        self._factory = factory
        self._storage = storage
        self._retry_delay_s = retry_delay_s

    async def execute(self, document_id: UUID) -> None:
        """Background job: never raises. Runs only for a document marked `pending`."""
        try:
            await self._run(document_id)
        except Exception as exc:
            log.exception("descriptions failed for %s", document_id)
            try:
                async with self._uow() as uow:
                    if await uow.docs.get(document_id) is not None:
                        await uow.docs.set_enrich(document_id, "failed", str(exc)[:500])
            except Exception:
                log.exception("could not mark descriptions of %s as failed", document_id)

    async def _run(self, document_id: UUID) -> None:
        async with self._uow() as uow:
            head = await uow.docs.get(document_id)
        if head is not None and head.enrich_status == "pending" and head.kind in ("audio", "video"):
            if self._audio is None:
                raise RuntimeError("la transcripción de audio no está configurada")
            if head.kind == "audio":
                await self._audio.run(head)
            else:
                await self._video(head)
            return
        async with self._uow() as uow:
            doc = await uow.docs.get(document_id)
            if doc is None or doc.enrich_status != "pending":
                return
            vision = await self._providers.active("vision")
            if vision is None:  # segments stay `pending`: a retry after adding one resumes
                await uow.docs.set_enrich(document_id, "skipped", NO_VISION)
                return
            await uow.docs.set_enrich(document_id, "running")
        outcome, attempted = await self._captions(doc, vision, ("image", "frame"))
        await finish_enrich(self._uow, document_id, outcome, attempted, "descripciones")

    async def _video(self, doc: Document) -> None:
        """Stage 2 of a video: captions of its frames (vision), then transcripts of its audio
        windows (STT). Either provider may be missing: its segments stay `pending` (partial)."""
        assert self._audio is not None
        vision, stt = await self._providers.active("vision"), await self._providers.active("stt")
        async with self._uow() as uow:
            windows = await uow.chunks.count_media(doc.id, ("audio",))
        stt = stt if windows else None
        if vision is None and stt is None:
            async with self._uow() as uow:
                await uow.docs.set_enrich(doc.id, "skipped", NO_VISION)
            return
        async with self._uow() as uow:
            await uow.docs.set_enrich(doc.id, "running")
        total, attempted = Outcome(), 0
        if vision is not None:
            outcome, count = await self._captions(doc, vision, ("frame",))
            total, attempted = _merge(total, outcome), attempted + count
        else:
            total.last_error = NO_VISION
        if stt is not None and not total.cancelled:
            outcome, count = await self._audio.transcribe(doc, stt)
            total, attempted = _merge(total, outcome), attempted + count
        elif not total.cancelled and windows:
            total.last_error = total.last_error or NO_STT
        await finish_enrich(self._uow, doc.id, total, attempted, "descripciones y transcripciones")

    async def _captions(
        self, doc: Document, vision, modalities: tuple[str, ...]
    ) -> tuple[Outcome, int]:
        """Describe the segments of those modalities still `pending`; returns the outcome and how
        many were attempted. Progress ("Describiendo fotogramas 3/40") counts only them."""
        document_id = doc.id
        async with self._uow() as uow:
            await uow.chunks.requeue_failed_captions(document_id, modalities)
            pending = await uow.chunks.pending_captions(document_id, modalities)
            total = await uow.chunks.count_media(document_id, modalities)
            everything = await uow.chunks.for_document(document_id)
            contexts = _page_contexts(everything)
            text_contexts = page_contexts(everything)
        embed_profile = await self._providers.get(doc.embedding_profile_id)
        if embed_profile is None:
            raise RuntimeError("el perfil de embeddings ya no existe")
        describer, embedder = self._factory.vision(vision), self._factory.embedder(embed_profile)
        regions = _Regions(
            self, doc, embed_profile, embedder, await self._existing_grids(document_id)
        )
        seen: dict[str, Caption] = {}  # media_sha -> caption: a repeated picture is described once

        outcome = Outcome()
        consecutive = 0
        already = total - len(pending)  # described in an earlier run: progress resumes from there
        for i, chunk in enumerate(pending):
            async with self._uow() as uow:
                if await uow.docs.set_progress(document_id, already + i, total, "describing"):
                    outcome.cancelled = True
                    break
            try:
                ask_regions = self._crop is not None and _is_whole_picture(chunk)
                caption = seen.get(chunk.media_sha or "") if ask_regions else None
                if caption is None:
                    caption = await self._describe(
                        describer, document_id, chunk, contexts.get(chunk.page_start), ask_regions
                    )
                    if ask_regions and chunk.media_sha:
                        seen[chunk.media_sha] = caption
                text = document_input(embed_profile, _with_text(chunk, caption.index_text))
                [vector] = await embedder.embed([text])
            except (ProviderError, InvalidInput, OSError, ValueError) as exc:
                outcome.failed += 1
                outcome.last_error = str(exc)
                consecutive += 1
                log.warning("segment %s of %s not described: %s", chunk.ordinal, document_id, exc)
                async with self._uow() as uow:
                    await uow.chunks.fail_caption(chunk.id)
                if consecutive >= STOP_AFTER_CONSECUTIVE_FAILURES:
                    outcome.aborted = True
                    break
                continue
            consecutive = 0
            async with self._uow() as uow:
                await uow.chunks.save_caption(chunk.id, caption.content, caption.index_text, vector)
            outcome.described += 1
            if self._crop is not None and _is_whole_picture(chunk):
                await regions.add(chunk, caption, text_contexts)
        return outcome, len(pending)

    async def _existing_grids(self, document_id: UUID) -> set[UUID]:
        async with self._uow() as uow:
            chunks = await uow.chunks.for_document(document_id)
        return {c.parent_id for c in chunks if c.origin == "tile" and c.parent_id}

    async def _describe(
        self, describer, document_id: UUID, chunk: Chunk, context: str | None, regions: bool = False
    ):
        """The parsed caption; an empty / failed answer is retried, then ProviderError."""
        if not chunk.asset_key:
            raise ValueError("el segmento no tiene imagen")
        path = self._storage.asset_path(document_id, "assets", chunk.asset_key)
        image = await asyncio.to_thread(path.read_bytes)
        error: ProviderError | None = None
        for attempt in range(ATTEMPTS):
            if attempt:
                await asyncio.sleep(self._retry_delay_s)
            try:
                caption = parse_caption(
                    await describer.describe(image, caption_prompt(context, regions))
                )
            except ProviderError as exc:  # includes EmptyCompletion
                error = exc
                continue
            if caption is not None:
                return caption
            error = ProviderError("el modelo de visión no describió la imagen")
        assert error is not None
        raise error


async def finish_enrich(
    uow_factory: UnitOfWorkFactory,
    document_id: UUID,
    o: Outcome,
    attempted: int,
    noun: str,
    cancelled: str = CANCELLED,
) -> None:
    """Close stage 2: ready (all done), partial (some failed / cancelled / left), failed (none)."""
    async with uow_factory() as uow:
        doc = await uow.docs.get(document_id)
        if doc is None:
            return
        left = await uow.chunks.pending_captions(document_id)  # not reached / not retried
        chunks = await uow.chunks.for_document(document_id)
        # Segments described by an earlier run count too: a retry that only re-fails one frame
        # must not turn a source with a finished transcript into "failed".
        already = sum(1 for c in chunks if c.modality != "text" and c.text_status == "ready")
        failed = o.failed
        if o.cancelled:
            status, error = "partial", cancelled
        elif o.aborted:
            status, error = ("failed" if o.described == 0 else "partial"), o.last_error
        elif failed and o.described + already == 0 and not left:
            status, error = "failed", o.last_error
        elif failed:
            status, error = "partial", f"{failed} de {attempted} {noun} fallaron"
            error += f": {o.last_error}" if o.last_error else ""
        elif left:  # nothing failed: a provider is missing or the run stopped early
            status, error = "partial", o.last_error or f"Quedan {len(left)} segmentos pendientes"
        else:
            status, error = "ready", None
        if status in ("partial", "failed") and doc.kind == "video":
            error = _video_parts(chunks) + (f". {error}" if error else "")
        await uow.docs.set_enrich(document_id, status, error and error[:500])


def _video_parts(chunks: list[Chunk]) -> str:
    """ "Audio listo · fotogramas fallidos": the state of each part of a video apart."""
    words = {
        "ready": ("listo", "listos"),
        "failed": ("fallido", "fallidos"),
        "partial": ("parcial", "parciales"),
        "pending": ("pendiente", "pendientes"),
    }
    parts = []
    for modality, name, plural in (("audio", "audio", False), ("frame", "fotogramas", True)):
        items = [c for c in chunks if c.modality == modality]
        if not items:
            continue
        ready = sum(1 for c in items if c.text_status == "ready")
        if ready == len(items):
            state = "ready"
        elif ready:
            state = "partial"
        elif any(c.text_status == "failed" for c in items):
            state = "failed"
        else:
            state = "pending"
        parts.append(f"{name} {words[state][plural]}")
    text = " · ".join(parts)
    return text[:1].upper() + text[1:]


def _merge(a: Outcome, b: Outcome) -> Outcome:
    return Outcome(
        a.described + b.described,
        a.failed + b.failed,
        b.last_error or a.last_error,
        a.cancelled or b.cancelled,
        a.aborted or b.aborted,
    )


def _with_text(chunk: Chunk, index_text: str) -> Chunk:
    return replace(chunk, index_text=index_text, content=index_text)


def _page_contexts(chunks: list[Chunk]) -> dict[int | None, str]:
    """Text of each page (start), to tell the model what a figure sits next to."""
    pages: dict[int | None, list[str]] = {}
    for c in chunks:
        if c.modality == "text" and c.page_start is not None:
            pages.setdefault(c.page_start, []).append(c.content)
    return {n: " ".join(parts)[:CONTEXT_PAGE_CHARS] for n, parts in pages.items()}


def _is_whole_picture(chunk: Chunk) -> bool:
    """A picture / figure / page segment: its content-aware regions are worth asking for."""
    return chunk.modality == "image" and chunk.origin not in ("tile", "region")


class _Regions:
    """Children of a described picture: a segment per salient object (content-aware regions from
    the same vision call) or, when the model gave none, a 3x3 grid. Each child is cropped from the
    parent's asset, embedded natively WITH the page context and its own description (text space).

    Decision: regions REPLACE the stage-1 grid of that picture (they are strictly more
    informative, and keeping both duplicates hits); the grid stays when no region came."""

    def __init__(self, job: "EnrichDocument", doc: Document, profile, embedder, grids: set[UUID]):
        self._job, self._doc, self._profile = job, doc, profile
        self._text, self._grids = embedder, grids
        self._native = (
            job._factory.multimodal_embedder(profile) if supports_images(profile) else None
        )

    async def add(self, parent: Chunk, caption: Caption, contexts: dict[int | None, str]) -> None:
        if self._native is None or parent.id is None or not parent.asset_key:
            return
        if caption.regions:
            items = [(r.bbox, r.label, r.text) for r in caption.regions]
        elif caption.needs_grid and parent.id not in self._grids:
            items = [(box, None, "") for box in tile_boxes()]
        else:
            return
        path = self._job._storage.asset_path(self._doc.id, "assets", parent.asset_key)
        try:
            image = await asyncio.to_thread(path.read_bytes)
        except OSError:
            return
        page_context = contexts.get(parent.page_start, "")
        crop, built = self._job._crop, []
        assert crop is not None
        for rel, label, text in items:
            try:
                piece = await asyncio.to_thread(crop, image, rel)
                context = " | ".join(p for p in (label, page_context) if p)
                vector = await self._native.embed_image(piece.embed, context or None)
            except InvalidInput as exc:  # expected: a decorative speck, not an error
                log.debug("region of segment %s skipped: %s", parent.id, exc.code)
                continue
            except (ProviderError, OSError, ValueError) as exc:
                log.warning("region of segment %s skipped: %r", parent.id, exc)
                continue
            built.append((rel, label, text, piece, vector))
        if not built:
            return
        if caption.regions:  # regions replace the grid of this picture
            async with self._job._uow() as uow:
                await uow.chunks.delete_children(parent.id, "tile")
        texts = [b[2] for b in built if b[1]]
        try:
            text_vectors = iter(
                await self._text.embed(
                    [document_input(self._profile, _with_text(parent, t)) for t in texts]
                )
                if texts
                else []
            )
        except (ProviderError, InvalidInput):
            text_vectors = iter([None] * len(texts))
        async with self._job._uow() as uow:
            ordinal = await uow.chunks.next_ordinal(self._doc.id)
            for rel, label, text, piece, vector in built:
                name = f"{ordinal}.webp"
                await self._job._storage.save_asset(self._doc.id, "assets", name, piece.asset)
                await self._job._storage.save_asset(self._doc.id, "thumbs", name, piece.thumb)
                described = bool(label)
                text_vector = next(text_vectors) if described else None
                child = Chunk(
                    ordinal=ordinal,
                    content=text if described else "",
                    index_text=text if described else None,
                    page_start=parent.page_start,
                    page_end=parent.page_end,
                    char_start=None,
                    char_end=None,
                    modality="image",
                    origin="region" if described else "tile",
                    bbox=compose_bbox(parent.bbox, rel) if parent.bbox else rel,
                    asset_key=name,
                    media_sha=piece.sha,
                    text_status="ready" if described and text_vector else "skipped",
                    parent_id=parent.id,
                    region_label=label,
                )
                await uow.chunks.add_segment(self._doc.id, child, vector, text_vector)
                ordinal += 1
