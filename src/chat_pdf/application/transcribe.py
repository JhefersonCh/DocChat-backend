"""Stage 2 of audio ingest (plan §2.4): the transcript of each window with the active STT profile.

Same contract as the image descriptions (application/enrich.py): remote and fallible, so it
never touches the source's `ready` status; resumable per window (`text_status`); the outcome
is `enrich_status` ready / partial / failed / skipped (no STT profile). Progress is reported as
"Transcribiendo audio 3/12" (stage `transcribing`)."""

import asyncio
import logging
from dataclasses import replace
from pathlib import Path

from chat_pdf.application.documents import document_input
from chat_pdf.application.enrich import (
    NO_STT,
    STOP_AFTER_CONSECUTIVE_FAILURES,
    Outcome,
    finish_enrich,
)
from chat_pdf.application.ports import (
    FileStorage,
    ProviderFactory,
    ProviderRepository,
    Transcriber,
    UnitOfWorkFactory,
)
from chat_pdf.application.ports_audio import AudioProbe
from chat_pdf.domain.audio import context_text
from chat_pdf.domain.errors import InvalidInput, ProviderError
from chat_pdf.domain.models import Chunk, Document
from chat_pdf.domain.providers import ProviderProfile

log = logging.getLogger(__name__)

ATTEMPTS = 2  # per window: a blip is retried once
CANCELLED = "Transcripción cancelada"


class TranscribeWindows:
    def __init__(
        self,
        uow: UnitOfWorkFactory,
        providers: ProviderRepository,
        factory: ProviderFactory,
        storage: FileStorage,
        probe: AudioProbe,
        retry_delay_s: float = 2.0,
    ):
        self._uow = uow
        self._providers = providers
        self._factory = factory
        self._storage = storage
        self._probe = probe
        self._retry_delay_s = retry_delay_s

    async def run(self, doc: Document) -> None:
        """Transcribe the windows still `pending` (failed ones are re-queued first)."""
        stt = await self._providers.active("stt")
        if stt is None:  # windows stay `pending`: a retry after adding a profile resumes
            async with self._uow() as uow:
                await uow.docs.set_enrich(doc.id, "skipped", NO_STT)
            return
        async with self._uow() as uow:
            await uow.docs.set_enrich(doc.id, "running", stage="transcribing")
        outcome, attempted = await self.transcribe(doc, stt)
        await finish_enrich(self._uow, doc.id, outcome, attempted, "transcripciones", CANCELLED)

    async def transcribe(self, doc: Document, stt: ProviderProfile) -> tuple[Outcome, int]:
        """The loop: transcribe the `audio` segments still `pending` with `stt`. Returns the
        outcome and how many windows were attempted; it does not close stage 2."""
        async with self._uow() as uow:
            await uow.chunks.requeue_failed_captions(doc.id, ("audio",))
            pending = await uow.chunks.pending_captions(doc.id, ("audio",))
            total = await uow.chunks.count_media(doc.id, ("audio",))
        embed_profile = await self._providers.get(doc.embedding_profile_id)
        if embed_profile is None:
            raise RuntimeError("el perfil de embeddings ya no existe")
        transcriber, embedder = (
            self._factory.transcriber(stt),
            self._factory.embedder(embed_profile),
        )
        path = self._storage.path(doc.id)

        outcome = Outcome()
        consecutive = 0
        already = total - len(pending)  # transcribed in an earlier run: progress resumes there
        for i, chunk in enumerate(pending):
            async with self._uow() as uow:
                if await uow.docs.set_progress(doc.id, already + i, total, "transcribing"):
                    outcome.cancelled = True
                    break
            try:
                text = await self._transcribe(transcriber, path, chunk)
                vector = None
                if text:
                    entry = document_input(embed_profile, _with(chunk, text))
                    [vector] = await embedder.embed([entry])
            except (ProviderError, InvalidInput, OSError, ValueError) as exc:
                outcome.failed += 1
                outcome.last_error = str(exc)
                consecutive += 1
                log.warning("window %s of %s not transcribed: %s", chunk.ordinal, doc.id, exc)
                async with self._uow() as uow:
                    await uow.chunks.fail_caption(chunk.id)
                if consecutive >= STOP_AFTER_CONSECUTIVE_FAILURES:
                    outcome.aborted = True
                    break
                continue
            consecutive = 0
            async with self._uow() as uow:
                if vector is None:  # no speech in the window: nothing to index
                    await uow.chunks.skip_caption(chunk.id)
                else:
                    await uow.chunks.save_caption(chunk.id, text, text, vector)
            outcome.described += 1
        await self._refresh_context(doc)
        return outcome, len(pending)

    async def _refresh_context(self, doc: Document) -> None:
        """Parent windows (~30 s) get the transcripts of their fine children chained in time
        order: what the answer reads (small-to-big); no extra speech-to-text calls."""
        async with self._uow() as uow:
            chunks = await uow.chunks.for_document(doc.id)
            children: dict = {}
            for c in chunks:
                if c.modality == "audio" and c.origin != "context" and c.parent_id is not None:
                    children.setdefault(c.parent_id, []).append(
                        (c.t_start or 0.0, c.t_end or 0.0, c.content)
                    )
            for parent in chunks:
                if parent.origin == "context" and parent.id in children:
                    await uow.chunks.set_context(parent.id, context_text(children[parent.id]))

    async def _transcribe(self, transcriber: Transcriber, path: Path, chunk: Chunk) -> str:
        wav = await self._probe.window_wav(path, chunk.t_start or 0.0, chunk.t_end or 0.0)
        error: ProviderError | None = None
        for attempt in range(ATTEMPTS):
            if attempt:
                await asyncio.sleep(self._retry_delay_s)
            try:
                return " ".join((await transcriber.transcribe(wav)).split())
            except ProviderError as exc:
                error = exc
        assert error is not None
        raise error


def _with(chunk: Chunk, text: str) -> Chunk:
    return replace(chunk, index_text=text, content=text)
