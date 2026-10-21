"""Stage 1 of audio ingest (plan §2.4): probe, window, embed each window natively and store the
waveform. Local only (no STT needed): the source is searchable by its native vectors before
any transcript exists. A window whose embedding fails stays as a text-only segment."""

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from chat_pdf.application.ports import FileStorage, ProviderFactory
from chat_pdf.application.ports_audio import AudioProbe
from chat_pdf.domain.audio import (
    FINE_MIN_S,
    FINE_OVERLAP_S,
    FINE_SNAP_S,
    FINE_WINDOW_S,
    MAX_AUDIO_S,
    find_silences,
    is_empty,
    plan_parents,
    plan_windows,
    waveform_peaks,
)
from chat_pdf.domain.errors import InvalidInput, ProviderError
from chat_pdf.domain.models import Chunk
from chat_pdf.domain.providers import ProviderProfile

log = logging.getLogger(__name__)

WAVEFORM_ASSET = "waveform.json"
MAX_CONSECUTIVE_EMBED_FAILURES = 3  # the server cannot embed audio: stop trying, keep the text
Progress = Callable[[int, int], Awaitable[None]]  # done, total windows; raises IngestCancelled
TOO_LONG = "El audio dura más de {hours} h; dividilo en partes más cortas"
NO_SOUND = "El audio no tiene sonido"


def supports_audio(profile: ProviderProfile) -> bool:
    """The embedding profile's probe found the server can embed audio."""
    return "audio" in profile.modalities


@dataclass(frozen=True)
class AudioSegments:
    chunks: list[Chunk]
    vectors: list[list[float] | None]  # None: the native embedding failed or is unsupported
    duration_s: float
    fine: int = 0  # retrieval windows (the first `fine` chunks); the rest are context parents

    @property
    def embedded(self) -> int:
        return sum(v is not None for v in self.vectors)


class AudioIngest:
    def __init__(self, probe: AudioProbe, factory: ProviderFactory, storage: FileStorage):
        self._probe = probe
        self._factory = factory
        self._storage = storage

    async def segments(
        self,
        document_id: UUID,
        path: Path,
        profile: ProviderProfile,
        progress: Progress,
        first_ordinal: int = 0,
        keep_assets: bool = False,
    ) -> AudioSegments:
        """Windows of the file's first audio stream. A video passes `first_ordinal` (its frames
        come first) and `keep_assets` (the frames' pictures live in the same folder)."""
        duration = await self._probe.duration(path)
        if duration > MAX_AUDIO_S:
            raise InvalidInput("audio_too_long", TOO_LONG.format(hours=int(MAX_AUDIO_S // 3600)))
        levels = await self._probe.levels(path)
        windows = [
            w
            for w in plan_windows(
                duration,
                find_silences(levels.rms_db, levels.frame_s),
                FINE_WINDOW_S,
                FINE_OVERLAP_S,
                FINE_SNAP_S,
                FINE_MIN_S,
            )
            if not is_empty(w, levels.rms_db, levels.frame_s)
        ]
        if not windows:
            raise InvalidInput("audio_silent", NO_SOUND)
        if not keep_assets:
            await self._storage.clear_assets(document_id)  # a re-index regenerates the waveform
        peaks = waveform_peaks(levels.peaks)
        payload = json.dumps({"peaks": peaks, "duration_s": round(duration, 3)}).encode()
        await self._storage.save_asset(document_id, "assets", WAVEFORM_ASSET, payload)

        parents = plan_parents(windows)
        parent_of = {i: k for k, p in enumerate(parents) for i in p.children}
        embedder = self._factory.multimodal_embedder(profile) if supports_audio(profile) else None
        chunks: list[Chunk] = []
        vectors: list[list[float] | None] = []
        failures = 0
        for ordinal, window in enumerate(windows):
            await progress(ordinal, len(windows))
            vector = None
            if embedder is not None and failures < MAX_CONSECUTIVE_EMBED_FAILURES:
                try:
                    wav = await self._probe.window_wav(path, window.t_start, window.t_end)
                    vector = await embedder.embed_audio(wav)
                    failures = 0
                except (ProviderError, InvalidInput):
                    failures += 1
                    log.warning("audio window %s of %s not embedded", ordinal, document_id,
                                exc_info=True)  # fmt: skip
            chunks.append(
                Chunk(
                    ordinal=first_ordinal + ordinal,
                    content="",  # the transcript fills it in stage 2
                    page_start=None,
                    page_end=None,
                    char_start=None,
                    char_end=None,
                    modality="audio",
                    origin="window",
                    t_start=window.t_start,
                    t_end=window.t_end,
                    text_status="pending",
                    parent_ordinal=first_ordinal + len(windows) + parent_of[ordinal],
                )
            )
            vectors.append(vector)
        await progress(len(windows), len(windows))
        for parent in parents:  # context for the answer: filled with the children's transcripts
            chunks.append(
                Chunk(
                    ordinal=first_ordinal + len(chunks),
                    content="",
                    index_text="",  # not searchable by itself: only its children are retrieved
                    page_start=None,
                    page_end=None,
                    char_start=None,
                    char_end=None,
                    modality="audio",
                    origin="context",
                    t_start=parent.t_start,
                    t_end=parent.t_end,
                    text_status="pending",
                )
            )
            vectors.append(None)
        return AudioSegments(chunks, vectors, duration, fine=len(windows))
