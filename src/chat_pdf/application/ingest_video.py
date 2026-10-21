"""Stage 1 of video ingest (plan D8, §2.5): playable variant, a frame per second (+ scene cuts)
with a native image embedding each (literally identical neighbours skipped), ~4 s clips embedded
as one multi-frame input and the audio track through the audio pipeline. Local only: the video
is searchable before any caption or transcript exists."""

import logging
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import UUID

from chat_pdf.application.ingest_audio import AudioIngest
from chat_pdf.application.ports import FileStorage, ProviderFactory, RenderedMedia
from chat_pdf.application.ports_video import VideoProbe
from chat_pdf.domain.errors import InvalidInput, ProviderError
from chat_pdf.domain.models import Chunk
from chat_pdf.domain.providers import ProviderProfile
from chat_pdf.domain.video import (
    FRAME_STEP_S,
    MAX_STATIC_GAP_S,
    MAX_VIDEO_S,
    MIN_CLIP_FRAMES,
    SCENE_THRESHOLD,
    VideoInfo,
    clip_blocks,
    frame_grid,
    frame_ranges,
    is_playable,
    is_static_repeat,
    mean_vector,
    pixel_change,
    representatives,
)

log = logging.getLogger(__name__)

MAX_CONSECUTIVE_EMBED_FAILURES = 3  # the embeddings server is down: stop instead of skipping all
Progress = Callable[[int, int, str], Awaitable[None]]  # done, total, stage; raises IngestCancelled
TOO_LONG = "El video dura más de {hours} h; dividilo en partes más cortas"
NO_FRAMES = "No se pudo extraer ningún fotograma del video"
SILENT_AUDIO = ("audio_silent", "audio_undecodable")  # the track is empty / broken: index frames


@dataclass(frozen=True)
class VideoSegments:
    chunks: list[Chunk]  # frames first, then the audio windows
    vectors: list[list[float] | None]
    info: VideoInfo
    frames: int
    windows: int
    transcoded: bool
    clips: int = 0


class VideoIngest:
    def __init__(
        self,
        probe: VideoProbe,
        audio: AudioIngest | None,
        factory: ProviderFactory,
        storage: FileStorage,
    ):
        self._probe = probe
        self._audio = audio
        self._factory = factory
        self._storage = storage

    async def segments(
        self, document_id: UUID, path: Path, profile: ProviderProfile, progress: Progress
    ) -> VideoSegments:
        info = await self._probe.info(path)
        if info.duration_s > MAX_VIDEO_S:
            raise InvalidInput("video_too_long", TOO_LONG.format(hours=int(MAX_VIDEO_S // 3600)))
        await self._storage.clear_assets(document_id)  # a re-index regenerates every frame
        await progress(0, 0, "extracting")
        transcoded = await self._playable(document_id, path, info)
        scenes = await self._probe.scene_times(path, SCENE_THRESHOLD)
        grid = frame_grid(scenes, info.duration_s)
        chunks, vectors = await self._frames(document_id, path, profile, info, grid, progress)
        clips = sum(1 for c in chunks if c.origin == "clip")
        windows = 0
        audio_chunks: list[Chunk] = []
        if info.has_audio and self._audio is not None:
            try:
                audio = await self._audio.segments(
                    document_id,
                    path,
                    profile,
                    lambda done, total: progress(done, total, "embedding"),
                    first_ordinal=len(chunks),
                    keep_assets=True,
                )
            except InvalidInput as exc:
                if exc.code not in SILENT_AUDIO:
                    raise
                log.info("video %s has no usable audio track: %s", document_id, exc)
            else:
                audio_chunks = audio.chunks
                chunks, vectors = [*chunks, *audio.chunks], [*vectors, *audio.vectors]
                windows = audio.fine
        frames = len(chunks) - clips - len(audio_chunks)
        return VideoSegments(chunks, vectors, info, frames, windows, transcoded, clips)

    async def _playable(self, document_id: UUID, path: Path, info: VideoInfo) -> bool:
        """Transcode once unless a browser plays the original (or a re-index finds the copy)."""
        if is_playable(info, path.suffix):
            return False
        dest = self._storage.playable_path(document_id)
        if not dest.is_file():
            await self._probe.transcode_playable(path, dest)
        return True

    async def _frames(
        self,
        document_id: UUID,
        path: Path,
        profile: ProviderProfile,
        info: VideoInfo,
        grid: list[tuple[float, bool]],
        progress: Progress,
    ) -> tuple[list[Chunk], list[list[float] | None]]:
        with tempfile.TemporaryDirectory(prefix="frames-") as tmp:
            cache: Path | None = Path(tmp)
            try:  # one ffmpeg pass for every whole second; scene cuts are seeked one by one
                await self._probe.extract_seconds(path, Path(tmp))
            except InvalidInput:
                log.warning("1 fps pass failed for %s; seeking frame by frame", document_id)
                cache = None
            return await self._frames_from(document_id, path, profile, info, grid, progress, cache)

    async def _frames_from(
        self,
        document_id: UUID,
        path: Path,
        profile: ProviderProfile,
        info: VideoInfo,
        grid: list[tuple[float, bool]],
        progress: Progress,
        frames_dir: Path | None,
    ) -> tuple[list[Chunk], list[list[float] | None]]:
        """A frame per second (+ scene cuts), each embedded natively; clips of ~4 s embedded as one
        multi-frame input. Representative frames are the parents (they get the vision caption);
        the other frames and the clips point to the representative they fall under."""
        embedder = self._factory.multimodal_embedder(profile)
        reps = representatives(grid)
        blocks = {b[-1]: b for b in clip_blocks(grid)}
        total = len(grid) + len(blocks)
        cache: dict[str, list[float]] = {}  # identical pictures are embedded once
        rendered_at: dict[int, RenderedMedia] = {}  # frames of the clip being built
        rep_at: dict[int, int | None] = {}  # grid index -> ordinal of its representative
        vector_at: dict[int, list[float]] = {}  # grid index -> its (or its kept twin's) vector
        kept: list[tuple[float, list[float]]] = []
        signatures: list[bytes] = []  # grey thumbnail of each kept frame
        chunks: list[Chunk] = []
        vectors: list[list[float] | None] = []
        frame_slots: list[int] = []  # positions in `chunks` of the frames (clips excluded)
        rep_ordinal: int | None = None
        failures = clip_failures = 0
        native_clips = hasattr(embedder, "embed_clip")  # else the mean of the frame vectors
        for i, (t, _) in enumerate(grid):
            await progress(i + sum(1 for end in blocks if end < i), total, "embedding")
            rep_at[i] = rep_ordinal
            try:
                rendered = await self._probe.frame(path, t, frames_dir)
            except InvalidInput:
                log.warning("no frame at %.1fs of %s; skipped", t, document_id)
                rendered = None
            if rendered is not None:
                rendered_at[i] = rendered
                signature = await self._probe.signature(rendered)
                is_rep = i in reps
                next_t = grid[i + 1][0] if i + 1 < len(grid) else info.duration_s
                if (
                    not is_rep
                    and kept
                    and is_static_repeat(
                        pixel_change(signatures[-1], signature), next_t - kept[-1][0]
                    )
                ):
                    vector_at[i] = kept[-1][1]  # literally the same picture: the kept one lasts
                else:
                    vector = cache.get(rendered.sha)
                    if vector is None:
                        try:
                            vector = await embedder.embed_image(rendered.embed)
                        except (ProviderError, InvalidInput):
                            failures += 1
                            log.warning("frame embedding failed for %s", document_id, exc_info=True)
                            if failures >= MAX_CONSECUTIVE_EMBED_FAILURES:
                                raise
                            vector = None
                        else:
                            cache[rendered.sha] = vector
                    if vector is not None:
                        vector_at[i] = vector
                        failures = 0
                        ordinal = len(chunks)
                        name = f"{ordinal}.webp"
                        await self._storage.save_asset(document_id, "assets", name, rendered.asset)
                        await self._storage.save_asset(document_id, "thumbs", name, rendered.thumb)
                        kept.append((t, vector))
                        signatures.append(signature)
                        frame_slots.append(ordinal)
                        chunks.append(
                            Chunk(
                                ordinal=ordinal,
                                content="",  # a representative's caption fills it in stage 2
                                page_start=None,
                                page_end=None,
                                char_start=None,
                                char_end=None,
                                modality="frame",
                                origin="frame",
                                t_start=t,
                                asset_key=name,
                                media_sha=rendered.sha,
                                text_status="pending",
                                parent_ordinal=None if is_rep else rep_ordinal,
                            )
                        )
                        vectors.append(vector)
                        if is_rep:
                            rep_ordinal = rep_at[i] = ordinal
            block = blocks.get(i)
            if block is not None:
                frame_vectors = [vector_at[k] for k in block if k in vector_at]
                made = None
                if native_clips:
                    made = await self._clip(
                        document_id, embedder, grid, block, rendered_at, frame_vectors,
                        rep_at[block[0]], len(chunks),
                    )  # fmt: skip
                    clip_failures = 0 if made else clip_failures + 1
                    native_clips = clip_failures < 2  # no multi-image input: use the mean instead
                if made is None:  # the mean of the frame vectors: no extra embedding cost
                    made = await self._clip(
                        document_id, None, grid, block, rendered_at, frame_vectors,
                        rep_at[block[0]], len(chunks),
                    )  # fmt: skip
                if made is not None:
                    chunks.append(made[0])
                    vectors.append(made[1])
                for k in block:
                    rendered_at.pop(k, None)
                    vector_at.pop(k, None)
        await progress(total, total, "embedding")
        if not frame_slots:
            raise InvalidInput("video_no_frames", NO_FRAMES)
        ranges = frame_ranges(
            [t for t, _ in kept], grid[-1][0], info.duration_s, max_gap=MAX_STATIC_GAP_S
        )
        for slot, (start, end) in zip(frame_slots, ranges, strict=True):
            chunks[slot] = replace(chunks[slot], t_start=start, t_end=end)
        return chunks, vectors

    async def _clip(
        self,
        document_id: UUID,
        embedder,
        grid: list[tuple[float, bool]],
        block: list[int],
        rendered_at: dict[int, RenderedMedia],
        frame_vectors: list[list[float]],
        parent_ordinal: int | None,
        ordinal: int,
    ) -> tuple[Chunk, list[float]] | None:
        """One segment for the frames of `block`: embedded together as ONE multi-image input when
        the server allows it (`embedder`), else the L2-normalized mean of the frame vectors (no
        extra embedding cost). None when it cannot be built."""
        frames = [rendered_at[k] for k in block if k in rendered_at]
        if len(frames) < MIN_CLIP_FRAMES:
            return None
        if embedder is not None:
            try:
                vector = await embedder.embed_clip([f.embed for f in frames])
            except (ProviderError, InvalidInput):
                log.warning("clip embedding failed for %s", document_id, exc_info=True)
                return None
        elif len(frame_vectors) >= MIN_CLIP_FRAMES:
            vector = mean_vector(frame_vectors)
        else:
            return None
        name = f"{ordinal}.webp"  # its picture is the clip's first frame
        await self._storage.save_asset(document_id, "assets", name, frames[0].asset)
        await self._storage.save_asset(document_id, "thumbs", name, frames[0].thumb)
        t_start = grid[block[0]][0]
        return (
            Chunk(
                ordinal=ordinal,
                content="",  # inherits the representative's caption in stage 2
                page_start=None,
                page_end=None,
                char_start=None,
                char_end=None,
                modality="frame",
                origin="clip",
                t_start=t_start,
                t_end=round(grid[block[-1]][0] + FRAME_STEP_S, 3),
                asset_key=name,
                text_status="pending",
                parent_ordinal=parent_ordinal,
            ),
            vector,
        )
