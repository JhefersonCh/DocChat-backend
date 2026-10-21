"""Video rules (plan D8, §2.5). Pure: keyframe timestamps, near-duplicate frames, which files a
browser can play and the storyboard. The pixels and the codecs are the adapters' business."""

import math
from dataclasses import dataclass
from uuid import UUID

from chat_pdf.domain.models import Chunk

MAX_VIDEO_S = 2 * 3600.0  # longer sources are refused (a frame costs ~1 s to embed on CPU)
SCENE_THRESHOLD = 0.3  # ffmpeg `scene` score that counts as a cut
MIN_GAP_S = 2.0  # two kept frames are never closer than this
MAX_GAP_S = 10.0  # ... and no stretch of video goes without one longer than this
MAX_FRAMES = 900
# Fine-grained indexing (granularity pass): a frame per second, searched natively; only the
# representative frames (scene keyframes + one every REP_EVERY_S) get a vision caption, the rest
# inherit it. Clips: several consecutive frames embedded together as one input (actions / events).
FRAME_STEP_S = 1.0
MAX_GRID_FRAMES = 3600  # one hour at 1 fps; longer videos are thinned evenly
REP_EVERY_S = 5.0
SCENE_MERGE_S = 0.4  # a scene cut this close to a grid frame IS that frame
MAX_STATIC_GAP_S = 2.0  # identical frames are skipped, but never leaving a hole longer than this
CLIP_S = 4.0
MIN_CLIP_FRAMES = 2
DUPLICATE_COSINE = 0.99  # consecutive frames this alike (embedding) AND pixel-alike are one picture
# Embeddings barely move when a slide swaps a few lines of text, so the pixels have the last word:
# a frame whose grey thumbnail changed in more than this fraction of its pixels is a new picture.
PIXEL_LEVEL = 24  # a grey level (0-255) difference that counts as "this pixel changed"
DUPLICATE_PIXEL_FRACTION = 0.004

# Containers / codecs every browser plays without help.
MP4_VIDEO = {"h264"}
MP4_AUDIO = {"aac", "mp3"}
WEBM_VIDEO = {"vp8", "vp9", "av1"}
WEBM_AUDIO = {"opus", "vorbis"}


@dataclass(frozen=True)
class VideoInfo:
    """What ffprobe says about a file (the first video stream and the first audio stream)."""

    duration_s: float
    width: int
    height: int
    video_codec: str
    audio_codec: str | None
    container: str  # ffprobe format_name, e.g. "mov,mp4,m4a,3gp,3g2,mj2" | "matroska,webm"
    pix_fmt: str | None = None

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None


def is_playable(info: VideoInfo, extension: str) -> bool:
    """True when a browser plays the ORIGINAL (mp4 with h264/aac in yuv420p, webm with vp8/vp9/av1
    and opus/vorbis). The extension matters: mov and mkv hold playable codecs but browsers refuse
    the container, so those are transcoded once."""
    ext = extension.lower().lstrip(".")
    containers = set(info.container.split(","))
    if ext in ("mp4", "m4v") and "mp4" in containers:
        yuv420 = info.pix_fmt in (None, "yuv420p", "yuvj420p")
        return (
            info.video_codec in MP4_VIDEO
            and yuv420
            and (info.audio_codec is None or info.audio_codec in MP4_AUDIO)
        )
    if ext == "webm" and "webm" in containers:
        return info.video_codec in WEBM_VIDEO and (
            info.audio_codec is None or info.audio_codec in WEBM_AUDIO
        )
    return False


def select_keyframes(
    scene_times: list[float],
    duration: float,
    min_gap: float = MIN_GAP_S,
    max_gap: float = MAX_GAP_S,
    cap: int = MAX_FRAMES,
) -> list[float]:
    """Timestamps (s) to extract: the first instant, every scene cut at least `min_gap` after the
    previous frame, and filler frames so that no gap exceeds `max_gap`. Over `cap`, the filler and
    cuts are thinned evenly (the first frame stays)."""
    if duration <= 0:
        return []
    cuts = sorted(t for t in scene_times if 0 < t < duration)
    times = [0.0]
    for cut in cuts:
        if cut - times[-1] >= min_gap:
            times.append(cut)
    filled: list[float] = []
    for i, start in enumerate(times):
        filled.append(start)
        end = times[i + 1] if i + 1 < len(times) else duration
        steps = math.ceil((end - start) / max_gap) - 1  # extra frames needed inside the gap
        if steps > 0:
            step = (end - start) / (steps + 1)
            filled.extend(start + step * k for k in range(1, steps + 1))
    filled = [round(t, 3) for t in filled]
    if len(filled) > cap:
        filled = [filled[i * len(filled) // cap] for i in range(cap)]
    return filled


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def mean_vector(vectors: list[list[float]]) -> list[float]:
    """L2-normalized mean: a clip's vector when the server cannot embed several frames at once."""
    total = [sum(column) for column in zip(*vectors, strict=True)]
    norm = math.sqrt(sum(x * x for x in total))
    return [x / norm for x in total] if norm else total


def dedupe_consecutive(
    vectors: list[list[float]], threshold: float = DUPLICATE_COSINE
) -> list[int]:
    """Indexes to keep: a frame is dropped when it is more similar than `threshold` to the last
    KEPT one (so a slow drift is still caught eventually). The first frame is always kept."""
    kept: list[int] = []
    for i, vector in enumerate(vectors):
        if not kept or cosine(vectors[kept[-1]], vector) <= threshold:
            kept.append(i)
    return kept


def pixel_change(a: bytes, b: bytes) -> float:
    """Fraction (0..1) of pixels that differ visibly between two equally-sized grey thumbnails
    (1.0 when they cannot be compared)."""
    if not a or len(a) != len(b):
        return 1.0
    return sum(1 for x, y in zip(a, b, strict=True) if abs(x - y) > PIXEL_LEVEL) / len(a)


def is_duplicate_frame(
    cos: float, change: float, gap_to_next: float, max_gap: float = MAX_GAP_S
) -> bool:
    """A frame is dropped (the last kept one simply lasts longer) only when it is the same picture
    by embedding AND by pixels, and dropping it leaves no stretch longer than `max_gap`:
    `gap_to_next` is the time from the last kept frame to the next candidate (or the end)."""
    return (
        cos > DUPLICATE_COSINE
        and change <= DUPLICATE_PIXEL_FRACTION
        and gap_to_next <= max_gap + 1e-6
    )


def frame_ranges(
    kept: list[float], last_extracted: float, duration: float, max_gap: float = MAX_GAP_S
) -> list[tuple[float, float]]:
    """(t_start, t_end) of each kept frame: it lasts until the next kept one; the last lasts up to
    one filler gap after the last extracted instant (dropped duplicates extend a frame's life)."""
    ends = [*kept[1:], min(duration, max(last_extracted, kept[-1] if kept else 0) + max_gap)]
    return [(t, max(end, t + 0.1)) for t, end in zip(kept, ends, strict=False)]


@dataclass(frozen=True)
class StoryboardEntry:
    segment_id: UUID
    t: float
    t_end: float | None


def storyboard(chunks: list[Chunk]) -> list[StoryboardEntry]:
    """The frames of a video with a thumbnail, in time order (a filmstrip)."""
    frames = [  # the representative frames: the fine 1 fps frames and clips hang under them
        c
        for c in chunks
        if c.modality == "frame"
        and c.origin != "clip"
        and c.parent_id is None
        and c.has_thumb
        and c.id is not None
    ]
    frames.sort(key=lambda c: (c.t_start or 0.0, c.ordinal))
    return [StoryboardEntry(c.id, c.t_start or 0.0, c.t_end) for c in frames if c.id is not None]


def frame_grid(
    scene_times: list[float],
    duration: float,
    step: float = FRAME_STEP_S,
    cap: int = MAX_GRID_FRAMES,
) -> list[tuple[float, bool]]:
    """(t, is_scene) of every frame to extract: one per `step` plus every scene cut that is not
    already (almost) on the grid. Over `cap` the grid is thinned evenly (scene cuts survive)."""
    if duration <= 0:
        return []
    count = int(duration // step) + 1
    grid = [round(i * step, 3) for i in range(count) if i * step < duration]
    cuts = sorted(t for t in scene_times if 0 < t < duration)
    scene_hits: set[float] = set()
    extra: list[float] = []
    for cut in cuts:
        nearest = min(grid, key=lambda g: abs(g - cut))
        if abs(nearest - cut) <= SCENE_MERGE_S:
            scene_hits.add(nearest)
        else:
            extra.append(round(cut, 3))
    if len(grid) + len(extra) > cap:
        room = max(cap - len(extra), 1)
        picked = {grid[i * len(grid) // room] for i in range(room)}
        grid = sorted({0.0, *picked, *scene_hits})
    merged = sorted({*grid, *extra})
    return [(t, t in scene_hits or t in extra) for t in merged]


def representatives(grid: list[tuple[float, bool]], every: float = REP_EVERY_S) -> set[int]:
    """Indexes of the frames that get a caption: the first, every scene cut and at least one
    every `every` seconds. The others inherit the caption of the nearest representative before."""
    reps: set[int] = set()
    last = -math.inf
    for i, (t, is_scene) in enumerate(grid):
        if i == 0 or is_scene or t - last >= every - 1e-6:
            reps.add(i)
            last = t
    return reps


def clip_blocks(grid: list[tuple[float, bool]], clip_s: float = CLIP_S) -> list[list[int]]:
    """Consecutive blocks of grid indexes spanning `clip_s` seconds each (scene-cut extras
    included); a tail with fewer than MIN_CLIP_FRAMES frames joins the previous block."""
    blocks: list[list[int]] = []
    for i, (t, _) in enumerate(grid):
        if not blocks or t - grid[blocks[-1][0]][0] >= clip_s - 1e-6:
            blocks.append([i])
        else:
            blocks[-1].append(i)
    if len(blocks) > 1 and len(blocks[-1]) < MIN_CLIP_FRAMES:
        tail = blocks.pop()
        blocks[-1].extend(tail)
    return [b for b in blocks if len(b) >= MIN_CLIP_FRAMES]


def is_static_repeat(change: float, gap_to_next: float) -> bool:
    """Skip a frame only when its pixels are literally the last kept frame's and skipping it
    leaves no hole longer than MAX_STATIC_GAP_S (`gap_to_next`: last kept -> next candidate)."""
    return change <= DUPLICATE_PIXEL_FRACTION and gap_to_next <= MAX_STATIC_GAP_S + 1e-6
