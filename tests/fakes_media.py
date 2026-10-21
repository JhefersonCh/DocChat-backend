"""Fakes for the multimedia ports (images, vision, native embeddings)."""

from uuid import uuid4

from chat_pdf.application.ports import MediaPlan, RenderedMedia
from chat_pdf.domain.errors import InvalidInput, ProviderError
from chat_pdf.domain.figures import MediaRegion
from chat_pdf.domain.providers import ProviderProfile


class FakeMultimodalEmbedder:
    """Embeds media into a 3-d space; `modalities` says what the fake "server" accepts."""

    def __init__(self, modalities: tuple[str, ...] = ()):
        self.modalities = modalities
        self.received: list[tuple[str, bytes]] = []

    async def _embed(self, kind: str, data: bytes) -> list[float]:
        if kind not in self.modalities:
            raise ProviderError(f"{kind} not supported")
        self.received.append((kind, data))
        return [0.0, 1.0, 0.0]

    async def embed_image(self, image, context=None):
        return await self._embed("image", image)

    async def embed_audio(self, wav):
        return await self._embed("audio", wav)

    async def embed_clip(self, frames):
        if "image" not in self.modalities:
            raise ProviderError("image not supported")
        self.received.append(("clip", b"|".join(frames)))
        return [0.0, 0.0, 1.0]


class FakeVision:
    """Scripted vision model: `answers` are consumed in order (an Exception is raised, then the
    next call gets the next answer); afterwards it answers a plain caption."""

    def __init__(self, *answers: str | Exception):
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.images: list[bytes] = []

    async def describe(self, image, prompt):
        self.images.append(image)
        self.prompts.append(prompt)
        answer = self.answers.pop(0) if self.answers else "Una imagen de prueba"
        if isinstance(answer, Exception):
            raise answer
        return answer


def vision_profile(**kw) -> ProviderProfile:
    return ProviderProfile(uuid4(), "vision", "vision", "http://vision", "deepseek-flash", **kw)


class FakeMediaExtractor:
    """Plans the given regions; renders deterministic bytes (one sha per `sha` or per region)."""

    def __init__(self, *regions: MediaRegion, width=None, height=None, unrenderable=()):
        self.regions = list(regions)
        self.size = (width, height)
        self.unrenderable = set(unrenderable)  # region indexes whose render raises InvalidInput
        self.plan_calls: list[dict] = []

    def plan(self, path, page_text_chars):
        self.plan_calls.append(dict(page_text_chars))
        return MediaPlan(self.regions, *self.size)

    def render(self, path, region):
        index = self.regions.index(region)
        if index in self.unrenderable:
            raise InvalidInput("invalid_image", "no se pudo renderizar")
        sha = region.sha or f"region-{index}"
        return RenderedMedia(f"asset-{index}".encode(), f"thumb-{index}".encode(), b"png", sha)


class FakeAudioProbe:
    """AudioProbe over a scripted file: `rms_db` per 50 ms frame, windows come out as stub WAVs
    named after their range. `slice_errors` are window starts whose slice cannot be cut."""

    def __init__(self, duration: float, rms_db: list[float] | None = None, slice_errors=()):
        from chat_pdf.domain.audio import FRAME_S, AudioLevels

        self.duration_s = duration
        frames = round(duration / FRAME_S)
        rms = rms_db if rms_db is not None else [-20.0] * frames
        self._levels = AudioLevels(FRAME_S, rms, [0.5] * len(rms))
        self.slice_errors = set(slice_errors)
        self.slices: list[tuple[float, float]] = []

    async def duration(self, path):
        return self.duration_s

    async def levels(self, path):
        return self._levels

    async def window_wav(self, path, t_start, t_end):
        if t_start in self.slice_errors:
            raise InvalidInput("audio_undecodable", "no se pudo extraer el tramo")
        self.slices.append((t_start, t_end))
        return f"wav:{t_start}-{t_end}".encode()


class FakeVideoProbe:
    """VideoProbe over a scripted video: `scenes` are the second where each new scene starts
    (frames in the same scene render identical bytes, so they are duplicates)."""

    def __init__(self, duration=40.0, scenes=(15.0,), info=None, unreadable=()):
        from chat_pdf.domain.video import VideoInfo

        self.duration_s = duration
        self.scenes = list(scenes)
        self._info = info or VideoInfo(
            duration, 1280, 720, "h264", "aac", "mov,mp4,m4a,3gp,3g2,mj2"
        )
        self.unreadable = set(unreadable)
        self.frames: list[float] = []
        self.transcodes: list[tuple] = []

    def scene_of(self, t: float) -> int:
        return sum(1 for s in self.scenes if t >= s)

    async def info(self, path):
        return self._info

    async def scene_times(self, path, threshold):
        return list(self.scenes)

    async def extract_seconds(self, path, dest):
        pass

    async def frame(self, path, t, cache=None):
        if t in self.unreadable:
            raise InvalidInput("frame_undecodable", "sin imagen")
        self.frames.append(t)
        scene = self.scene_of(t)
        return RenderedMedia(
            f"asset@{t}".encode(), f"thumb@{t}".encode(), f"scene-{scene}".encode(), f"sha-{scene}"
        )

    async def signature(self, frame):
        return bytes([int(frame.sha.split("-")[1]) * 100 % 256]) * 100  # scene -> flat grey

    async def transcode_playable(self, source, dest):
        self.transcodes.append((source, dest))


class SceneEmbedder(FakeMultimodalEmbedder):
    """Image vectors by scene (`scene-N` bytes): orthogonal between scenes, equal within one."""

    async def embed_image(self, image, context=None):
        await super().embed_image(image, context)
        scene = int(image.decode().split("-")[1])
        vector = [0.0] * 8
        vector[scene % 8] = 1.0
        return vector
