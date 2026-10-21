"""Ports of the video pipeline (kept apart from ports.py: that file is near its size budget)."""

from pathlib import Path
from typing import Protocol

from chat_pdf.application.ports import RenderedMedia
from chat_pdf.domain.video import VideoInfo


class VideoProbe(Protocol):
    """Reads a video of any container / codec (ffmpeg behind it)."""

    async def info(self, path: Path) -> VideoInfo:
        """Duration, size and codecs; InvalidInput if the file has no decodable video."""
        ...

    async def scene_times(self, path: Path, threshold: float) -> list[float]:
        """Seconds of every scene cut (ffmpeg `scene` score above `threshold`)."""
        ...

    async def extract_seconds(self, path: Path, dest: Path) -> None:
        """Single pass writing every whole second's picture into `dest` (used by `frame`)."""
        ...

    async def frame(self, path: Path, t: float, cache: Path | None = None) -> RenderedMedia:
        """The picture at second `t` (taken from the `extract_seconds` folder `cache` when it is
        a whole second there), rendered in the three sizes of the image pipeline;
        InvalidInput if there is none."""
        ...

    async def signature(self, frame: RenderedMedia) -> bytes:
        """A tiny grey thumbnail of a rendered frame, to compare pixels (text changes on slides)."""
        ...

    async def transcode_playable(self, source: Path, dest: Path) -> None:
        """Re-encode to h264 / aac mp4 (720p max, faststart) at `dest`."""
        ...
