"""Ports of the audio pipeline (kept apart from ports.py: that file is near its size budget)."""

from pathlib import Path
from typing import Protocol

from chat_pdf.domain.audio import AudioLevels


class AudioProbe(Protocol):
    """Reads an audio file of any container / codec (ffmpeg behind it)."""

    async def duration(self, path: Path) -> float:
        """Length in seconds; InvalidInput if the file is not decodable audio."""
        ...

    async def levels(self, path: Path) -> AudioLevels:
        """Loudness (RMS and peak per 50 ms frame) of the whole file."""
        ...

    async def window_wav(self, path: Path, t_start: float, t_end: float) -> bytes:
        """The [t_start, t_end] slice as a 16 kHz mono 16-bit WAV."""
        ...
