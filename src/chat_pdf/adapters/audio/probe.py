import asyncio
import io
import wave
from collections.abc import Awaitable, Callable
from pathlib import Path

from chat_pdf.domain.audio import FRAME_S, AudioLevels
from chat_pdf.domain.errors import InvalidInput

TARGET_RATE = 16_000
PROBE_TIMEOUT_S = 30.0
LEVELS_TIMEOUT_S = 900.0  # a 3 h file decodes in a couple of minutes; this is a hang guard
SLICE_TIMEOUT_S = 60.0
FRAME_SAMPLES = round(TARGET_RATE * FRAME_S)
LEVEL_KEYS = {"lavfi.astats.Overall.RMS_level": "rms", "lavfi.astats.Overall.Peak_level": "peak"}

Run = Callable[[list[str], float], Awaitable[tuple[int, bytes]]]  # (args, timeout) -> (rc, stdout)


async def run_process(args: list[str], timeout: float) -> tuple[int, bytes]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except (TimeoutError, FileNotFoundError, OSError) as exc:
        raise InvalidInput("audio_undecodable", "No se pudo procesar el audio") from exc
    return proc.returncode or 0, out


def parse_levels(text: str) -> AudioLevels:
    """`ametadata=print` output of `astats` (one RMS + peak line per frame) -> AudioLevels."""
    rms: list[float] = []
    peaks: list[float] = []
    for line in text.splitlines():
        key, _, value = line.partition("=")
        which = LEVEL_KEYS.get(key)
        if which is None:
            continue
        try:
            db = float(value)
        except ValueError:
            continue
        if which == "rms":
            rms.append(db)
        else:
            peaks.append(10 ** (db / 20) if db > -200 else 0.0)  # "-inf" -> 0
    frames = min(len(rms), len(peaks))
    return AudioLevels(FRAME_S, rms[:frames], peaks[:frames])


def _wrap_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


class FfmpegAudioProbe:
    """Implements AudioProbe with ffprobe / ffmpeg subprocesses."""

    def __init__(self, run: Run = run_process):
        self._run = run

    async def duration(self, path: Path) -> float:
        args = [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path),
        ]  # fmt: skip
        code, out = await self._run(args, PROBE_TIMEOUT_S)
        try:
            seconds = float(out.decode().strip().splitlines()[0])
        except (ValueError, IndexError):
            seconds = 0.0
        if code != 0 or seconds <= 0:
            raise InvalidInput("audio_undecodable", "El archivo no es un audio válido")
        return seconds

    async def levels(self, path: Path) -> AudioLevels:
        graph = (
            f"aresample={TARGET_RATE},asetnsamples=n={FRAME_SAMPLES}:p=0,"
            "astats=metadata=1:reset=1:measure_perchannel=none:measure_overall=Peak_level+RMS_level,"
            "ametadata=mode=print:file=-"
        )
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-vn", "-ac", "1", "-af", graph, "-f", "null", "-",
        ]  # fmt: skip
        code, out = await self._run(args, LEVELS_TIMEOUT_S)
        levels = parse_levels(out.decode(errors="replace"))
        if code != 0 or not levels.rms_db:
            raise InvalidInput("audio_undecodable", "El audio no se pudo decodificar")
        return levels

    async def window_wav(self, path: Path, t_start: float, t_end: float) -> bytes:
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t_start:.3f}",
            "-t", f"{max(t_end - t_start, 0.1):.3f}", "-i", str(path),
            "-vn", "-ac", "1", "-ar", str(TARGET_RATE), "-f", "s16le", "pipe:1",
        ]  # fmt: skip
        code, pcm = await self._run(args, SLICE_TIMEOUT_S)
        if code != 0 or not pcm:
            raise InvalidInput("audio_undecodable", "No se pudo extraer el tramo de audio")
        return _wrap_wav(pcm)
