"""ffprobe / ffmpeg adapter of the VideoProbe port (subprocesses; `run` is injectable for tests)."""

import asyncio
import io
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from chat_pdf.adapters.media.encode import render_all
from chat_pdf.application.ports import RenderedMedia
from chat_pdf.domain.errors import InvalidInput
from chat_pdf.domain.video import VideoInfo

PROBE_TIMEOUT_S = 60.0
SCENES_TIMEOUT_S = 3600.0  # a 2 h video decodes (scaled down) in minutes; this is a hang guard
FRAME_TIMEOUT_S = 60.0
SECOND_FRAME_WIDTH = 1280  # frames of the 1 fps pass are stored at most this wide
TRANSCODE_TIMEOUT_S = 4 * 3600.0
SCENE_WIDTH = 320  # scene detection runs on a small copy: 10x faster, same cuts
SIGNATURE_SIZE = (96, 54)  # grey thumbnail compared between consecutive frames
PTS_TIME = re.compile(r"pts_time:(\d+(?:\.\d+)?)")

Run = Callable[[list[str], float], Awaitable[tuple[int, bytes]]]  # (args, timeout) -> (rc, stdout)


async def run_process(args: list[str], timeout: float) -> tuple[int, bytes]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except (TimeoutError, FileNotFoundError, OSError) as exc:
        raise InvalidInput("video_undecodable", "No se pudo procesar el video") from exc
    return proc.returncode or 0, out


def parse_info(raw: bytes) -> VideoInfo:
    """`ffprobe -print_format json -show_streams -show_format` -> VideoInfo."""
    try:
        data = json.loads(raw)
    except ValueError:
        data = {}
    streams = data.get("streams") or []
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format") or {}
    try:
        duration = float(fmt.get("duration") or (video or {}).get("duration") or 0)
        width, height = int(video["width"]), int(video["height"])  # type: ignore[index]
    except (TypeError, ValueError, KeyError):
        duration, width, height = 0.0, 0, 0
    if video is None or duration <= 0 or not width or not height:
        raise InvalidInput("video_undecodable", "El archivo no es un video válido")
    return VideoInfo(
        duration_s=duration,
        width=width,
        height=height,
        video_codec=str(video.get("codec_name", "")),
        audio_codec=str(audio["codec_name"]) if audio and audio.get("codec_name") else None,
        container=str(fmt.get("format_name", "")),
        pix_fmt=video.get("pix_fmt"),
    )


def parse_scene_times(text: str) -> list[float]:
    """`metadata=print` output of the `select` filter -> the seconds of each kept frame."""
    return [float(m) for m in PTS_TIME.findall(text)]


class FfmpegVideoProbe:
    """Implements VideoProbe with ffprobe / ffmpeg subprocesses."""

    def __init__(self, run: Run = run_process):
        self._run = run

    async def info(self, path: Path) -> VideoInfo:
        args = [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_streams", "-show_format", str(path),
        ]  # fmt: skip
        code, out = await self._run(args, PROBE_TIMEOUT_S)
        if code != 0:
            raise InvalidInput("video_undecodable", "El archivo no es un video válido")
        return parse_info(out)

    async def scene_times(self, path: Path, threshold: float) -> list[float]:
        graph = f"scale={SCENE_WIDTH}:-2,select='gt(scene,{threshold})',metadata=print:file=-"
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-an", "-sn", "-vf", graph, "-f", "null", "-",
        ]  # fmt: skip
        code, out = await self._run(args, SCENES_TIMEOUT_S)
        if code != 0:
            raise InvalidInput("video_undecodable", "No se pudo analizar el video")
        return parse_scene_times(out.decode(errors="replace"))

    async def extract_seconds(self, path: Path, dest: Path) -> None:
        """One ffmpeg pass: `dest/<second>.jpg` for every whole second (fps=1, exact pts)."""
        graph = f"fps=1,scale='min({SECOND_FRAME_WIDTH},iw)':-2"
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-an", "-sn",
            "-vf", graph, "-vsync", "cfr", "-q:v", "3", "-start_number", "0",
            str(dest / "%06d.jpg"),
        ]  # fmt: skip
        code, _ = await self._run(args, SCENES_TIMEOUT_S)
        if code != 0:
            raise InvalidInput("video_undecodable", "No se pudieron extraer los fotogramas")

    async def frame(self, path: Path, t: float, cache: Path | None = None) -> RenderedMedia:
        if cache is not None and t == int(t):
            cached = cache / f"{int(t):06d}.jpg"
            if cached.is_file():
                try:
                    data = cached.read_bytes()
                    cached.unlink(missing_ok=True)
                    return await asyncio.to_thread(_render, data)
                except (UnidentifiedImageError, OSError):
                    pass  # fall through to a direct seek
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(path),
            "-an", "-sn", "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "pipe:1",
        ]  # fmt: skip
        code, png = await self._run(args, FRAME_TIMEOUT_S)
        if code != 0 or not png:
            raise InvalidInput("frame_undecodable", f"No hay imagen en el segundo {t:.1f}")
        try:
            return await asyncio.to_thread(_render, png)
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidInput("frame_undecodable", "No se pudo leer el fotograma") from exc

    async def signature(self, frame: RenderedMedia) -> bytes:
        return await asyncio.to_thread(_signature, frame.embed)

    async def transcode_playable(self, source: Path, dest: Path) -> None:
        partial = dest.with_name(dest.name + ".part")
        args = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "25", "-vf", "scale=-2:'min(720,ih)'", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-f", "mp4", str(partial),
        ]  # fmt: skip
        code, _ = await self._run(args, TRANSCODE_TIMEOUT_S)
        if code != 0 or not partial.is_file():
            partial.unlink(missing_ok=True)
            raise InvalidInput("transcode_failed", "No se pudo convertir el video a mp4")
        partial.replace(dest)


def _signature(png: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(png)) as image:
            return image.convert("L").resize(SIGNATURE_SIZE, Image.BILINEAR).tobytes()
    except (UnidentifiedImageError, OSError):
        return b""  # not comparable: pixel_change() counts it as a different picture


def _render(png: bytes) -> RenderedMedia:
    with Image.open(io.BytesIO(png)) as image:
        image.load()
        return render_all(image)
