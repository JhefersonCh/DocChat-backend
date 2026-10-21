import asyncio
import io
import wave

from chat_pdf.domain.errors import InvalidInput

TARGET_RATE = 16_000
CONVERT_TIMEOUT_S = 20.0


def _is_target_wav(audio: bytes) -> bool:
    """True if `audio` is already a 16 kHz mono 16-bit PCM WAV (no ffmpeg round trip needed)."""
    try:
        with wave.open(io.BytesIO(audio)) as w:
            return (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (TARGET_RATE, 1, 2)
    except (wave.Error, EOFError):
        return False


class FfmpegConverter:
    """Implements AudioConverter: any browser recording (webm/ogg/mp4/wav) -> 16 kHz mono WAV."""

    def __init__(self, max_seconds: int):
        self._max_seconds = max_seconds

    async def to_wav16k(self, audio: bytes) -> bytes:
        if _is_target_wav(audio):
            return audio
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
            "-t", str(self._max_seconds + 1), "-vn", "-ac", "1", "-ar", str(TARGET_RATE),
            "-f", "s16le", "pipe:1",
        ]  # fmt: skip
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, _ = await asyncio.wait_for(proc.communicate(audio), CONVERT_TIMEOUT_S)
        except (TimeoutError, FileNotFoundError, OSError) as exc:
            raise InvalidInput("audio_undecodable", "No se pudo procesar el audio") from exc
        if proc.returncode != 0 or not out:
            raise InvalidInput("audio_undecodable", "El audio no se pudo decodificar")
        # Raw PCM on a pipe: a WAV header written to a pipe cannot be seeked back to fix its sizes.
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(TARGET_RATE)
            w.writeframes(out)
        return buf.getvalue()
