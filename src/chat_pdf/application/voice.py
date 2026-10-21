import io
import wave

from chat_pdf.application.ports import (
    AudioClip,
    AudioConverter,
    ProviderFactory,
    ProviderRepository,
)
from chat_pdf.domain.errors import Conflict, InvalidInput
from chat_pdf.domain.speech import MAX_SPEAK_CHARS, strip_for_speech

MAX_AUDIO_BYTES = 8 * 1024 * 1024
MAX_AUDIO_SECONDS = 30
MIN_AUDIO_SECONDS = 0.25  # shorter than this is a click, not speech


def _seconds(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav)) as w:
        return w.getnframes() / w.getframerate()


class TranscribeAudio:
    def __init__(
        self, providers: ProviderRepository, factory: ProviderFactory, converter: AudioConverter
    ):
        self._providers = providers
        self._factory = factory
        self._converter = converter

    async def execute(self, audio: bytes) -> str:
        """Text of a short recording using the active STT profile ("" if there is no speech)."""
        profile = await self._providers.active("stt")
        if profile is None:
            raise Conflict("no_stt_profile", "No hay un proveedor de transcripción activo")
        if not audio:
            raise InvalidInput("empty_audio", "El audio está vacío")
        if len(audio) > MAX_AUDIO_BYTES:
            raise InvalidInput("audio_too_large", "El audio es demasiado grande")
        wav = await self._converter.to_wav16k(audio)
        seconds = _seconds(wav)
        if seconds > MAX_AUDIO_SECONDS:
            raise InvalidInput(
                "audio_too_long", f"El audio supera los {MAX_AUDIO_SECONDS} segundos"
            )
        if seconds < MIN_AUDIO_SECONDS:
            return ""
        return await self._factory.transcriber(profile).transcribe(wav)


class SpeakText:
    def __init__(self, providers: ProviderRepository, factory: ProviderFactory):
        self._providers = providers
        self._factory = factory

    async def execute(self, text: str) -> AudioClip:
        """Speech for `text` (markdown and [n] markers removed) using the active TTS profile."""
        profile = await self._providers.active("tts")
        if profile is None:
            raise Conflict("no_tts_profile", "No hay un proveedor de voz activo")
        clean = strip_for_speech(text)
        if not clean:
            raise InvalidInput("nothing_to_speak", "No hay texto para leer")
        if len(clean) > MAX_SPEAK_CHARS:
            raise InvalidInput("text_too_long", "Enviá el texto de a una frase o párrafo")
        return await self._factory.speaker(profile).speak(clean)
