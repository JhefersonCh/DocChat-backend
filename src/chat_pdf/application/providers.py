import io
import math
import struct
import wave
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from chat_pdf.application.ports import Database, ProviderFactory, ProviderRepository
from chat_pdf.domain.calibration import parse_calibration
from chat_pdf.domain.errors import (
    Conflict,
    InvalidInput,
    NotFound,
    ProviderError,
    ServiceUnavailable,
    ToolsUnsupported,
)
from chat_pdf.domain.prompts import (
    PROBE_PING,
    PROBE_TOOL_DESCRIPTION,
    PROBE_TOOL_NAME,
    PROBE_TOOL_REQUEST,
)
from chat_pdf.domain.providers import ProviderKind, ProviderProfile
from chat_pdf.domain.rules import normalize_profile_data, validate_k_range

PING_TOOL = {
    "type": "function",
    "function": {
        "name": PROBE_TOOL_NAME,
        "description": PROBE_TOOL_DESCRIPTION,
        "parameters": {"type": "object", "properties": {}},
    },
}

PROBE_SPEECH = "Hola"
PROBE_VISION = "Describí esta imagen en una frase corta."
PROBE_CAPTION_CHARS = 200


def probe_wav(seconds: float = 1.0, rate: int = 16_000) -> bytes:
    """A tiny 16 kHz mono WAV (a quiet 440 Hz tone) to test speech-to-text endpoints."""
    n = int(seconds * rate)
    samples = [int(3000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n)]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{n}h", *samples))
    return buf.getvalue()


def probe_png(size: int = 64) -> bytes:
    """A tiny RGB PNG (horizontal gradient) to test image-capable embedding endpoints."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + b"".join(bytes((x * 255 // (size - 1), 80, 160)) for x in range(size))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


VECTOR_SPACE_FIELDS = ("model", "dimensions", "query_prefix", "document_prefix")

LOCAL_TTS = {
    "kind": "tts",
    "name": "Supertonic 3 F2 (local)",
    "model": "supertonic-3",
    "voice": "F2",
    "language": "es",
}

LOCAL_EMBEDDING = {
    "kind": "embedding",
    "name": "EmbeddingGemma 2 (local)",
    "model": "embeddinggemma-2",
    "dimensions": 768,
    "query_prefix": "task: question answering | query: ",
    "document_prefix": "title: {title} | text: ",
}


def _not_found() -> NotFound:
    return NotFound("provider_not_found", "Proveedor no encontrado")


async def _require(providers: ProviderRepository, profile_id: UUID) -> ProviderProfile:
    profile = await providers.get(profile_id)
    if profile is None:
        raise _not_found()
    return profile


class ManageProviders:
    """CRUD of provider profiles (several operations, so methods instead of `execute`)."""

    def __init__(self, providers: ProviderRepository):
        self._providers = providers

    async def list(self) -> list[ProviderProfile]:
        return await self._providers.list()

    async def create(self, data: dict[str, Any]) -> ProviderProfile:
        validate_k_range(data["default_k"], data["max_k"])
        return await self._providers.create(normalize_profile_data(data["kind"], data))

    async def update(self, profile_id: UUID, changes: dict[str, Any]) -> ProviderProfile:
        current = await _require(self._providers, profile_id)
        if changes.get("calibration") is not None:
            if current.kind != "embedding":
                raise InvalidInput(
                    "invalid_calibration", "Solo los perfiles de embeddings se calibran"
                )
            changes = {**changes, "calibration": parse_calibration(changes["calibration"])}
        validate_k_range(
            changes.get("default_k", current.default_k), changes.get("max_k", current.max_k)
        )
        changes_vector_space = any(
            k in changes and changes[k] != getattr(current, k) for k in VECTOR_SPACE_FIELDS
        )
        if (
            current.kind == "embedding"
            and changes_vector_space
            and await self._providers.in_use(profile_id)
        ):
            raise Conflict(
                "provider_in_use",
                "Hay documentos indexados con este perfil; cambiar el modelo, las dimensiones "
                "o los prefijos dejaría sus vectores inconsistentes. Creá un perfil nuevo.",
            )
        if current.kind in ("stt", "tts"):
            current_data = {k: v for k, v in vars(current).items() if k != "api_key"}
            merged = normalize_profile_data(current.kind, {**current_data, **changes})
            changes = {
                k: v
                for k, v in merged.items()
                if k not in ("id", "kind") and (k in changes or v != current_data[k])
            }
        return await self._providers.update(profile_id, changes)

    async def delete(self, profile_id: UUID) -> None:
        await self._providers.delete(profile_id)

    async def active_ids(self) -> dict[ProviderKind, UUID | None]:
        return await self._providers.active_ids()  # type: ignore[return-value]


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    detail: str
    dimensions: int | None = None  # embedding profiles
    modalities: list[str] | None = None  # embedding profiles: text + what it can also embed
    supports_tools: bool | None = None  # chat profiles
    transcript: str | None = None  # stt profiles: what the probe audio was transcribed to
    audio_bytes: int | None = None  # tts profiles: size of the synthesized probe
    caption: str | None = None  # vision profiles: what the model said about the probe image


class ProbeProvider:
    def __init__(self, providers: ProviderRepository, factory: ProviderFactory):
        self._providers = providers
        self._factory = factory

    async def execute(self, profile_id: UUID) -> ProbeResult:
        """Probe a profile and persist what was detected (dimensions / supports_tools)."""
        profile = await _require(self._providers, profile_id)
        try:
            if profile.kind == "embedding":
                return await self._probe_embedding(profile)
            if profile.kind == "stt":
                return await self._probe_stt(profile)
            if profile.kind == "tts":
                return await self._probe_tts(profile)
            if profile.kind == "vision":
                return await self._probe_vision(profile)
            return await self._probe_chat(profile)
        except ProviderError as exc:
            return ProbeResult(False, str(exc))

    async def _probe_embedding(self, profile: ProviderProfile) -> ProbeResult:
        vectors = await self._factory.embedder(profile).embed([PROBE_PING])
        dims = len(vectors[0])
        if profile.dimensions is None:
            await self._providers.update(profile.id, {"dimensions": dims})
        modalities = await self._detect_modalities(profile)
        if modalities != profile.modalities:
            await self._providers.update(profile.id, {"modalities": modalities})
        return ProbeResult(True, "embedding ok", dimensions=dims, modalities=modalities)

    async def _detect_modalities(self, profile: ProviderProfile) -> list[str]:
        """Try one tiny image and one tiny WAV: a server without the mmproj just rejects them."""
        embedder = self._factory.multimodal_embedder(profile)
        found = ["text"]
        for name, embed, sample in (
            ("image", embedder.embed_image, probe_png()),
            ("audio", embedder.embed_audio, probe_wav()),
        ):
            try:
                await embed(sample)
            except (ProviderError, InvalidInput):
                continue
            found.append(name)
        return found

    async def _probe_stt(self, profile: ProviderProfile) -> ProbeResult:
        """Send ~1 s of generated audio: proves the endpoint, model and protocol accept audio
        (the transcript of a tone is meaningless, so only success is checked)."""
        text = await self._factory.transcriber(profile).transcribe(probe_wav())
        return ProbeResult(True, "stt ok", transcript=text)

    async def _probe_tts(self, profile: ProviderProfile) -> ProbeResult:
        clip = await self._factory.speaker(profile).speak(PROBE_SPEECH)
        return ProbeResult(True, "tts ok", audio_bytes=len(clip.data))

    async def _probe_vision(self, profile: ProviderProfile) -> ProbeResult:
        """Send a small generated image: proves the endpoint takes image parts and answers with
        non-empty `content` (an empty one raises EmptyCompletion, a ProviderError)."""
        text = await self._factory.vision(profile).describe(probe_png(), PROBE_VISION)
        return ProbeResult(True, "vision ok", caption=text[:PROBE_CAPTION_CHARS])

    async def _probe_chat(self, profile: ProviderProfile) -> ProbeResult:
        chat = self._factory.chat(profile)
        await chat.complete([{"role": "user", "content": PROBE_PING}])
        messages = [{"role": "user", "content": PROBE_TOOL_REQUEST}]
        supports = False
        for choice in ("required", "auto"):
            try:
                result = await chat.complete(messages, tools=[PING_TOOL], tool_choice=choice)
            except ToolsUnsupported:
                continue
            supports = bool(result.tool_calls)
            break
        await self._providers.update(profile.id, {"supports_tools": supports})
        return ProbeResult(True, "chat ok", supports_tools=supports)


class SetActiveProfiles:
    def __init__(self, providers: ProviderRepository):
        self._providers = providers

    async def execute(
        self, changes: Mapping[ProviderKind, UUID | None]
    ) -> dict[ProviderKind, UUID | None]:
        """Activate (or, with None, deactivate) one profile per kind; returns the active ids."""
        for kind, profile_id in changes.items():
            if profile_id is not None:
                profile = await _require(self._providers, profile_id)
                if profile.kind != kind:
                    raise InvalidInput(
                        "kind_mismatch", f"El perfil {profile_id} no es de tipo {kind}"
                    )
            await self._providers.set_active(kind, profile_id)
        return await self._providers.active_ids()  # type: ignore[return-value]


class SeedDefaults:
    def __init__(
        self, providers: ProviderRepository, embedding_base_url: str, tts_base_url: str = ""
    ):
        self._providers = providers
        self._base_url = embedding_base_url
        self._tts_base_url = tts_base_url

    async def execute(self) -> None:
        """Plan §7.2: create the local embedding profile only if no embedding profile exists.
        Same for the local TTS profile. There is no STT default: the user adds it from the UI."""
        if not await self._providers.list("embedding"):
            profile = await self._providers.create({**LOCAL_EMBEDDING, "base_url": self._base_url})
            await self._providers.set_active("embedding", profile.id)
        if self._tts_base_url and not await self._providers.list("tts"):
            profile = await self._providers.create({**LOCAL_TTS, "base_url": self._tts_base_url})
            await self._providers.set_active("tts", profile.id)


@dataclass(frozen=True)
class ProviderHealth:
    name: str
    reachable: bool


class CheckHealth:
    def __init__(self, database: Database, providers: ProviderRepository, factory: ProviderFactory):
        self._database = database
        self._providers = providers
        self._factory = factory

    async def execute(self) -> dict[ProviderKind, ProviderHealth | None]:
        """Active profile and reachability per kind; ServiceUnavailable if the DB is down."""
        try:
            await self._database.ping()
        except Exception as exc:
            raise ServiceUnavailable(
                "database_unavailable", "La base de datos no está disponible"
            ) from exc
        health: dict[ProviderKind, ProviderHealth | None] = {}
        for kind in ("chat", "embedding"):
            profile = await self._providers.active(kind)
            health[kind] = (
                None
                if profile is None
                else ProviderHealth(profile.name, await self._factory.reachable(profile))
            )
        return health
