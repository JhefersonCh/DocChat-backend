"""In-memory adapters implementing the application ports. No rollback in FakeUnitOfWork."""

from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

from fakes_media import FakeMediaExtractor, FakeMultimodalEmbedder, FakeVision, vision_profile  # noqa: F401
from fakes_store import (  # noqa: F401
    FakeChatRepository,
    FakeChunkStore,
    FakeDb,
    FakeDocumentRepository,
    FakeUnitOfWork,
)

from chat_pdf.application.ports import AudioClip, ChatResult
from chat_pdf.domain.errors import InvalidInput, StreamingUnsupported
from chat_pdf.domain.models import Chunk, Page
from chat_pdf.domain.providers import ProviderProfile

CORPUS = [Chunk(i, f"chunk {i}", 1, 1, 0, 1, id=uuid4()) for i in range(30)]


def chat_profile(**kw) -> ProviderProfile:
    return ProviderProfile(uuid4(), "chat", "chat", "http://chat", "m", **kw)


def stt_profile(**kw) -> ProviderProfile:
    kw = {"protocol": "ollama", "language": "es", **kw}
    return ProviderProfile(uuid4(), "stt", "stt", "http://stt", "gemma4:e2b", **kw)


def tts_profile(**kw) -> ProviderProfile:
    kw = {"voice": "F2", "language": "es", **kw}
    return ProviderProfile(uuid4(), "tts", "tts", "http://tts", "supertonic-3", **kw)


def embedding_profile(query_prefix: str | None = None) -> ProviderProfile:
    return ProviderProfile(
        uuid4(), "embedding", "emb", "http://emb", "m", query_prefix=query_prefix
    )


# --- Providers --------------------------------------------------------------------------------


class FakeEmbedder:
    def __init__(self, error: Exception | None = None):
        self.texts: list[str] = []
        self.error = error

    async def embed(self, texts):
        if self.error:
            raise self.error
        self.texts += texts
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeChat:
    """Scripted chat: `rounds` answer complete(tools=...) calls in order, then no tool calls."""

    def __init__(
        self,
        rounds=(),
        stream_unsupported=False,
        stream_error: Exception | None = None,
        answer: str = "full answer [1]",
    ):
        self.answer = answer
        self.completions = 0  # complete() calls without tools
        self.rounds = list(rounds)
        self.stream_unsupported = stream_unsupported
        self.stream_error = stream_error
        self.tool_rounds = 0
        self.tool_systems: list[str] = []  # system message of each tool round
        self.tool_messages: list[list[dict]] = []  # full messages of each tool round
        self.tool_specs: list[list[dict]] = []  # tools offered in each tool round
        self.streamed: list[list[dict]] = []  # messages of each stream() call

    async def complete(self, messages, tools=None, tool_choice=None):
        if tools is None:
            self.completions += 1
            return ChatResult(self.answer)
        self.tool_rounds += 1
        self.tool_systems.append(messages[0]["content"])
        self.tool_messages.append(list(messages))
        self.tool_specs.append(tools)
        return self.rounds.pop(0) if self.rounds else ChatResult("no tools needed")

    async def stream(self, messages):
        self.streamed.append(messages)
        if self.stream_unsupported:
            raise StreamingUnsupported("no streaming")
        yield "streamed [1]"
        if self.stream_error:
            raise self.stream_error


class FakeTranscriber:
    def __init__(self, text: str = "hola mundo", error: Exception | None = None):
        self.text = text
        self.error = error
        self.received: list[bytes] = []

    async def transcribe(self, wav):
        if self.error:
            raise self.error
        self.received.append(wav)
        return self.text


class FakeSpeaker:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.spoken: list[str] = []

    async def speak(self, text):
        if self.error:
            raise self.error
        self.spoken.append(text)
        return AudioClip(b"RIFFfake", "audio/wav")


class FakeProviderFactory:
    def __init__(
        self,
        chat: FakeChat | None = None,
        embedder: FakeEmbedder | None = None,
        transcriber: FakeTranscriber | None = None,
        speaker: FakeSpeaker | None = None,
        multimodal: FakeMultimodalEmbedder | None = None,
        vision: FakeVision | None = None,
    ):
        self._chat = chat or FakeChat()
        self.multimodal = multimodal or FakeMultimodalEmbedder()
        self.embedding = embedder or FakeEmbedder()
        self.stt = transcriber or FakeTranscriber()
        self.tts = speaker or FakeSpeaker()
        self.vision_model = vision or FakeVision()
        self.embedder_profiles: list[UUID] = []

    def embedder(self, profile):
        self.embedder_profiles.append(profile.id)
        return self.embedding

    def multimodal_embedder(self, profile):
        return self.multimodal

    def chat(self, profile):
        return self._chat

    def transcriber(self, profile):
        return self.stt

    def speaker(self, profile):
        return self.tts

    def vision(self, profile):
        return self.vision_model

    async def reachable(self, profile) -> bool:
        return True


class FakeProviderRepository:
    def __init__(
        self, *profiles: ProviderProfile, active: dict[str, ProviderProfile] | None = None
    ):
        self.profiles = {p.id: p for p in profiles}
        self._active = active or {}
        self.used: set[UUID] = set()  # profiles referenced by documents

    async def get(self, profile_id):
        return self.profiles.get(profile_id)

    async def create(self, data):
        profile = ProviderProfile(uuid4(), **data)
        self.profiles[profile.id] = profile
        return profile

    async def update(self, profile_id, changes):
        self.profiles[profile_id] = replace(self.profiles[profile_id], **changes)
        return self.profiles[profile_id]

    async def delete(self, profile_id):
        del self.profiles[profile_id]

    async def in_use(self, profile_id):
        return profile_id in self.used

    async def active_ids(self):
        return {kind: p.id for kind, p in self._active.items()}

    async def active(self, kind):
        return self._active.get(kind)

    async def set_active(self, kind, profile_id):
        self._active[kind] = self.profiles[profile_id] if profile_id else None

    async def list(self, kind=None):
        return [p for p in self.profiles.values() if kind in (None, p.kind)]


# --- Files ------------------------------------------------------------------------------------


class FakeFileStorage:
    def __init__(self):
        self.files: dict[UUID, bytes] = {}
        self.exts: dict[UUID, str] = {}
        self.assets: dict[tuple[UUID, str, str], bytes] = {}

    async def save(self, document_id, content, ext=""):
        self.files[document_id] = b"".join([block async for block in content])
        self.exts[document_id] = ext

    async def delete(self, document_id):
        self.files.pop(document_id, None)

    def path(self, document_id) -> Path:
        return Path(f"/fake/{document_id}")

    def playable_path(self, document_id) -> Path:
        return Path(f"/fake/{document_id}/play.mp4")

    async def save_asset(self, document_id, kind, name, data):
        self.assets[(document_id, kind, name)] = data

    async def clear_assets(self, document_id):
        self.assets = {k: v for k, v in self.assets.items() if k[0] != document_id}

    def asset_path(self, document_id, kind, name) -> Path:
        if name != Path(name).name:
            raise InvalidInput("invalid_asset_name", "bad name")
        return Path(f"/fake/{document_id}/{kind}/{name}")


class IdentityReranker:
    async def rerank(self, query, chunks):
        return chunks


class FakeLoader:
    def __init__(self, *pages: Page):
        self._pages = list(pages)

    def load(self, path) -> list[Page]:
        return self._pages
