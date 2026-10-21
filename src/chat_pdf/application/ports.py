"""All ports (interfaces the application needs). Adapters implement them."""

from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Protocol, Self
from uuid import UUID

from chat_pdf.domain.figures import MediaRegion
from chat_pdf.domain.find_all import MediaMatch
from chat_pdf.domain.models import Chat, Chunk, Citation, Document, Message, Page, Search
from chat_pdf.domain.providers import ProviderKind, ProviderProfile

ChatMessage = dict[str, Any]  # OpenAI chat message shape
AssetKind = Literal["assets", "thumbs"]  # full-size crop/frame (<= 1024 px) vs 256 px thumbnail


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]  # already JSON-decoded; {} if the model sent invalid JSON


@dataclass(frozen=True)
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


# --- Providers (LLM / embeddings) -------------------------------------------------------------


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class MultimodalEmbedder(Protocol):
    """Embeds one image or one WAV into the same space as the text embedder (plan §2.1)."""

    async def embed_image(self, image: bytes, context: str | None = None) -> list[float]:
        """PNG/JPEG bytes (any size; resized for the model) -> L2-normalized vector. `context`
        is a short text embedded together with the picture in ONE input (its surroundings)."""
        ...

    async def embed_audio(self, wav: bytes) -> list[float]:
        """16 kHz mono WAV bytes -> L2-normalized vector."""
        ...

    async def embed_clip(self, frames: list[bytes]) -> list[float]:
        """Several consecutive frames as ONE input (video as frames) -> L2-normalized vector."""
        ...


class ChatModel(Protocol):
    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> ChatResult: ...

    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]: ...


@dataclass(frozen=True)
class AudioClip:
    data: bytes
    mime: str  # e.g. audio/wav


class Transcriber(Protocol):
    async def transcribe(self, wav: bytes) -> str:
        """Text of a 16 kHz mono WAV ("" when there is no speech)."""
        ...


class Speaker(Protocol):
    async def speak(self, text: str) -> AudioClip: ...


class AudioConverter(Protocol):
    async def to_wav16k(self, audio: bytes) -> bytes:
        """Any container/codec -> 16 kHz mono 16-bit WAV; InvalidInput if undecodable."""
        ...


class VisionModel(Protocol):
    async def describe(self, image: bytes, prompt: str) -> str:
        """The model's text answer about one image (PNG/JPEG/WebP bytes). Only `content` is
        read, never the model's reasoning. EmptyCompletion (a ProviderError) if it is empty."""
        ...


class ProviderFactory(Protocol):
    def embedder(self, profile: ProviderProfile) -> Embedder: ...

    def multimodal_embedder(self, profile: ProviderProfile) -> MultimodalEmbedder: ...

    def chat(self, profile: ProviderProfile) -> ChatModel: ...

    def transcriber(self, profile: ProviderProfile) -> Transcriber: ...

    def speaker(self, profile: ProviderProfile) -> Speaker: ...

    def vision(self, profile: ProviderProfile) -> VisionModel: ...

    async def reachable(self, profile: ProviderProfile) -> bool:
        """Cheap liveness check of the provider's endpoint."""
        ...


# --- Indexing ---------------------------------------------------------------------------------


class Loader(Protocol):
    """Reads a file of one MIME type into pages (sync: callers run it in a thread)."""

    def load(self, path: Path) -> list[Page]: ...


@dataclass(frozen=True)
class MediaPlan:
    regions: list[MediaRegion]
    width: int | None = None  # standalone images
    height: int | None = None


@dataclass(frozen=True)
class RenderedMedia:
    """One region rendered once, in the three sizes the pipeline needs."""

    asset: bytes  # WebP, longest side <= 1024 px (lightbox, vision model)
    thumb: bytes  # WebP, 256 px
    embed: bytes  # PNG, <= 384 px (native image embedding)
    sha: str  # content hash of `embed`: identical pictures share one embedding


class MediaExtractor(Protocol):
    """Finds the image regions of a file and renders them (sync: callers run it in a thread)."""

    def plan(self, path: Path, page_text_chars: Mapping[int, int]) -> MediaPlan:
        """Which regions become segments. `page_text_chars`: OCR/text length per 1-based page
        (decides tiling of scanned pages); no pixels are produced here."""
        ...

    def render(self, path: Path, region: MediaRegion) -> RenderedMedia: ...


class ChunkStore(Protocol):
    async def replace(
        self, document_id: UUID, chunks: list[Chunk], vectors: list[list[float] | None]
    ) -> None:
        """Store the document's chunks, dropping any it had (re-index). A None vector is a
        segment without a native embedding (audio window the server could not embed)."""
        ...

    async def for_document(self, document_id: UUID) -> list[Chunk]:
        """All the document's chunks in reading order."""
        ...

    async def text_vectors(self, document_ids: list[UUID]) -> dict[UUID, list[float]]:
        """Text-space vector of every chunk of those documents that has one (text chunks and
        captions / transcripts of media; native-only vectors are skipped), by chunk id."""
        ...

    async def pending_captions(
        self, document_id: UUID, modalities: tuple[str, ...] | None = None
    ) -> list[Chunk]:
        """Media segments (of those modalities, default all) still waiting for their
        description / transcript, in reading order."""
        ...

    async def requeue_failed_captions(
        self, document_id: UUID, modalities: tuple[str, ...] | None = None
    ) -> int:
        """Failed descriptions go back to `pending` (a retry). Returns how many."""
        ...

    async def save_caption(
        self, chunk_id: UUID, content: str, index_text: str, vector: list[float]
    ) -> None:
        """Store a segment's description (content + indexed text + text-space embedding)."""
        ...

    async def fail_caption(self, chunk_id: UUID) -> None: ...

    async def set_context(self, chunk_id: UUID, text: str) -> None:
        """Text of an audio context parent (its children's transcripts); never searchable."""
        ...

    async def next_ordinal(self, document_id: UUID) -> int:
        """Ordinal for the next appended segment."""
        ...

    async def add_segment(
        self,
        document_id: UUID,
        chunk: Chunk,
        vector: list[float] | None,
        text_vector: list[float] | None = None,
    ) -> None:
        """Append a child segment (a region found by the vision model) after the ingest."""
        ...

    async def delete_children(self, parent_id: UUID, origin: str) -> list[str]:
        """Drop a segment's children of that origin; returns their asset file names."""
        ...

    async def skip_caption(self, chunk_id: UUID) -> None:
        """Nothing to index for the segment (an audio window without speech): `skipped`."""
        ...

    async def count_media(
        self, document_id: UUID, modalities: tuple[str, ...] | None = None
    ) -> int:
        """How many media segments (non-text, of those modalities if given) the document has."""
        ...

    async def asset_of(self, chunk_id: UUID) -> tuple[UUID, str | None] | None:
        """(document id, asset file name or None) of a segment; None if it does not exist."""
        ...

    async def verdicts(self, chunk_ids: list[UUID], query_key: str) -> dict[UUID, bool]:
        """Cached vision-model yes/no of segments for a normalized query (find_all)."""
        ...

    async def save_verdict(self, chunk_id: UUID, query_key: str, verdict: bool) -> None: ...


@dataclass(frozen=True)
class Candidates:
    """Independent rankings of the chunks of one or more documents, best first. Each one is a
    top-n within its own space: cosines are never compared across modalities (plan D6)."""

    by_vector: list[Chunk]  # text-space cosine (text chunks + descriptions / transcripts)
    by_text: list[Chunk]  # full-text match of the query words
    by_image: list[Chunk] = field(default_factory=list)  # native image / frame vectors
    by_audio: list[Chunk] = field(default_factory=list)  # native audio vectors


class Retriever(Protocol):
    async def candidates(
        self,
        document_ids: list[UUID],
        query: str,
        vector: list[float],
        n: int,
        max_page: int | None = None,
        modalities: tuple[str, ...] | None = None,
    ) -> Candidates:
        """Top-`n` of each ranking (the caller fuses them), over the chunks of all `document_ids`:
        text vector, full text, and the native image / audio vectors of the same query embedding.

        With `max_page`, only chunks that start on or before that page (or have no page). With
        `modalities`, only segments of those modalities (and only the native rankings they need).
        """
        ...

    async def media_matches(
        self,
        document_ids: list[UUID],
        query: str,
        vector: list[float],
        modalities: tuple[str, ...],
        max_page: int | None = None,
    ) -> list[MediaMatch]:
        """EVERY media segment of those modalities with its native and description cosines and
        whether its description matched the query words (exhaustive search: no top-n cut)."""
        ...


class Reranker(Protocol):
    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Reorder fused candidates by relevance to `query` (identity until one is configured)."""
        ...


# --- Repositories -----------------------------------------------------------------------------


class ProviderRepository(Protocol):
    async def get(self, profile_id: UUID) -> ProviderProfile | None: ...

    async def create(self, data: dict[str, Any]) -> ProviderProfile: ...

    async def update(self, profile_id: UUID, changes: dict[str, Any]) -> ProviderProfile: ...

    async def delete(self, profile_id: UUID) -> None: ...

    async def in_use(self, profile_id: UUID) -> bool:
        """True if any document was indexed with this profile (the same FK that blocks delete)."""
        ...

    async def active_ids(self) -> dict[str, UUID | None]: ...

    async def active(self, kind: ProviderKind) -> ProviderProfile | None: ...

    async def set_active(self, kind: ProviderKind, profile_id: UUID | None) -> None: ...

    # Last on purpose: a `list` method shadows the builtin in annotations declared after it.
    async def list(self, kind: ProviderKind | None = None) -> list[ProviderProfile]: ...


class ChatRepository(Protocol):
    async def create(self) -> Chat: ...

    async def get(self, chat_id: UUID) -> Chat | None: ...

    async def delete(self, chat_id: UUID) -> list[UUID]:
        """Delete chat + collection. Returns the document ids whose files must be removed."""
        ...

    async def lock(self, chat_id: UUID) -> None:
        """Row-lock the chat until the transaction ends (serializes writers of the same chat)."""
        ...

    async def set_title(self, chat_id: UUID, title: str) -> None: ...

    async def messages(self, chat_id: UUID) -> list[Message]: ...

    async def history(self, chat_id: UUID, limit: int) -> list[Message]: ...

    async def set_suggestions(self, message_id: UUID, questions: list[str]) -> None: ...

    async def set_verification(self, message_id: UUID, verification: dict) -> None: ...

    async def add_exchange(
        self,
        chat_id: UUID,
        question: str,
        answer: str,
        citations: list[Citation],
        searches: list[Search],
        weak_evidence: bool = False,
        persona: str | None = None,
    ) -> UUID: ...

    async def update_settings(self, chat_id: UUID, changes: dict[str, Any]) -> Chat:
        """Apply the style / custom_instructions / length `changes` and return the chat."""
        ...

    # Last on purpose: a `list` method shadows the builtin in annotations declared after it.
    async def list(self) -> list[Chat]: ...


class DocumentRepository(Protocol):
    async def create(
        self,
        document_id: UUID,
        chat_id: UUID,
        filename: str,
        mime: str,
        sha256: str,
        embedding_profile_id: UUID,
        kind: str = "pdf",
    ) -> None: ...

    async def get(self, document_id: UUID) -> Document | None: ...

    async def list_for_chat(self, chat_id: UUID) -> list[Document]:
        """Every source of the chat's collection, in the order they were added (S1..Sn)."""
        ...

    async def delete(self, document_id: UUID) -> None: ...

    async def set_status(
        self,
        document_id: UUID,
        status: str,
        error: str | None = None,
        page_count: int | None = None,
        summary_status: str | None = None,
        enrich_status: str | None = None,
        width: int | None = None,
        height: int | None = None,
        duration_s: float | None = None,
    ) -> None:
        """Set the status (clearing progress / stage / cancel flag); None args keep the value."""
        ...

    async def set_progress(
        self, document_id: UUID, done: int, total: int, stage: str | None = None
    ) -> bool:
        """Record done/total steps of `stage` (kept when None). Returns True if a cancel was
        requested meanwhile."""
        ...

    async def request_cancel(self, document_id: UUID) -> bool:
        """Flag a `processing` document, or one whose descriptions are running, for
        cancellation. False if neither."""
        ...

    async def set_enrich(
        self, document_id: UUID, status: str, error: str | None = None, stage: str = "describing"
    ) -> None:
        """Set the descriptions / transcripts stage's status. `running` starts `stage`
        (describing | transcribing); any other status ends it (progress, stage, cancel cleared)."""
        ...

    async def mark_enrich_pending(self, document_id: UUID) -> bool:
        """Atomically move a `ready` document whose descriptions are partial / failed / skipped
        to `pending` (retry); False otherwise."""
        ...

    async def set_summary(
        self,
        document_id: UUID,
        status: str,
        summary: str | None = None,
        questions: list[str] | None = None,
    ) -> None:
        """Set the overview status; summary/questions are kept when None."""
        ...

    async def mark_summary_pending(self, document_id: UUID) -> bool:
        """Atomically move a `ready` document without a pending overview to `pending`."""
        ...

    async def mark_processing(self, document_id: UUID) -> bool:
        """Atomically move a `ready`/`failed` document to `processing` (dropping its overview);
        False if it was busy."""
        ...

    async def fail_unfinished(self, error: str) -> None:
        """Mark every `pending` / `processing` document, and every pending overview, as `failed`."""
        ...


# --- Infrastructure ---------------------------------------------------------------------------


class FileStorage(Protocol):
    """Where a source's files live, addressed by document id (never by the original filename).

    One folder per source: the original plus derived assets (crops, frames) and thumbnails."""

    async def save(self, document_id: UUID, content: AsyncIterator[bytes], ext: str = "") -> None:
        """Store the original (`ext` like ".pdf", kept so players/loaders can sniff it)."""
        ...

    async def delete(self, document_id: UUID) -> None:
        """Remove the source's files (original and assets); a missing one is not an error."""
        ...

    def path(self, document_id: UUID) -> Path:
        """The original file (`{id}/original.*`, or the flat legacy file `{id}`)."""
        ...

    def playable_path(self, document_id: UUID) -> Path:
        """Where the browser-playable transcode of a video lives (it may not exist)."""
        ...

    async def save_asset(
        self, document_id: UUID, kind: AssetKind, name: str, data: bytes
    ) -> None: ...

    def asset_path(self, document_id: UUID, kind: AssetKind, name: str) -> Path:
        """Derived file; InvalidInput if `name` is not a plain file name."""
        ...

    async def clear_assets(self, document_id: UUID) -> None:
        """Drop every derived file (a re-index regenerates them); the original stays."""
        ...


class Database(Protocol):
    async def ping(self) -> None:
        """Raise if the database cannot answer."""
        ...


class UnitOfWork(Protocol):
    """One connection, one transaction: commit on clean exit, rollback on exception."""

    chats: ChatRepository
    docs: DocumentRepository
    chunks: ChunkStore

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


UnitOfWorkFactory = Callable[[], UnitOfWork]
