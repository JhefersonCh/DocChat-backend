"""Every request/response model of the HTTP API."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from chat_pdf.application.chats import ChatDetail
from chat_pdf.crypto import mask
from chat_pdf.domain.errors import InvalidInput
from chat_pdf.domain.models import Chat, Document
from chat_pdf.domain.providers import DEFAULT_K, MAX_K, MAX_TOOL_CALLS, ProviderProfile
from chat_pdf.domain.rules import validate_k_range
from chat_pdf.domain.sources import assign_aliases

# --- Chats, documents, messages ----------------------------------------------------------------


class ChatOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    created_at: datetime
    style: Literal["normal", "simple", "conciso", "custom"]
    custom_instructions: str | None
    length: Literal["corta", "normal", "larga"]
    persona: str | None = None
    disabled_sources: list[UUID] = []  # sources the user muted for this chat
    source_count: int = 0  # documents in the chat (sidebar badge)
    project_id: UUID | None = None  # set when the chat lives in a project
    pinned_at: datetime | None = None  # set when pinned in the sidebar

    @classmethod
    def of(cls, chat: Chat) -> "ChatOut":
        return cls.model_validate(chat)


class ProgressOut(BaseModel):
    done: int
    total: int


class ChatSettingsPatch(BaseModel):
    """Absent field = untouched; `custom_instructions: null` clears them."""

    style: Literal["normal", "simple", "conciso", "custom"] | None = None
    custom_instructions: str | None = Field(default=None, max_length=1000)
    length: Literal["corta", "normal", "larga"] | None = None
    persona: str | None = Field(default=None, max_length=120)
    # Sources the chat reads (the rest are muted); stored as `disabled_sources`.
    active_source_ids: list[UUID] | None = None
    pinned: bool | None = None  # pin / unpin in the sidebar

    def changes(self) -> dict[str, object]:
        given = self.model_dump(exclude_unset=True)
        if isinstance(given.get("persona"), str):
            given["persona"] = given["persona"].strip() or None  # blank clears the persona
        nullable = ("custom_instructions", "persona")
        return {k: v for k, v in given.items() if k in nullable or v is not None}


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    status: str
    error: str | None
    page_count: int | None
    created_at: datetime
    summary: str | None
    suggested_questions: list[str]
    summary_status: Literal["pending", "ready", "failed", "skipped"]
    progress: ProgressOut | None = None
    kind: Literal["pdf", "text", "image", "audio", "video"] = "pdf"
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    stage: Literal["extracting", "embedding", "describing", "transcribing"] | None = None
    enrich_status: Literal["pending", "running", "ready", "partial", "failed", "skipped"] = (
        "skipped"
    )
    enrich_error: str | None = None  # why the descriptions stage is partial / failed

    @classmethod
    def of(cls, doc: Document) -> "DocumentOut":
        return cls.model_validate(doc)


class SourceOut(DocumentOut):
    """A source of the chat: the document plus its alias (S1..Sn, by order added) and whether the
    chat currently reads it."""

    alias: str
    active: bool = True


class SourceSummaryOut(BaseModel):
    alias: str
    document_id: UUID
    filename: str
    kind: Literal["pdf", "text", "image", "audio", "video"]
    summary: str | None = None


class CollectionOverviewOut(BaseModel):
    """Aggregate of the per-source overviews, only for chats with 2 or more sources."""

    sources: list[SourceSummaryOut]
    summary: str | None
    suggested_questions: list[str]


class ChatDetailOut(ChatOut):
    # `document` = the first source (S1), kept for clients written when a chat had one document;
    # new clients read `sources`.
    document: DocumentOut | None
    sources: list[SourceOut] = []
    active_source_ids: list[UUID] = []
    overview: CollectionOverviewOut | None = None

    @classmethod
    def of(cls, detail: ChatDetail) -> "ChatDetailOut":  # type: ignore[override]
        document = DocumentOut.of(detail.document) if detail.document else None
        aliases = assign_aliases(detail.sources)
        muted = set(detail.chat.disabled_sources)
        sources = [
            SourceOut(
                **DocumentOut.of(d).model_dump(), alias=aliases[d.id], active=d.id not in muted
            )
            for d in detail.sources
        ]
        overview = (
            CollectionOverviewOut.model_validate(detail.overview, from_attributes=True)
            if detail.overview
            else None
        )
        return cls(
            **ChatOut.of(detail.chat).model_dump(),
            document=document,
            sources=sources,
            active_source_ids=[s.id for s in sources if s.active],
            overview=overview,
        )


class UploadOut(BaseModel):
    document_id: UUID
    status: Literal["pending"] = "pending"


class ReindexOut(BaseModel):
    document_id: UUID
    status: Literal["processing"] = "processing"


class CancelOut(BaseModel):
    document_id: UUID
    status: Literal["processing"] = "processing"


class EnrichOut(BaseModel):
    document_id: UUID
    enrich_status: Literal["pending"] = "pending"


class WaveformOut(BaseModel):
    peaks: list[float]  # ~800 bars, normalized 0-1
    duration_s: float


class StoryboardFrameOut(BaseModel):
    segment_id: UUID
    t: float  # second of the frame
    t_end: float | None = None  # until the next frame
    thumb_url: str  # GET it as is (256 px WebP)


class SummaryOut(BaseModel):
    document_id: UUID
    summary_status: Literal["pending"] = "pending"


class AskBody(BaseModel):
    content: str = Field(min_length=1)
    max_page: int | None = Field(default=None, ge=1)  # spoiler-safe: only chunks up to this page
    # Phase 7: read only these sources of the chat for this question (default: the active ones).
    source_ids: list[UUID] | None = None


class CitationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    n: int
    document_id: UUID
    filename: str
    page: int | None
    snippet: str
    chunk_id: UUID | None = None
    page_end: int | None = None
    content: str | None = None
    section: str | None = None
    score: float | None = None
    # Multimedia contract: absent in citations persisted before it (kind then means "text").
    kind: Literal["text", "image", "audio", "video"] = "text"
    bbox: list[float] | None = None  # [x0, y0, x1, y1] normalized 0-1 over the page / image
    t_start: float | None = None  # seconds
    t_end: float | None = None
    has_thumb: bool = False  # GET /api/v1/segments/{chunk_id}/thumb
    source_kind: Literal["pdf", "text", "image", "audio", "video"] | None = None
    source_alias: str | None = None  # S1..Sn; null in citations persisted before multi-source
    region_label: str | None = None  # object a region segment shows ("colibrí")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_id(self) -> UUID:
        return self.document_id

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_name(self) -> str:
        return self.filename


class FoundOut(BaseModel):
    n: int
    chunk_id: UUID | None = None
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    score: float | None = None
    # Multimedia contract (same meaning as in CitationOut); absent in older answers.
    kind: Literal["text", "image", "audio", "video"] = "text"
    bbox: list[float] | None = None
    t_start: float | None = None
    t_end: float | None = None
    has_thumb: bool = False
    document_id: UUID | None = None  # source of the segment (phase 7)
    source_alias: str | None = None
    region_label: str | None = None


class SearchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    query: str
    k: int
    found: list[FoundOut] = []  # empty in answers persisted before the X-ray
    panorama: bool = False
    modality: str = "auto"  # hint the model gave: auto | texto | imagen | audio | video
    exhaustive: bool = False  # True for find_all (every occurrence, verified)
    rankings: dict[str, int] = {}  # candidates per ranking: vector / text / image / audio
    source: str = "all"  # sources the search read: "all" or aliases ("S1,S3")


class ClaimOut(BaseModel):
    text: str
    sentence_index: int
    status: Literal["respaldada", "parcial", "sin_respaldo"]
    source_ns: list[int]
    quote: str | None = None
    reason: str = ""


class VerificationOut(BaseModel):
    sentences: list[str]
    claims: list[ClaimOut]


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    role: str
    content: str
    created_at: datetime
    citations: list[CitationOut] | None
    searches: list[SearchOut] | None
    suggestions: list[str] | None
    weak_evidence: bool = False
    verification: VerificationOut | None = None
    persona: str | None = None


class SearchHitOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: UUID | None
    page_start: int | None
    page_end: int | None
    section: str | None
    snippet: str
    content: str | None
    score: float | None
    kind: Literal["text", "image", "audio", "video"] = "text"
    bbox: list[float] | None = None
    t_start: float | None = None
    t_end: float | None = None
    has_thumb: bool = False


class DocumentSearchOut(BaseModel):
    query: str
    hits: list[SearchHitOut]


class SuggestionsOut(BaseModel):
    questions: list[str]


# --- Providers and settings --------------------------------------------------------------------


class ProviderIn(BaseModel):
    kind: Literal["chat", "embedding", "stt", "tts", "vision"]
    name: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key: str | None = None
    dimensions: int | None = Field(default=None, ge=1)
    query_prefix: str | None = None
    document_prefix: str | None = None
    supports_tools: bool = False
    supports_vision: bool = False  # chat
    default_k: int = Field(default=DEFAULT_K, ge=1)
    max_k: int = Field(default=MAX_K, ge=1)
    max_tool_calls: int = Field(default=MAX_TOOL_CALLS, ge=1)
    voice: str | None = None  # tts
    language: str | None = None  # stt / tts
    protocol: Literal["openai", "ollama"] | None = None  # stt

    @model_validator(mode="after")
    def _k_range(self):
        try:
            validate_k_range(self.default_k, self.max_k)
        except InvalidInput as exc:  # keep pydantic's 422 body shape for malformed requests
            raise ValueError(exc.message) from exc
        return self


class ProviderPatch(BaseModel):
    """`kind` is immutable. Absent field = untouched; `api_key: null` or "" clears the key."""

    name: str | None = Field(default=None, min_length=1)
    base_url: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)
    api_key: str | None = None
    dimensions: int | None = Field(default=None, ge=1)
    query_prefix: str | None = None
    document_prefix: str | None = None
    supports_tools: bool | None = None
    supports_vision: bool | None = None
    default_k: int | None = Field(default=None, ge=1)
    max_k: int | None = Field(default=None, ge=1)
    max_tool_calls: int | None = Field(default=None, ge=1)
    voice: str | None = None
    language: str | None = None
    protocol: Literal["openai", "ollama"] | None = None
    # embedding: {"text"|"image"|"audio": {"floor", "threshold"}}; null restores the defaults
    calibration: dict[str, dict[str, float]] | None = None


class ProviderOut(BaseModel):
    id: UUID
    kind: str
    name: str
    base_url: str
    model: str
    api_key_masked: str | None
    dimensions: int | None
    query_prefix: str | None
    document_prefix: str | None
    supports_tools: bool
    default_k: int
    max_k: int
    max_tool_calls: int
    voice: str | None = None
    language: str | None = None
    protocol: str | None = None
    supports_vision: bool = False
    modalities: list[str] = ["text"]  # embedding: filled by the probe
    calibration: dict[str, dict[str, float]] | None = None  # embedding: null = code defaults

    @classmethod
    def of(cls, p: ProviderProfile) -> "ProviderOut":
        data = {k: v for k, v in vars(p).items() if k != "api_key"}
        return cls(**data, api_key_masked=mask(p.api_key))


class ActiveSettings(BaseModel):
    chat_profile_id: UUID | None = None
    embedding_profile_id: UUID | None = None
    stt_profile_id: UUID | None = None
    tts_profile_id: UUID | None = None
    vision_profile_id: UUID | None = None


class ProbeOut(BaseModel):
    """Fields that do not apply to the probed kind are omitted from the response."""

    model_config = ConfigDict(from_attributes=True)

    ok: bool
    detail: str
    dimensions: int | None = None
    modalities: list[str] | None = None
    supports_tools: bool | None = None
    transcript: str | None = None
    audio_bytes: int | None = None
    caption: str | None = None


class ProviderHealthOut(BaseModel):
    name: str
    reachable: bool


class HealthOut(BaseModel):
    status: Literal["ok"] = "ok"
    providers: dict[str, ProviderHealthOut | None]
