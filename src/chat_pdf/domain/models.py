from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from chat_pdf.domain.calibration import CalibrationTable, chunk_relevance


@dataclass(frozen=True)
class Page:
    number: int | None  # 1-based; None for formats without pages (TXT/MD)
    text: str
    # Heading lines found from layout (PDF font size). None: unknown -> text rules in chunking
    headings: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    content: str
    page_start: int | None
    page_end: int | None
    char_start: int | None  # offsets into the cleaned document text (pages joined by "\n\n")
    char_end: int | None  # None for media segments
    id: UUID | None = None  # set when loaded from storage
    section: str | None = None  # title of the heading the chunk falls under
    # `content` stays verbatim (the viewer highlights it in the PDF text layer); this is the
    # normalized text that is embedded and full-text indexed. Not loaded back by searches.
    index_text: str | None = None
    # Cosine similarity of the chunk to the query (retrieval only; None when unknown).
    similarity: float | None = None
    # Cosine of the NATIVE vector (image / audio) to the query: only comparable within the
    # segment's own modality, never against `similarity` (text space).
    native_similarity: float | None = None
    # Multimedia segments (plan §1.2). Defaults describe a plain text chunk.
    modality: str = "text"  # text | image | audio | frame
    origin: str | None = (
        None  # embedded_image | figure | page | tile | whole_image | window | frame
    )
    bbox: list[float] | None = None  # [x0, y0, x1, y1] normalized 0-1 over the page / image
    t_start: float | None = None  # seconds (audio / video)
    t_end: float | None = None
    asset_key: str | None = None  # derived file (crop / frame) in the source's folder
    media_sha: str | None = None  # content hash of the picture (repeated pictures share vectors)
    text_status: str | None = None  # pending | ready | failed | skipped: caption / transcript
    document_id: UUID | None = None  # source it belongs to (set by searches; lets N sources mix)
    # Segment hierarchy: a child (object region, grid tile, fine slice) points to its parent.
    parent_id: UUID | None = None
    region_label: str | None = None  # what the region shows ("colibrí"); None for whole images
    # Small-to-big (answer time only, never stored): the parent's description + page context.
    parent_text: str | None = field(default=None, compare=False)
    regions_text: str | None = field(default=None, compare=False)  # a picture's region labels
    # Ingest only (never stored): ordinal of the parent in the same batch; the store resolves it
    # to `parent_id` after inserting (ids do not exist before).
    parent_ordinal: int | None = field(default=None, compare=False)

    @property
    def has_thumb(self) -> bool:
        return self.modality != "text" and self.asset_key is not None


@dataclass(frozen=True)
class Document:
    id: UUID
    collection_id: UUID
    filename: str
    mime: str
    status: str  # pending | processing | ready | failed
    error: str | None
    page_count: int | None
    embedding_profile_id: UUID
    created_at: datetime
    summary: str | None = None  # overview generated after ingest
    suggested_questions: list[str] = field(default_factory=list)
    summary_status: str = "skipped"  # pending | ready | failed | skipped
    progress_done: int | None = None  # chunks embedded so far while `processing`
    progress_total: int | None = None
    cancel_requested: bool = False
    kind: str = "pdf"  # pdf | text | image | audio | video
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    stage: str | None = None  # extracting | embedding | describing | transcribing
    enrich_status: str = "skipped"  # pending | running | ready | partial | failed | skipped
    enrich_error: str | None = None  # why the descriptions stage ended partial / failed

    @property
    def progress(self) -> dict[str, int] | None:
        if self.progress_done is None or self.progress_total is None:
            return None
        return {"done": self.progress_done, "total": self.progress_total}


# Chunk modality -> what the viewer opens: a frame is a moment of a video.
CITATION_KIND = {"text": "text", "image": "image", "audio": "audio", "frame": "video"}


@dataclass(frozen=True)
class SourceRef:
    """A source (document) of the chat as the answer sees it: its alias S1..Sn, name and kind."""

    document_id: UUID
    alias: str
    filename: str
    kind: str


@dataclass(frozen=True)
class Source:
    """A retrieved chunk with its stable citation number `[n]`."""

    n: int
    chunk: Chunk
    document_id: UUID
    filename: str
    source_kind: str | None = None  # kind of the document the chunk belongs to
    alias: str | None = None  # S1..Sn of the document inside its chat
    labeled: bool = False  # the chat has several sources: the prompt names each one

    def cite(self, snippet_chars: int, table: CalibrationTable | None = None) -> "Citation":
        snippet = " ".join(self.chunk.content.split())[:snippet_chars]
        return Citation(
            self.n,
            self.document_id,
            self.filename,
            self.chunk.page_start,
            snippet,
            self.chunk.id,
            self.chunk.page_end,
            self.chunk.content,
            self.chunk.section,
            chunk_relevance(self.chunk, table),
            CITATION_KIND[self.chunk.modality],
            self.chunk.bbox,
            self.chunk.t_start,
            self.chunk.t_end,
            self.chunk.has_thumb,
            self.source_kind,
            self.alias,
            self.chunk.region_label,
        )


@dataclass(frozen=True)
class Citation:
    """What the user sees for a source: stable number, where it came from, a short snippet.

    The last four fields let the viewer trace the source (page range, chunk text, section);
    they are optional because citations persisted before they existed lack them.
    """

    n: int
    document_id: UUID
    filename: str
    page: int | None
    snippet: str
    chunk_id: UUID | None = None
    page_end: int | None = None
    content: str | None = None
    section: str | None = None
    score: float | None = None  # relevance in [0, 1] (query cosine similarity)
    # Multimedia contract (plan §1.5). Citations persisted before it have none of these: kind
    # defaults to "text", the rest to "absent".
    kind: str = "text"  # text | image | audio | video
    bbox: list[float] | None = None  # normalized region of the page / image
    t_start: float | None = None  # seconds
    t_end: float | None = None
    has_thumb: bool = False  # GET /segments/{chunk_id}/thumb exists
    source_kind: str | None = None  # kind of the source document (pdf | image | ...)
    source_alias: str | None = None  # S1..Sn: which source of the chat (API adds source_id/name)
    region_label: str | None = None  # object the region shows ("colibrí"), when a region segment


@dataclass(frozen=True)
class Search:
    """One retrieval the model made while answering."""

    query: str
    k: int
    # What it retrieved, for the viewer's X-ray: [{n, chunk_id, page_start, page_end, section,
    # score}]. Not part of equality: it is detail of the search, not its identity.
    found: list[dict] = field(default_factory=list, compare=False)
    panorama: bool = field(default=False, compare=False)
    # Phase 3: the modality hint the model gave, whether it was an exhaustive `find_all`, and how
    # many candidates each ranking contributed ({"vector", "text", "image", "audio"}).
    modality: str = field(default="auto", compare=False)
    exhaustive: bool = field(default=False, compare=False)
    rankings: dict[str, int] = field(default_factory=dict, compare=False)
    # Phase 7: which sources the search read: "all" or aliases ("S1,S3").
    source: str = field(default="all", compare=False)


@dataclass(frozen=True)
class Chat:
    id: UUID
    title: str
    created_at: datetime
    style: str = "normal"  # normal | simple | conciso | custom
    custom_instructions: str | None = None
    length: str = "normal"  # corta | normal | larga
    persona: str | None = None  # voice the answers are written in ("el zorro", "el autor")
    # Sources the user muted for this chat (multi-source, phase 7); new sources start active.
    disabled_sources: list[UUID] = field(default_factory=list)
    source_count: int = 0  # documents in the chat's collection (sidebar badge)
    project_id: UUID | None = None  # project the chat belongs to (its collection is a project)
    pinned_at: datetime | None = None  # set when pinned in the sidebar


@dataclass(frozen=True)
class Message:
    id: UUID
    role: str  # user | assistant
    content: str
    created_at: datetime
    citations: list[Citation] | None = None  # assistant messages only
    searches: list[Search] | None = None  # assistant messages only
    suggestions: list[str] | None = None  # follow-up questions, once requested
    weak_evidence: bool = False  # best source was below the relevance threshold
    verification: dict | None = None  # claim check ({sentences, claims}), once requested
    persona: str | None = None  # persona the answer was written in
