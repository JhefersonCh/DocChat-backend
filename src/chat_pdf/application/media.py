import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from chat_pdf.application.ingest_audio import WAVEFORM_ASSET
from chat_pdf.application.ports import AssetKind, ChunkStore, DocumentRepository, FileStorage
from chat_pdf.domain.errors import NotFound
from chat_pdf.domain.video import StoryboardEntry, storyboard

ASSET_MIME = "image/webp"


@dataclass(frozen=True)
class StoredAsset:
    path: Path
    mime: str


class OpenSegmentAsset:
    """Resolves the derived image of a segment (crop / frame at <= 1024 px, or its 256 px thumb)."""

    def __init__(self, chunks: ChunkStore, storage: FileStorage):
        self._chunks = chunks
        self._storage = storage

    async def execute(self, segment_id: UUID, kind: AssetKind) -> StoredAsset:
        located = await self._chunks.asset_of(segment_id)
        if located is None:
            raise NotFound("segment_not_found", "Segmento no encontrado")
        document_id, asset_key = located
        path = self._storage.asset_path(document_id, kind, asset_key) if asset_key else None
        if path is None or not path.is_file():
            raise NotFound("asset_not_found", "El segmento no tiene imagen derivada")
        return StoredAsset(path, ASSET_MIME)


class GetWaveform:
    """The normalized peaks of an audio source, computed at ingest (for the player's waveform)."""

    def __init__(self, docs: DocumentRepository, storage: FileStorage):
        self._docs = docs
        self._storage = storage

    async def execute(self, document_id: UUID) -> dict:
        if await self._docs.get(document_id) is None:
            raise NotFound("document_not_found", "Documento no encontrado")
        path = self._storage.asset_path(document_id, "assets", WAVEFORM_ASSET)
        if not path.is_file():
            raise NotFound("waveform_not_found", "El documento no tiene forma de onda")
        return json.loads(path.read_text())


class GetStoryboard:
    """The frames of a video (time + thumbnail) for a filmstrip."""

    def __init__(self, docs: DocumentRepository, chunks: ChunkStore):
        self._docs = docs
        self._chunks = chunks

    async def execute(self, document_id: UUID) -> list[StoryboardEntry]:
        doc = await self._docs.get(document_id)
        if doc is None:
            raise NotFound("document_not_found", "Documento no encontrado")
        if doc.kind != "video":
            raise NotFound("storyboard_not_found", "El documento no es un video")
        return storyboard(await self._chunks.for_document(document_id))
