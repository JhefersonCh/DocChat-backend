import uuid
from typing import Annotated

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from chat_pdf.application.inspector import ListChunks

router = APIRouter(prefix="/api/v1")

MAX_LIMIT = 500


class ChunkItem(BaseModel):
    chunk_id: uuid.UUID | None
    ordinal: int
    page_start: int | None
    page_end: int | None
    section: str | None
    chars: int
    content: str
    modality: str = "text"  # text | image | audio | frame
    origin: str | None = None  # embedded_image | figure | page | tile | whole_image | ...
    bbox: list[float] | None = None
    t_start: float | None = None  # seconds (audio window)
    t_end: float | None = None
    text_status: str | None = None  # media: pending | ready | failed (the description)
    has_thumb: bool = False  # GET /segments/{chunk_id}/thumb exists


class ChunksOut(BaseModel):
    total: int
    offset: int
    items: list[ChunkItem]


@router.get("/documents/{document_id}/chunks", response_model=ChunksOut)
async def list_chunks(
    document_id: uuid.UUID,
    request: Request,
    page: Annotated[int | None, Query(ge=1)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 200,
):
    use_case: ListChunks = request.app.state.use_cases.list_chunks
    total, chunks = await use_case.execute(document_id, page, offset, limit)
    items = [
        ChunkItem(
            chunk_id=c.id,
            ordinal=c.ordinal,
            page_start=c.page_start,
            page_end=c.page_end,
            section=c.section,
            chars=len(c.content),
            content=c.content,
            modality=c.modality,
            origin=c.origin,
            bbox=c.bbox,
            t_start=c.t_start,
            t_end=c.t_end,
            text_status=c.text_status,
            has_thumb=c.modality != "text" and c.asset_key is not None,
        )
        for c in chunks
    ]
    return ChunksOut(total=total, offset=offset, items=items)
