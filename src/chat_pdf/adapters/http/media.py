import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import FileResponse

from chat_pdf.adapters.http.deps import (
    get_enrich_document,
    get_get_storyboard,
    get_get_waveform,
    get_open_segment_asset,
    get_request_enrichment,
)
from chat_pdf.adapters.http.schemas import EnrichOut, StoryboardFrameOut, WaveformOut
from chat_pdf.application.enrich import EnrichDocument, RequestEnrichment
from chat_pdf.application.media import GetStoryboard, GetWaveform, OpenSegmentAsset
from chat_pdf.application.ports import AssetKind

router = APIRouter(prefix="/api/v1")

# Segment ids change on re-index (new rows), so a given URL's bytes never change.
IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


async def _asset(segment_id: uuid.UUID, kind: AssetKind, use_case: OpenSegmentAsset):
    stored = await use_case.execute(segment_id, kind)
    return FileResponse(stored.path, media_type=stored.mime, headers=IMMUTABLE)


@router.get("/segments/{segment_id}/thumb", response_class=FileResponse)
async def get_segment_thumb(
    segment_id: uuid.UUID, use_case: Annotated[OpenSegmentAsset, Depends(get_open_segment_asset)]
):
    return await _asset(segment_id, "thumbs", use_case)


@router.get("/segments/{segment_id}/asset", response_class=FileResponse)
async def get_segment_asset(
    segment_id: uuid.UUID, use_case: Annotated[OpenSegmentAsset, Depends(get_open_segment_asset)]
):
    return await _asset(segment_id, "assets", use_case)


@router.get("/documents/{document_id}/waveform", response_model=WaveformOut)
async def get_waveform(
    document_id: uuid.UUID, use_case: Annotated[GetWaveform, Depends(get_get_waveform)]
):
    """Peaks of an audio source for drawing its waveform (computed at ingest)."""
    return WaveformOut(**await use_case.execute(document_id))


@router.get("/documents/{document_id}/storyboard", response_model=list[StoryboardFrameOut])
async def get_storyboard(
    document_id: uuid.UUID, use_case: Annotated[GetStoryboard, Depends(get_get_storyboard)]
):
    """The frames of a video in time order, for a filmstrip: [{segment_id, t, t_end, thumb_url}]."""
    return [
        StoryboardFrameOut(
            segment_id=e.segment_id,
            t=e.t,
            t_end=e.t_end,
            thumb_url=f"/api/v1/segments/{e.segment_id}/thumb",
        )
        for e in await use_case.execute(document_id)
    ]


@router.post("/documents/{document_id}/enrich", status_code=202, response_model=EnrichOut)
async def enrich_document(
    document_id: uuid.UUID,
    background: BackgroundTasks,
    request: Annotated[RequestEnrichment, Depends(get_request_enrichment)],
    enrich: Annotated[EnrichDocument, Depends(get_enrich_document)],
):
    """(Re)start the image descriptions / audio transcripts: resumes what is pending or failed."""
    await request.execute(document_id)
    background.add_task(enrich.execute, document_id)
    return EnrichOut(document_id=document_id)
