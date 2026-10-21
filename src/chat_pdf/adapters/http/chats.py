import logging
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from chat_pdf.adapters.http.deps import (
    get_ask_question,
    get_cancel_ingest,
    get_create_chat,
    get_delete_chat,
    get_generate_overview,
    get_get_chat,
    get_get_document,
    get_ingest_document,
    get_list_chats,
    get_list_messages,
    get_open_document_file,
    get_reindex_document,
    get_remove_source,
    get_request_summary,
    get_search_document,
    get_suggest_follow_ups,
    get_update_chat_settings,
    get_upload_document,
    get_verify_answer,
)
from chat_pdf.adapters.http.errors import error_body
from chat_pdf.adapters.http.schemas import (
    AskBody,
    CancelOut,
    ChatDetailOut,
    ChatOut,
    ChatSettingsPatch,
    DocumentOut,
    DocumentSearchOut,
    MessageOut,
    ReindexOut,
    SuggestionsOut,
    SummaryOut,
    UploadOut,
    VerificationOut,
)
from chat_pdf.adapters.http.sse import sse
from chat_pdf.application.ask import AskQuestion
from chat_pdf.application.chats import (
    CreateChat,
    DeleteChat,
    GetChat,
    ListChats,
    ListMessages,
    RemoveSource,
    UpdateChatSettings,
)
from chat_pdf.application.documents import (
    CancelIngest,
    GetDocument,
    IngestDocument,
    OpenDocumentFile,
    ReindexDocument,
    UploadDocument,
)
from chat_pdf.application.generation import GenerateOverview, RequestSummary, SuggestFollowUps
from chat_pdf.application.search import MAX_HITS, SearchDocument
from chat_pdf.application.verification import VerifyAnswer

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

UPLOAD_READ_BYTES = 1024 * 1024


@router.post("/chats", status_code=201, response_model=ChatOut)
async def create_chat(use_case: Annotated[CreateChat, Depends(get_create_chat)]):
    return await use_case.execute()


@router.get("/chats", response_model=list[ChatOut])
async def list_chats(use_case: Annotated[ListChats, Depends(get_list_chats)]):
    return await use_case.execute()


@router.get("/chats/{chat_id}", response_model=ChatDetailOut)
async def get_chat(chat_id: uuid.UUID, use_case: Annotated[GetChat, Depends(get_get_chat)]):
    return ChatDetailOut.of(await use_case.execute(chat_id))


@router.patch("/chats/{chat_id}", response_model=ChatOut)
async def update_chat_settings(
    chat_id: uuid.UUID,
    body: ChatSettingsPatch,
    use_case: Annotated[UpdateChatSettings, Depends(get_update_chat_settings)],
):
    return await use_case.execute(chat_id, body.changes())


@router.delete("/chats/{chat_id}", status_code=204)
async def delete_chat(
    chat_id: uuid.UUID, use_case: Annotated[DeleteChat, Depends(get_delete_chat)]
):
    await use_case.execute(chat_id)


@router.get("/chats/{chat_id}/messages", response_model=list[MessageOut])
async def list_messages(
    chat_id: uuid.UUID, use_case: Annotated[ListMessages, Depends(get_list_messages)]
):
    return await use_case.execute(chat_id)


async def _blocks(file: UploadFile) -> AsyncIterator[bytes]:
    while block := await file.read(UPLOAD_READ_BYTES):
        yield block


@router.post("/chats/{chat_id}/sources", status_code=202, response_model=UploadOut)
@router.post("/chats/{chat_id}/documents", status_code=202, response_model=UploadOut)
async def upload_document(
    chat_id: uuid.UUID,
    file: UploadFile,
    background: BackgroundTasks,
    upload: Annotated[UploadDocument, Depends(get_upload_document)],
    ingest: Annotated[IngestDocument, Depends(get_ingest_document)],
):
    document_id = await upload.execute(
        chat_id, file.filename or "", file.content_type, _blocks(file)
    )
    background.add_task(ingest.execute, document_id)
    return UploadOut(document_id=document_id)


@router.delete("/chats/{chat_id}/sources/{document_id}", status_code=204)
async def remove_source(
    chat_id: uuid.UUID,
    document_id: uuid.UUID,
    use_case: Annotated[RemoveSource, Depends(get_remove_source)],
):
    """Deletes one source of the chat: the document, its chunks and its stored files."""
    await use_case.execute(chat_id, document_id)


@router.get("/documents/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: uuid.UUID, use_case: Annotated[GetDocument, Depends(get_get_document)]
):
    return await use_case.execute(document_id)


@router.get("/documents/{document_id}/search", response_model=DocumentSearchOut)
async def search_document(
    document_id: uuid.UUID,
    q: Annotated[str, Query(min_length=1, max_length=500)],
    use_case: Annotated[SearchDocument, Depends(get_search_document)],
    k: Annotated[int, Query(ge=1, le=MAX_HITS)] = 10,
):
    return DocumentSearchOut(query=q, hits=await use_case.execute(document_id, q, k))


@router.post("/documents/{document_id}/reindex", status_code=202, response_model=ReindexOut)
async def reindex_document(
    document_id: uuid.UUID,
    background: BackgroundTasks,
    reindex: Annotated[ReindexDocument, Depends(get_reindex_document)],
    ingest: Annotated[IngestDocument, Depends(get_ingest_document)],
):
    await reindex.execute(document_id)
    background.add_task(ingest.execute, document_id)
    return ReindexOut(document_id=document_id)


@router.post("/documents/{document_id}/cancel", status_code=202, response_model=CancelOut)
async def cancel_ingest(
    document_id: uuid.UUID, use_case: Annotated[CancelIngest, Depends(get_cancel_ingest)]
) -> CancelOut:
    await use_case.execute(document_id)
    return CancelOut(document_id=document_id)


@router.post("/documents/{document_id}/summary", status_code=202, response_model=SummaryOut)
async def summarize_document(
    document_id: uuid.UUID,
    background: BackgroundTasks,
    request: Annotated[RequestSummary, Depends(get_request_summary)],
    overview: Annotated[GenerateOverview, Depends(get_generate_overview)],
):
    await request.execute(document_id)
    background.add_task(overview.execute, document_id)
    return SummaryOut(document_id=document_id)


@router.get("/documents/{document_id}/file", response_class=FileResponse)
async def get_document_file(
    document_id: uuid.UUID,
    use_case: Annotated[OpenDocumentFile, Depends(get_open_document_file)],
    variant: Annotated[Literal["playable", "original"], Query()] = "playable",
):
    """The file (HTTP Range supported). A video that browsers cannot play was transcoded at ingest
    and that mp4 is served; `?variant=original` returns what was uploaded."""
    stored = await use_case.execute(document_id, variant)
    return FileResponse(
        stored.path,
        media_type=stored.mime,
        filename=stored.filename,  # Starlette emits filename*=utf-8'' for non-ASCII names
        content_disposition_type="inline",
    )


async def _event_stream(events: AsyncIterator[tuple[str, object]]) -> AsyncIterator[str]:
    try:
        async for name, data in events:
            yield sse(name, data)
    except Exception as exc:
        log.exception("ask failed")
        yield sse("error", error_body(exc))


@router.post("/chats/{chat_id}/messages")
async def ask(
    chat_id: uuid.UUID, body: AskBody, use_case: Annotated[AskQuestion, Depends(get_ask_question)]
):
    events = await use_case.execute(chat_id, body.content, body.max_page, body.source_ids)
    return StreamingResponse(
        _event_stream(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chats/{chat_id}/messages/{message_id}/suggestions", response_model=SuggestionsOut)
async def suggest_follow_ups(
    chat_id: uuid.UUID,
    message_id: uuid.UUID,
    use_case: Annotated[SuggestFollowUps, Depends(get_suggest_follow_ups)],
):
    return SuggestionsOut(questions=await use_case.execute(chat_id, message_id))


@router.post("/chats/{chat_id}/messages/{message_id}/verify", response_model=VerificationOut)
async def verify_answer(
    chat_id: uuid.UUID,
    message_id: uuid.UUID,
    use_case: Annotated[VerifyAnswer, Depends(get_verify_answer)],
):
    return await use_case.execute(chat_id, message_id)
