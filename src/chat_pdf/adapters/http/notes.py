import uuid
from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from chat_pdf.adapters.http.schemas import CitationOut
from chat_pdf.application.notes import TITLE_MAX, DeleteNote, ListNotes, RenameNote, SaveNote

router = APIRouter(prefix="/api/v1")


class NoteBody(BaseModel):
    message_id: uuid.UUID
    title: str | None = Field(default=None, max_length=TITLE_MAX * 2)


class NotePatch(BaseModel):
    title: str = Field(max_length=TITLE_MAX * 2)


class NoteOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    message_id: uuid.UUID | None
    title: str
    content: str
    citations: list[CitationOut]
    created_at: datetime


def _use_cases(request: Request):
    return request.app.state.use_cases


@router.post("/chats/{chat_id}/notes", status_code=201, response_model=NoteOut)
async def save_note(chat_id: uuid.UUID, body: NoteBody, request: Request):
    use_case: SaveNote = _use_cases(request).save_note
    return await use_case.execute(chat_id, body.message_id, body.title)


@router.get("/chats/{chat_id}/notes", response_model=list[NoteOut])
async def list_notes(chat_id: uuid.UUID, request: Request):
    use_case: ListNotes = _use_cases(request).list_notes
    return await use_case.execute(chat_id)


@router.patch("/chats/{chat_id}/notes/{note_id}", response_model=NoteOut)
async def rename_note(chat_id: uuid.UUID, note_id: uuid.UUID, body: NotePatch, request: Request):
    use_case: RenameNote = _use_cases(request).rename_note
    return await use_case.execute(chat_id, note_id, body.title)


@router.delete("/chats/{chat_id}/notes/{note_id}", status_code=204)
async def delete_note(chat_id: uuid.UUID, note_id: uuid.UUID, request: Request):
    use_case: DeleteNote = _use_cases(request).delete_note
    await use_case.execute(chat_id, note_id)
