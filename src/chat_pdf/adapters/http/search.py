"""GET /api/v1/search: keyword search over all chats. See docs/frontend-api.md."""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from chat_pdf.application.global_search import SearchAll, SearchAllResult

router = APIRouter(prefix="/api/v1")


def search_all(request: Request) -> SearchAll:
    return request.app.state.use_cases.search_all


class MessageHitOut(BaseModel):
    message_id: uuid.UUID
    chat_id: uuid.UUID
    chat_title: str
    project_id: uuid.UUID | None = None
    project_name: str | None = None
    role: str
    created_at: datetime
    snippet: str
    highlights: list[tuple[int, int]]


class ChatHitOut(BaseModel):
    chat_id: uuid.UUID
    title: str
    project_id: uuid.UUID | None = None
    project_name: str | None = None
    created_at: datetime
    highlights: list[tuple[int, int]]


class ProjectHitOut(BaseModel):
    project_id: uuid.UUID
    name: str
    created_at: datetime
    highlights: list[tuple[int, int]]


class SourceHitOut(BaseModel):
    document_id: uuid.UUID
    filename: str
    chat_id: uuid.UUID | None = None
    project_id: uuid.UUID | None = None
    highlights: list[tuple[int, int]]


class SearchOut(BaseModel):
    messages: list[MessageHitOut]
    chats: list[ChatHitOut]
    projects: list[ProjectHitOut]
    sources: list[SourceHitOut]

    @classmethod
    def of(cls, r: SearchAllResult) -> "SearchOut":
        return cls(
            messages=[MessageHitOut(**vars(m)) for m in r.messages],
            chats=[ChatHitOut(**vars(c)) for c in r.chats],
            projects=[ProjectHitOut(**vars(p)) for p in r.projects],
            sources=[SourceHitOut(**vars(s)) for s in r.sources],
        )


@router.get("/search", response_model=SearchOut)
async def search(
    uc: Annotated[SearchAll, Depends(search_all)],
    q: Annotated[str, Query(max_length=200)] = "",
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
):
    """Keyword (not natural-language) search over every chat, project chats included: messages
    (with a snippet and highlight offsets), chat titles, projects and source file names. Case and
    accent insensitive, partial words, several words = AND, "quoted phrases" exact."""
    return SearchOut.of(await uc.execute(q, limit))
