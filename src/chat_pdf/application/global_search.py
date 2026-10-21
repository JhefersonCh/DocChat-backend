"""Global search over every chat (project chats included): messages, chats, projects, sources."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from chat_pdf.domain.chat_search import (
    clean_text,
    find_highlights,
    like_pattern,
    make_snippet,
    parse_query,
    score,
)

CANDIDATES = 300  # newest matches fetched before ranking
SIDE_LIMIT = 8  # chats / projects / sources shown


class ChatSearchRepository(Protocol):
    async def messages(self, patterns: list[str], limit: int) -> list[dict[str, Any]]: ...

    async def chats(self, patterns: list[str], limit: int) -> list[dict[str, Any]]: ...

    async def projects(self, patterns: list[str], limit: int) -> list[dict[str, Any]]: ...

    async def sources(self, patterns: list[str], limit: int) -> list[dict[str, Any]]: ...


Spans = list[tuple[int, int]]


@dataclass(frozen=True)
class MessageHit:
    message_id: UUID
    chat_id: UUID
    chat_title: str
    project_id: UUID | None
    project_name: str | None
    role: str
    created_at: datetime
    snippet: str
    highlights: Spans


@dataclass(frozen=True)
class ChatHit:
    chat_id: UUID
    title: str
    project_id: UUID | None
    project_name: str | None
    created_at: datetime
    highlights: Spans


@dataclass(frozen=True)
class ProjectHit:
    project_id: UUID
    name: str
    created_at: datetime
    highlights: Spans


@dataclass(frozen=True)
class SourceHit:
    document_id: UUID
    filename: str
    chat_id: UUID | None
    project_id: UUID | None
    highlights: Spans


@dataclass(frozen=True)
class SearchAllResult:
    messages: list[MessageHit] = field(default_factory=list)
    chats: list[ChatHit] = field(default_factory=list)
    projects: list[ProjectHit] = field(default_factory=list)
    sources: list[SourceHit] = field(default_factory=list)


class SearchAll:
    def __init__(self, repo: ChatSearchRepository):
        self._repo = repo

    async def execute(self, query: str, limit: int = 20) -> SearchAllResult:
        terms = parse_query(query)
        if not terms:
            return SearchAllResult()
        patterns = [like_pattern(t) for t in terms]
        side = min(limit, SIDE_LIMIT)
        rows = await self._repo.messages(patterns, CANDIDATES)
        ranked = sorted(
            rows, key=lambda r: score(r["content"], terms, r["created_at"]), reverse=True
        )
        messages = []
        for r in ranked[:limit]:
            snip = make_snippet(r["content"], terms)
            messages.append(
                MessageHit(
                    r["id"],
                    r["chat_id"],
                    r["chat_title"],
                    r["project_id"],
                    r["project_name"],
                    r["role"],
                    r["created_at"],
                    snip.text,
                    snip.highlights,
                )  # fmt: skip
            )
        chats = [
            ChatHit(
                r["id"],
                r["title"],
                r["project_id"],
                r["project_name"],
                r["created_at"],
                find_highlights(r["title"], terms),
            )  # fmt: skip
            for r in await self._repo.chats(patterns, side)
        ]
        projects = [
            ProjectHit(r["id"], r["name"], r["created_at"], find_highlights(r["name"], terms))
            for r in await self._repo.projects(patterns, side)
        ]
        sources = [
            SourceHit(
                r["id"],
                r["filename"],
                r["chat_id"],
                r["project_id"],
                find_highlights(clean_text(r["filename"]), terms),
            )  # fmt: skip
            for r in await self._repo.sources(patterns, side)
        ]
        return SearchAllResult(messages, chats, projects, sources)
