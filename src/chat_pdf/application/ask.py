import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from chat_pdf.application.chats import require_chat
from chat_pdf.application.find_all import ExhaustiveFinder
from chat_pdf.application.ports import (
    ChatModel,
    ChatRepository,
    ChunkStore,
    DocumentRepository,
    FileStorage,
    ProviderFactory,
    ProviderRepository,
    Reranker,
    Retriever,
    ToolCall,
)
from chat_pdf.application.source_search import DocumentSearch, Event
from chat_pdf.domain.calibration import calibration_table
from chat_pdf.domain.catalog import has_media
from chat_pdf.domain.config import RagConfig
from chat_pdf.domain.errors import Conflict, ProviderError, ToolsUnsupported
from chat_pdf.domain.models import Chat, Chunk, Document, Source
from chat_pdf.domain.panorama import is_global_question
from chat_pdf.domain.prompting import (
    build_answer_messages,
    find_all_tool,
    format_sources,
    parse_cited,
    pictures_to_attach,
    search_tool,
    spoiler_instruction,
    strip_citations,
    style_instructions,
)
from chat_pdf.domain.prompts import (
    FIND_ALL_TOOL_NAME,
    SEARCH_TOOL_NAME,
    SPOILER_PLANNER,
    TOOL_LIMIT_REACHED,
    TOOL_NO_RESULTS,
    TOOL_UNKNOWN,
    search_system,
)
from chat_pdf.domain.providers import ProviderProfile
from chat_pdf.domain.retrieval import clamp_k, fit_sources, is_weak_evidence
from chat_pdf.domain.sources import assign_aliases, readable_sources

log = logging.getLogger(__name__)


class RetrievalPlanner:
    """Tool loop: the model decides `query` and `k`; the backend enforces the limits."""

    def __init__(self, chat: ChatModel, profile: ProviderProfile, history: list[dict[str, str]]):
        self._chat = chat
        self._profile = profile
        self._history = history

    async def plan(self, question: str, search: DocumentSearch) -> AsyncIterator[Event]:
        """Runs the searches, yielding one `search` event as each one completes."""
        if self._profile.supports_tools:
            async for event in self._tool_loop(question, search):
                yield event
        if not search.searches:  # no tools, unsupported, or the model never searched
            await search.run(question, self._profile.default_k)
            for event in search.announce():
                yield event
        if is_global_question(question) and search.needs_panorama():
            # Safety net for small models: a global question must not rest on one narrow search.
            # Model hits keep their numbers; the panorama chunks follow in document order.
            search.add_panorama()
            for event in search.announce():
                yield event

    async def _tool_loop(self, question: str, search: DocumentSearch) -> AsyncIterator[Event]:
        media = search.has_media
        aliases = search.aliases if search.multi else None
        tools = [
            search_tool(self._profile.max_k, media, aliases),
            *([find_all_tool(aliases)] if media else []),
        ]
        system = search_system(
            self._profile.max_tool_calls, self._profile.max_k, media, len(search.aliases)
        )
        if search.max_page is not None:
            system += "\n" + SPOILER_PLANNER.format(page=search.max_page)
        catalog = f"{search.catalog}\n" if search.catalog else ""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": f"{system}\n\n{catalog}{search.index}"},
            *self._history,
            {"role": "user", "content": question},
        ]
        tool_choice = "required"
        for _ in range(self._profile.max_tool_calls):  # bounds rounds, valid searches or not
            try:
                result = await self._complete(messages, tools, tool_choice)
            except ToolsUnsupported:
                return
            tool_choice = "auto"
            if not result.tool_calls:
                return
            messages.append(
                {
                    "role": "assistant",
                    "content": result.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in result.tool_calls
                    ],
                }
            )
            searched = len(search.searches)
            for call in result.tool_calls:
                content = await self._run_tool(call, question, search)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
                for event in search.announce():
                    yield event
            if len(search.searches) == searched:  # no valid search this round: stop insisting
                return

    async def _complete(self, messages, tools, tool_choice):
        try:
            return await self._chat.complete(messages, tools, tool_choice)
        except ToolsUnsupported:
            if tool_choice != "required":
                raise
            return await self._chat.complete(messages, tools, "auto")  # may raise -> fallback

    async def _run_tool(self, call: ToolCall, question: str, search: DocumentSearch) -> str:
        if call.name not in (SEARCH_TOOL_NAME, FIND_ALL_TOOL_NAME):
            return TOOL_UNKNOWN.format(name=call.name)
        if len(search.searches) >= self._profile.max_tool_calls:
            return TOOL_LIMIT_REACHED
        query = str(call.arguments.get("query") or "").strip() or question
        modality = call.arguments.get("modality")
        only, unknown = search.resolve(call.arguments.get("source"))
        if unknown:
            return unknown
        if call.name == FIND_ALL_TOOL_NAME:
            return await search.find_all(query, modality or "imagen", only)
        try:
            requested = int(call.arguments.get("k"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            requested = None
        k = clamp_k(requested, self._profile.default_k, self._profile.max_k)
        found = await search.run(query, k, modality or "auto", only)
        return format_sources(found) or TOOL_NO_RESULTS


def _read_asset_url(path) -> str:
    return "data:image/webp;base64," + base64.b64encode(path.read_bytes()).decode()


@dataclass
class _Prepared:
    chat: Chat
    docs: list[Document]  # the sources this answer reads (ready, active), in S1..Sn order
    aliases: dict[UUID, str]  # of ALL the chat's sources: an alias never changes with the filter
    labeled: bool  # the chat has several sources
    embedding_profile: ProviderProfile
    chat_profile: ProviderProfile
    history: list[dict[str, str]]
    max_page: int | None = None


class AskQuestion:
    def __init__(
        self,
        docs: DocumentRepository,
        chats: ChatRepository,
        providers: ProviderRepository,
        retriever: Retriever,
        reranker: Reranker,
        factory: ProviderFactory,
        config: RagConfig,
        chunks: ChunkStore,
        storage: FileStorage | None = None,
    ):
        self._storage = storage
        self._chunks = chunks
        self._docs = docs
        self._chats = chats
        self._providers = providers
        self._retriever = retriever
        self._reranker = reranker
        self._factory = factory
        self._cfg = config

    async def execute(
        self,
        chat_id: UUID,
        question: str,
        max_page: int | None = None,
        source_ids: list[UUID] | None = None,
    ) -> AsyncIterator[Event]:
        """Validate preconditions (raises DomainError), then return the SSE event stream.

        The question reads the chat's ready sources minus the ones the user muted; `source_ids`
        (per ask) replaces that selection."""
        chat = await require_chat(self._chats, chat_id)
        chat_profile = await self._providers.active("chat")
        if chat_profile is None:
            raise Conflict("no_chat_profile", "No hay un proveedor de chat activo")
        all_docs = await self._docs.list_for_chat(chat_id)
        if not all_docs:
            raise Conflict("no_document", "El chat no tiene documento")
        docs = readable_sources(all_docs, chat.disabled_sources, source_ids)
        if not docs:
            if any(d.status == "ready" for d in all_docs):
                raise Conflict("no_active_sources", "No hay fuentes activas para consultar")
            raise Conflict(
                "document_not_ready", f"El documento está en estado {all_docs[0].status}"
            )
        # The sources' own embedding profile, NOT the active one: vectors must match. A chat's
        # sources share one (uploads join their siblings'); a stray one is left out.
        embedding_profile = await self._providers.get(docs[0].embedding_profile_id)
        if embedding_profile is None:
            raise Conflict(
                "embedding_profile_missing", "El perfil de embeddings del documento ya no existe"
            )
        docs = [d for d in docs if d.embedding_profile_id == embedding_profile.id]
        messages = await self._chats.history(chat_id, self._cfg.history_limit)
        history = [  # `[n]` of old answers would clash with the numbering of this one
            {
                "role": m.role,
                "content": strip_citations(m.content) if m.role == "assistant" else m.content,
            }
            for m in messages
        ]
        prepared = _Prepared(
            chat, docs, assign_aliases(all_docs), len(all_docs) > 1, embedding_profile,
            chat_profile, history, max_page,
        )  # fmt: skip
        return self._run(prepared, question)

    async def _run(self, p: _Prepared, question: str) -> AsyncIterator[Event]:
        chat = self._factory.chat(p.chat_profile)
        doc_chunks = [c for d in p.docs for c in await self._chunks.for_document(d.id)]
        search = DocumentSearch(
            self._retriever,
            self._reranker,
            self._factory.embedder(p.embedding_profile),
            p.embedding_profile,
            p.docs,
            self._cfg,
            doc_chunks,
            p.max_page,
            await self._finder(p.embedding_profile, doc_chunks),
            p.aliases,
            p.labeled,
        )
        async for event in RetrievalPlanner(chat, p.chat_profile, p.history).plan(question, search):
            yield event
        sources = fit_sources(search.sources, self._cfg.max_context_chars)
        citations = [s.cite(self._cfg.snippet_chars, search.table) for s in sources]
        yield "citations", citations

        weak = is_weak_evidence(
            sources, self._cfg.weak_evidence_threshold, p.embedding_profile.calibration
        )
        style = " ".join(
            part for part in (style_instructions(p.chat), spoiler_instruction(p.max_page)) if part
        )
        images = await self._pictures(p.chat_profile, sources)
        messages = build_answer_messages(p.history, question, sources, style, weak, images)
        parts: list[str] = []
        try:
            async for token in chat.stream(messages):
                parts.append(token)
                yield "token", {"text": token}
        except ProviderError:  # includes StreamingUnsupported
            if parts:
                raise
            result = await chat.complete(messages)
            parts.append(result.content or "")
            yield "token", {"text": parts[0]}

        answer = "".join(parts)
        message_id = await self._chats.add_exchange(
            p.chat.id, question, answer, citations, search.searches, weak, p.chat.persona
        )
        yield (
            "done",
            {
                "message_id": str(message_id),
                "cited": parse_cited(answer, sources),
                "weak_evidence": weak,
            },
        )

    async def _finder(
        self, embedding_profile: ProviderProfile, chunks: list[Chunk]
    ) -> ExhaustiveFinder | None:
        """`find_all` machinery, built only for documents with media. The vision model is the
        ACTIVE vision profile (None: candidates are returned unverified, with a confidence)."""
        if not has_media(chunks):
            return None
        profile = await self._providers.active("vision")
        vision = self._factory.vision(profile) if profile else None
        table = calibration_table(embedding_profile.calibration, self._cfg.weak_evidence_threshold)
        return ExhaustiveFinder(self._retriever, self._chunks, self._storage, vision, table)

    async def _pictures(
        self, chat_profile: ProviderProfile, sources: list[Source]
    ) -> list[tuple[int, str]]:
        """Pictures of the best image sources for a chat model that can see (at most 3)."""
        if not chat_profile.supports_vision or self._storage is None:
            return []
        attached = []
        for s in pictures_to_attach(sources):
            path = self._storage.asset_path(s.document_id, "assets", s.chunk.asset_key)  # type: ignore[arg-type]
            try:
                attached.append((s.n, await asyncio.to_thread(_read_asset_url, path)))
            except OSError:
                log.warning("asset of segment %s is unreadable; answering without it", s.chunk.id)
        return attached
