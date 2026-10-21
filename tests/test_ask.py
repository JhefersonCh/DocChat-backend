import asyncio
from dataclasses import dataclass, replace
from uuid import uuid4

from fakes import (
    CORPUS,
    FakeChat,
    FakeChatRepository,
    FakeChunkStore,
    FakeDb,
    FakeDocumentRepository,
    FakeProviderFactory,
    FakeProviderRepository,
    IdentityReranker,
    chat_profile,
    embedding_profile,
)

from chat_pdf.application.ask import AskQuestion
from chat_pdf.application.ports import ChatResult, ToolCall
from chat_pdf.domain.config import RagConfig
from chat_pdf.domain.models import Chunk, Search


@dataclass
class Asked:
    events: list[tuple[str, object]]
    db: FakeDb
    chunks: FakeChunkStore
    factory: FakeProviderFactory

    @property
    def ks(self) -> list[int]:
        """k of each search actually run (as persisted with the answer)."""
        return [s.k for s in self.db.exchanges[0]["searches"]]


def _search(k):
    return ChatResult(None, [ToolCall("c", "search_document", {"query": "q", "k": k})])


def _unknown():
    return ChatResult(None, [ToolCall("c", "no_such_tool", {})])


def _ask(
    chat: FakeChat,
    profile=None,
    doc_profile=None,
    similarity=None,
    style=None,
    corpus=CORPUS,
    question="question",
    max_page=None,
) -> Asked:
    profile = profile or chat_profile()
    doc_profile = doc_profile or embedding_profile()
    db = FakeDb()
    chat_row = db.add_chat()
    if style:
        db.chats[chat_row.id] = replace(chat_row, **style)
    db.chunks[db.add_document(chat_row.id, doc_profile.id).id] = corpus
    providers = FakeProviderRepository(
        profile, doc_profile, embedding_profile(), active={"chat": profile}
    )
    chunks = FakeChunkStore(db, similarity)
    factory = FakeProviderFactory(chat)
    use_case = AskQuestion(
        FakeDocumentRepository(db),
        FakeChatRepository(db),
        providers,
        chunks,
        IdentityReranker(),
        factory,
        RagConfig(),
        chunks,
    )

    async def collect():
        return [event async for event in await use_case.execute(chat_row.id, question, max_page)]

    return Asked(asyncio.run(collect()), db, chunks, factory)


def test_falls_back_to_non_streaming_when_stream_unsupported():
    events = _ask(FakeChat(stream_unsupported=True)).events

    assert [name for name, _ in events] == ["search", "citations", "token", "done"]
    assert events[2][1] == {"text": "full answer [1]"}
    assert events[3][1]["cited"] == [1]


def test_query_is_embedded_with_document_profile_not_active_one():
    doc_profile = embedding_profile(query_prefix="DOC: ")

    asked = _ask(FakeChat(), doc_profile=doc_profile)

    assert asked.factory.embedder_profiles == [doc_profile.id]
    assert asked.factory.embedding.texts == ["DOC: question"]


def test_tool_loop_clamps_k_and_stops_at_max_tool_calls():
    profile = chat_profile(supports_tools=True, max_k=7, max_tool_calls=2)

    asked = _ask(FakeChat(rounds=[_search(100)] * 10), profile)

    assert asked.ks == [7, 7]
    names = [name for name, _ in asked.events]
    assert names[:3] == ["search", "search", "citations"]  # none for calls beyond the limit
    event = asked.events[0][1]
    assert {k: event[k] for k in ("query", "k", "hits", "panorama")} == {
        "query": "q", "k": 7, "hits": 7, "panorama": False
    }  # fmt: skip
    assert [f["n"] for f in event["found"]] == list(range(1, 8))  # X-ray: where each hit sits
    assert event["found"][0].keys() >= {"chunk_id", "page_start", "page_end", "section", "score"}
    assert len(asked.events[2][1]) == 7  # deduplicated: same chunks both times
    assert asked.db.exchanges[0]["searches"] == [Search("q", 7)] * 2


def test_no_tool_call_falls_back_to_default_k():
    profile = chat_profile(supports_tools=True, default_k=4)

    asked = _ask(FakeChat(rounds=[ChatResult("I know it")]), profile)

    assert asked.ks == [4]
    assert asked.events[0][0] == "search" and asked.events[0][1]["hits"] == 4
    assert asked.events[0][1]["query"] == "question"


def test_no_tools_support_falls_back_to_default_k():
    chat = FakeChat()

    asked = _ask(chat, chat_profile(supports_tools=False, default_k=3))

    assert chat.tool_rounds == 0
    assert asked.ks == [3]


def test_tool_loop_stops_when_model_never_makes_a_valid_search():
    chat = FakeChat(rounds=[_unknown()] * 10)

    asked = _ask(chat, chat_profile(supports_tools=True, default_k=4, max_tool_calls=3))

    assert chat.tool_rounds == 1  # a round without a valid search ends the loop
    assert asked.ks == [4]  # falls through to the default search
    assert asked.events[-1][0] == "done"


def test_weak_evidence_flags_done_and_warns_the_model():
    chat = FakeChat()

    asked = _ask(chat, similarity=0.1)

    assert asked.events[-1][1]["weak_evidence"] is True
    assert asked.db.exchanges[0]["weak_evidence"] is True
    assert "débil" in chat.streamed[0][0]["content"]
    assert asked.events[1][1][0].score == 0.0  # calibrated: 0.1 is under the text floor (0.50)


def test_strong_evidence_is_not_flagged():
    chat = FakeChat()

    asked = _ask(chat, similarity=0.8)

    assert asked.events[-1][1]["weak_evidence"] is False
    assert "débil" not in chat.streamed[0][0]["content"]


def test_chat_style_and_length_are_injected_into_the_answer_prompt():
    chat = FakeChat()

    _ask(chat, style={"style": "simple", "length": "corta"})

    system = chat.streamed[0][0]["content"]
    assert "simple" in system and "breve" in system


def test_global_question_with_narrow_search_adds_panorama_across_sections():
    corpus = [
        Chunk(i, f"chunk {i}", 1, 1, 0, 1, id=uuid4(), section=f"S{i // 4}") for i in range(20)
    ]  # 5 sections of 4 chunks
    chat = FakeChat(
        rounds=[
            ChatResult(
                None, [ToolCall("c", "search_document", {"query": "Dame un resumen", "k": 2})]
            )
        ]
    )

    asked = _ask(
        chat,
        chat_profile(supports_tools=True, default_k=2),
        corpus=corpus,
        question="Dame un resumen general del proyecto",
    )

    events = asked.events
    assert [name for name, _ in events] == ["search", "search", "citations", "token", "done"]
    assert events[1][1]["query"] == "panorama del documento"
    assert events[1][1]["hits"] == events[1][1]["k"] == 4  # S0 was covered; S1..S4 added
    citations = events[2][1]
    assert [c.n for c in citations] == [1, 2, 3, 4, 5, 6]  # model hits first, then document order
    assert {c.section for c in citations} == {f"S{i}" for i in range(5)}
    assert [s.query for s in asked.db.exchanges[0]["searches"]] == [
        "Dame un resumen",
        "panorama del documento",
    ]
    system = chat.tool_systems[0]
    assert "Secciones (5): S0; S1; S2; S3; S4" in system


def test_specific_question_gets_no_panorama():
    corpus = [Chunk(i, f"chunk {i}", 1, 1, 0, 1, id=uuid4(), section=f"S{i}") for i in range(6)]

    asked = _ask(
        FakeChat(), chat_profile(default_k=2), corpus=corpus, question="¿Qué dijo el zorro?"
    )

    assert [name for name, _ in asked.events] == ["search", "citations", "token", "done"]


def test_search_system_prompt_derives_numbers_from_profile():
    from chat_pdf.domain.prompts import search_system

    text = search_system(max_searches=2, max_k=4)
    assert "máximo de 2 búsquedas" in text
    assert "k de 4 a 4" in text  # k range is clamped to max_k
    assert "10" not in text
    assert "en paralelo" in text
    assert "1 búsqueda enfocada," in search_system(1, 3)
