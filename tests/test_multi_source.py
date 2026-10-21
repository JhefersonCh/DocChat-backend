"""Phase 7 through the use cases: add / remove sources, the title rule, the muted selection and a
question over several sources (aliases, `source` filter, citations)."""

import asyncio
from dataclasses import replace
from uuid import uuid4

import pytest
from fakes import (
    FakeChat,
    FakeChatRepository,
    FakeChunkStore,
    FakeDb,
    FakeDocumentRepository,
    FakeEmbedder,
    FakeFileStorage,
    FakeProviderFactory,
    FakeProviderRepository,
    FakeUnitOfWork,
    IdentityReranker,
    chat_profile,
    embedding_profile,
)

from chat_pdf.adapters.http.sse import sse
from chat_pdf.application.ask import AskQuestion
from chat_pdf.application.chats import GetChat, RemoveSource, UpdateChatSettings
from chat_pdf.application.documents import UploadDocument
from chat_pdf.application.ports import ChatResult, ToolCall
from chat_pdf.application.source_search import DocumentSearch
from chat_pdf.domain.config import RagConfig
from chat_pdf.domain.errors import Conflict, InvalidInput, NotFound
from chat_pdf.domain.models import Chunk


def _upload(db, profile, storage, chat_id, name):
    upload = UploadDocument(
        lambda: FakeUnitOfWork(db),
        FakeProviderRepository(profile, active={"embedding": profile}),
        storage,
    )

    async def content():
        yield b"data"

    return asyncio.run(upload.execute(chat_id, name, "text/plain", content()))


def test_adding_sources_keeps_the_old_ones_and_titles_the_chat():
    profile, db, storage = embedding_profile(), FakeDb(), FakeFileStorage()
    chat = db.add_chat()

    first = _upload(db, profile, storage, chat.id, "a.txt")
    assert db.chats[chat.id].title == "a.txt"
    second = _upload(db, profile, storage, chat.id, "b.txt")

    assert set(db.docs) == {first, second} and set(storage.files) == {first, second}
    assert db.chats[chat.id].title == "2 fuentes"


def test_a_new_source_joins_the_embedding_profile_of_its_siblings():
    old, new = embedding_profile(), embedding_profile()
    db, storage = FakeDb(), FakeFileStorage()
    chat = db.add_chat()
    sibling = db.add_document(chat.id, old.id)
    upload = UploadDocument(
        lambda: FakeUnitOfWork(db),
        FakeProviderRepository(old, new, active={"embedding": new}),  # the active one changed
        storage,
    )

    async def content():
        yield b"x"

    added = asyncio.run(upload.execute(chat.id, "b.txt", "text/plain", content()))

    assert db.docs[added].embedding_profile_id == old.id and sibling.id in db.docs


def test_a_chat_has_a_source_limit():
    profile, db, storage = embedding_profile(), FakeDb(), FakeFileStorage()
    chat = db.add_chat()
    for _ in range(12):
        db.add_document(chat.id, profile.id)

    with pytest.raises(Conflict) as err:
        _upload(db, profile, storage, chat.id, "x.txt")

    assert err.value.code == "too_many_sources"


def test_remove_source_deletes_document_files_and_retitles():
    profile, db, storage = embedding_profile(), FakeDb(), FakeFileStorage()
    chat = db.add_chat()
    a = _upload(db, profile, storage, chat.id, "a.txt")
    b = _upload(db, profile, storage, chat.id, "b.txt")
    db.chats[chat.id] = replace(db.chats[chat.id], disabled_sources=[a])
    remove = RemoveSource(lambda: FakeUnitOfWork(db), storage)

    asyncio.run(remove.execute(chat.id, a))

    assert list(db.docs) == [b] and list(storage.files) == [b]
    assert db.chats[chat.id].title == "b.txt" and db.chats[chat.id].disabled_sources == []
    with pytest.raises(NotFound):
        asyncio.run(remove.execute(chat.id, a))  # no longer a source of this chat


def test_get_chat_lists_sources_in_order_with_the_first_as_primary():
    profile, db = embedding_profile(), FakeDb()
    chat = db.add_chat()
    first = db.add_document(chat.id, profile.id)
    second = db.add_document(chat.id, profile.id)

    detail = asyncio.run(
        GetChat(FakeChatRepository(db), FakeDocumentRepository(db)).execute(chat.id)
    )

    assert [d.id for d in detail.sources] == [first.id, second.id]
    assert detail.document.id == first.id
    assert detail.overview is not None  # two sources


def test_active_source_ids_are_stored_as_the_muted_complement():
    profile, db = embedding_profile(), FakeDb()
    chat = db.add_chat()
    a, b = db.add_document(chat.id, profile.id), db.add_document(chat.id, profile.id)
    use_case = UpdateChatSettings(FakeChatRepository(db), FakeDocumentRepository(db))

    updated = asyncio.run(use_case.execute(chat.id, {"active_source_ids": [a.id]}))
    assert updated.disabled_sources == [b.id]

    with pytest.raises(InvalidInput):
        asyncio.run(use_case.execute(chat.id, {"active_source_ids": [uuid4()]}))


# --- asking over several sources ------------------------------------------------------------


def _chunk(text, similarity=0.8):
    return Chunk(0, text, 1, 1, 0, 10, id=uuid4(), similarity=similarity)


class Ask:
    def __init__(self, rounds=(), muted=()):
        self.chat_p, self.emb = (
            chat_profile(supports_tools=True, max_tool_calls=3),
            embedding_profile(),
        )
        self.db = FakeDb()
        chat = self.db.add_chat()
        self.chat_id = chat.id
        self.docs = []
        for name, kind, text in [("informe.pdf", "pdf", "perro"), ("clip.mp4", "video", "gato")]:
            doc = self.db.add_document(chat.id, self.emb.id)
            doc = replace(doc, filename=name, kind=kind)
            self.db.docs[doc.id] = doc
            self.db.chunks[doc.id] = [_chunk(f"El {text} aparece aquí")]
            self.docs.append(doc)
        self.db.chats[chat.id] = replace(chat, disabled_sources=[self.docs[i].id for i in muted])
        self.model = FakeChat(rounds=rounds)
        store = FakeChunkStore(self.db)
        self.use_case = AskQuestion(
            FakeDocumentRepository(self.db), FakeChatRepository(self.db),
            FakeProviderRepository(self.chat_p, self.emb, active={"chat": self.chat_p}),
            store, IdentityReranker(), FakeProviderFactory(self.model), RagConfig(), store,
        )  # fmt: skip

    def ask(self, **kw):
        async def go():
            return [e async for e in await self.use_case.execute(self.chat_id, "¿perros?", **kw)]

        return asyncio.run(go())


def _document_search(ask):
    store = FakeChunkStore(ask.db)
    return DocumentSearch(store, IdentityReranker(), FakeEmbedder(), ask.emb, ask.docs, RagConfig())


def _search(**args):
    return ChatResult(None, [ToolCall("c", "search_document", {"query": "perro", "k": 5, **args})])


def test_question_without_tools_reads_every_source_and_cites_the_alias():
    ask = Ask()

    events = ask.ask()

    citations = dict(events)["citations"]
    assert [(c.source_alias, c.filename) for c in citations] == [
        ("S1", "informe.pdf"), ("S2", "clip.mp4")
    ]  # fmt: skip
    payload = sse("citations", citations)
    assert '"source_id"' in payload and '"source_name": "clip.mp4"' in payload
    system = ask.model.streamed[0][0]["content"]
    assert "varias fuentes" in system
    assert "[2] (S2 · clip.mp4)" in ask.model.streamed[0][-1]["content"]


def test_search_tool_filters_by_source_alias():
    ask = Ask(rounds=[_search(source="S2")])

    events = ask.ask()

    citations = dict(events)["citations"]
    assert [c.source_alias for c in citations] == ["S2"]
    assert ask.db.exchanges[0]["searches"][0].source == "S2"
    spec = ask.model.tool_specs[0][0]["function"]["parameters"]["properties"]["source"]
    assert spec["enum"] == ["all", "S1", "S2"]
    assert "S1 · informe.pdf" in ask.model.tool_systems[0]  # the catalog names every source
    assert "TODAS las del catálogo" in ask.model.tool_systems[0]  # cross-source guidance


def test_unknown_source_alias_is_reported_to_the_model_and_the_search_falls_back():
    ask = Ask(rounds=[_search(source="S7")])

    events = ask.ask()

    assert [c.source_alias for c in dict(events)["citations"]] == ["S1", "S2"]  # fallback: all
    only, error = _document_search(ask).resolve("S7")
    assert only is None and "S7" in error and "S1, S2" in error


def test_muted_source_is_excluded_but_keeps_its_alias():
    ask = Ask(muted=[0])

    events = ask.ask()

    assert [c.source_alias for c in dict(events)["citations"]] == ["S2"]
    assert "S1 · informe.pdf" not in ask.model.tool_systems[0] if ask.model.tool_systems else True


def test_per_ask_selection_overrides_the_muted_list():
    ask = Ask(muted=[0])

    events = ask.ask(source_ids=[ask.docs[0].id])

    assert [c.source_alias for c in dict(events)["citations"]] == ["S1"]


def test_all_sources_muted_is_a_clear_error():
    ask = Ask(muted=[0, 1])

    with pytest.raises(Conflict) as err:
        ask.ask()

    assert err.value.code == "no_active_sources"
