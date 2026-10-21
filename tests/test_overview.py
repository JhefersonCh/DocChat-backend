import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fakes import (
    FakeChat,
    FakeChatRepository,
    FakeDb,
    FakeDocumentRepository,
    FakeEmbedder,
    FakeFileStorage,
    FakeLoader,
    FakeProviderFactory,
    FakeProviderRepository,
    FakeUnitOfWork,
    chat_profile,
    embedding_profile,
)

from chat_pdf.application.documents import FailInterruptedIngests, IngestDocument
from chat_pdf.application.generation import GenerateOverview, RequestSummary, SuggestFollowUps
from chat_pdf.domain.config import RagConfig
from chat_pdf.domain.errors import Conflict, NotFound
from chat_pdf.domain.models import Message, Page

OVERVIEW = '{"summary": "Un cuento.", "questions": ["¿A?", "¿B?", "¿C?"]}'


def _message(role, content) -> Message:
    return Message(uuid4(), role, content, datetime.now(UTC))


def test_suggestions_are_generated_once_and_only_for_assistant_messages():
    db, chat = FakeDb(), FakeChat(answer='["¿Uno?", "¿Dos?", "¿Tres?"]')
    chat_id = db.add_chat().id
    question, answer = _message("user", "¿Quién?"), _message("assistant", "El zorro [1].")
    db.messages[chat_id] = [question, answer]
    profile = chat_profile()
    use_case = SuggestFollowUps(
        FakeChatRepository(db),
        FakeProviderRepository(profile, active={"chat": profile}),
        FakeProviderFactory(chat),
    )

    first = asyncio.run(use_case.execute(chat_id, answer.id))
    second = asyncio.run(use_case.execute(chat_id, answer.id))

    assert first == second == ["¿Uno?", "¿Dos?", "¿Tres?"]
    assert chat.completions == 1  # the second call came from storage
    assert db.messages[chat_id][1].suggestions == first
    with pytest.raises(NotFound):
        asyncio.run(use_case.execute(chat_id, question.id))


def test_suggestions_without_active_chat_profile_conflict():
    db = FakeDb()
    chat_id = db.add_chat().id
    answer = _message("assistant", "hola")
    db.messages[chat_id] = [answer]
    use_case = SuggestFollowUps(
        FakeChatRepository(db), FakeProviderRepository(), FakeProviderFactory()
    )

    with pytest.raises(Conflict) as exc:
        asyncio.run(use_case.execute(chat_id, answer.id))
    assert exc.value.code == "no_chat_profile"


def _ingest(db, providers, chat: FakeChat, profile):
    uow = lambda: FakeUnitOfWork(db)  # noqa: E731
    factory = FakeProviderFactory(chat, FakeEmbedder())
    return IngestDocument(
        uow,
        providers,
        factory,
        {"text/plain": FakeLoader(Page(None, "Había una vez un zorro."))},
        FakeFileStorage(),
        RagConfig(),
        GenerateOverview(uow, providers, factory),
    )


@pytest.mark.parametrize(("active", "expected"), [(True, "ready"), (False, "skipped")])
def test_overview_follows_ingest_and_is_skipped_without_chat_profile(active, expected):
    emb, chat = embedding_profile(), chat_profile()
    db, fake_chat = FakeDb(), FakeChat(answer=OVERVIEW)
    doc = db.add_document(db.add_chat().id, emb.id, status="pending")
    providers = FakeProviderRepository(emb, chat, active={"chat": chat} if active else {})

    asyncio.run(_ingest(db, providers, fake_chat, emb).execute(doc.id))

    done = db.docs[doc.id]
    assert (done.status, done.summary_status) == ("ready", expected)
    assert fake_chat.completions == (1 if active else 0)
    if active:
        assert (done.summary, done.suggested_questions) == ("Un cuento.", ["¿A?", "¿B?", "¿C?"])


def test_failed_overview_never_fails_the_document_and_summary_request_rules():
    emb, chat = embedding_profile(), chat_profile()
    db = FakeDb()
    doc = db.add_document(db.add_chat().id, emb.id, status="pending")
    providers = FakeProviderRepository(emb, chat, active={"chat": chat})

    asyncio.run(_ingest(db, providers, FakeChat(answer="no es json"), emb).execute(doc.id))

    assert (db.docs[doc.id].status, db.docs[doc.id].summary_status) == ("ready", "failed")
    request = RequestSummary(FakeDocumentRepository(db), providers)
    asyncio.run(request.execute(doc.id))  # retry allowed from failed
    assert db.docs[doc.id].summary_status == "pending"
    with pytest.raises(Conflict) as exc:  # already pending
        asyncio.run(request.execute(doc.id))
    assert exc.value.code == "document_busy"
    asyncio.run(FailInterruptedIngests(FakeDocumentRepository(db)).execute())
    assert db.docs[doc.id].summary_status == "failed"  # restart leaves no pending overview
