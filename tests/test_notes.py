import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fakes import FakeChat, FakeChatRepository, FakeDb, FakeProviderFactory
from fakes_notes import FakeNoteRepository
from fastapi.testclient import TestClient
from test_flow import API, _ready_chat, _settings

from chat_pdf.application.notes import DeleteNote, ListNotes, RenameNote, SaveNote
from chat_pdf.domain.errors import NotFound
from chat_pdf.domain.models import Citation, Message
from chat_pdf.main import create_app


def _setup():
    db, notes = FakeDb(), FakeNoteRepository()
    chat_id = db.add_chat().id
    cite = Citation(1, uuid4(), "x.pdf", 3, "snip", uuid4(), None, "texto completo", "Intro", 0.8)
    answer = Message(uuid4(), "assistant", "Respuesta [1]", datetime.now(UTC), [cite])
    question = Message(uuid4(), "user", "pregunta", datetime.now(UTC))
    db.messages[chat_id] = [question, answer]
    chats = FakeChatRepository(db)
    return chat_id, question, answer, notes, chats


def test_save_snapshots_answer_and_citations_and_is_idempotent_per_message():
    chat_id, _, answer, notes, chats = _setup()
    save = SaveNote(chats, notes)

    first = asyncio.run(save.execute(chat_id, answer.id, "  Plazo  "))
    again = asyncio.run(save.execute(chat_id, answer.id, None))

    assert first.title == "Plazo" and first.content == "Respuesta [1]"
    assert first.citations[0]["content"] == "texto completo" and first.citations[0]["page"] == 3
    assert again.id == first.id and len(notes.items) == 1


def test_save_rejects_unknown_chat_missing_or_user_messages():
    chat_id, question, _, notes, chats = _setup()
    save = SaveNote(chats, notes)
    for cid, mid in [(uuid4(), uuid4()), (chat_id, uuid4()), (chat_id, question.id)]:
        with pytest.raises(NotFound):
            asyncio.run(save.execute(cid, mid, None))


def test_list_rename_delete():
    chat_id, _, answer, notes, chats = _setup()
    note = asyncio.run(SaveNote(chats, notes).execute(chat_id, answer.id, None))

    renamed = asyncio.run(RenameNote(notes).execute(chat_id, note.id, "Nuevo"))
    listed = asyncio.run(ListNotes(chats, notes).execute(chat_id))
    asyncio.run(DeleteNote(notes).execute(chat_id, note.id))

    assert renamed.title == "Nuevo" and [n.id for n in listed] == [note.id]
    for use_case in (
        RenameNote(notes).execute(chat_id, note.id, "x"),
        DeleteNote(notes).execute(chat_id, note.id),
    ):
        with pytest.raises(NotFound):
            asyncio.run(use_case)


def test_notes_over_http_and_cascade_with_chat(empty_db_url, tmp_path):
    factory = FakeProviderFactory(FakeChat(answer="El plazo es 30 dias [1]"))
    with TestClient(create_app(_settings(empty_db_url, tmp_path), factory)) as client:
        chat_id, _ = _ready_chat(client)
        client.post(f"{API}/chats/{chat_id}/messages", json={"content": "plazo?"})
        answer = client.get(f"{API}/chats/{chat_id}/messages").json()[1]
        base = f"{API}/chats/{chat_id}/notes"

        missing = client.post(base, json={"message_id": str(uuid4())})
        created = client.post(base, json={"message_id": answer["id"], "title": "Plazo"})
        listed = client.get(base).json()
        patched = client.patch(f"{base}/{created.json()['id']}", json={"title": "Otro"})
        gone = client.delete(f"{base}/{created.json()['id']}")
        again = client.delete(f"{base}/{created.json()['id']}")
        client.post(base, json={"message_id": answer["id"]})
        client.delete(f"{API}/chats/{chat_id}")

    assert missing.status_code == 404 and created.status_code == 201
    assert (
        listed[0]["content"] == answer["content"] and listed[0]["citations"] == answer["citations"]
    )
    assert listed[0]["message_id"] == answer["id"] and listed[0]["title"] == "Plazo"
    assert patched.json()["title"] == "Otro"
    assert gone.status_code == 204 and again.status_code == 404
