import json
import time
from uuid import uuid4

from fakes import FakeChat, FakeProviderFactory
from fastapi.testclient import TestClient

from chat_pdf.config import Settings
from chat_pdf.domain.errors import ProviderError
from chat_pdf.main import create_app

API = "/api/v1"


def parse_sse(text: str) -> list[tuple[str, object]]:
    events = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def _settings(empty_db_url, tmp_path) -> Settings:
    return Settings(
        database_url=empty_db_url, app_secret_key="any string", upload_dir=str(tmp_path)
    )


def _ready_chat(client) -> tuple[str, str]:
    """Activates a chat profile, creates a chat and uploads a document until it is `ready`."""
    chat_profile = client.post(
        f"{API}/providers",
        json={"kind": "chat", "name": "fake", "base_url": "http://x/v1", "model": "m"},
    ).json()
    active = client.get(f"{API}/settings/active").json()
    active["chat_profile_id"] = chat_profile["id"]
    assert client.put(f"{API}/settings/active", json=active).status_code == 200

    chat_id = client.post(f"{API}/chats").json()["id"]
    upload = client.post(
        f"{API}/chats/{chat_id}/documents",
        files={"file": ("contrato.txt", b"El plazo de entrega es de 30 dias.", "text/plain")},
    )
    assert upload.status_code == 202
    doc_id = upload.json()["document_id"]
    _wait_ready(client, doc_id)
    return chat_id, doc_id


def _wait_ready(client, doc_id: str) -> None:
    for _ in range(50):
        status = client.get(f"{API}/documents/{doc_id}").json()["status"]
        if status != "pending" and status != "processing":
            break
        time.sleep(0.1)
    assert status == "ready"


def test_upload_ask_and_persist(empty_db_url, tmp_path):
    with TestClient(create_app(_settings(empty_db_url, tmp_path), FakeProviderFactory())) as client:
        chat_id, doc_id = _ready_chat(client)

        file = client.get(f"{API}/documents/{doc_id}/file")
        missing = client.get(f"{API}/documents/{uuid4()}/file")

        res = client.post(f"{API}/chats/{chat_id}/messages", json={"content": "Cual es el plazo?"})
        events = parse_sse(res.text)
        messages = client.get(f"{API}/chats/{chat_id}/messages").json()

    names = [name for name, _ in events]
    assert names[0] == "search" and "citations" in names and names[-1] == "done"
    assert set(names[names.index("citations") + 1 : -1]) == {"token"}
    citations = next(d for n, d in events if n == "citations")
    assert citations and citations[0]["filename"] == "contrato.txt"
    assert set(events[-1][1]["cited"]) <= {c["n"] for c in citations}
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["citations"] == citations
    assert citations[0]["chunk_id"] and citations[0]["content"].startswith("El plazo")
    assert citations[0]["page_end"] is None  # TXT has no pages
    assert "section" in citations[0] and citations[0]["section"] is None
    assert file.status_code == 200
    assert file.content == b"El plazo de entrega es de 30 dias."
    assert file.headers["content-type"].startswith("text/plain")
    assert file.headers["content-disposition"] == 'inline; filename="contrato.txt"'
    assert missing.status_code == 404 and missing.json()["code"] == "document_not_found"


def test_stream_failure_ends_with_error_event_and_persists_nothing(empty_db_url, tmp_path):
    chat = FakeChat(stream_error=ProviderError("se cortó el stream"))
    with TestClient(
        create_app(_settings(empty_db_url, tmp_path), FakeProviderFactory(chat))
    ) as client:
        chat_id, _ = _ready_chat(client)

        res = client.post(f"{API}/chats/{chat_id}/messages", json={"content": "Cual es el plazo?"})
        messages = client.get(f"{API}/chats/{chat_id}/messages").json()

    events = parse_sse(res.text)
    assert "token" in [name for name, _ in events]  # it did start streaming
    assert events[-1][0] == "error" and events[-1][1]["code"] == "provider_unreachable"
    assert messages == []


def test_reindex_rebuilds_chunks_that_hybrid_search_finds(empty_db_url, tmp_path):
    with TestClient(create_app(_settings(empty_db_url, tmp_path), FakeProviderFactory())) as client:
        chat_id, doc_id = _ready_chat(client)

        res = client.post(f"{API}/documents/{doc_id}/reindex")
        _wait_ready(client, doc_id)
        missing = client.post(f"{API}/documents/{uuid4()}/reindex")
        events = parse_sse(
            client.post(f"{API}/chats/{chat_id}/messages", json={"content": "plazo"}).text
        )

    assert res.status_code == 202 and res.json() == {"document_id": doc_id, "status": "processing"}
    assert missing.status_code == 404
    citations = next(d for n, d in events if n == "citations")
    assert [c["content"] for c in citations] == ["El plazo de entrega es de 30 dias."]


def test_cancel_endpoint_rejects_idle_and_unknown_documents_and_documents_expose_progress(
    empty_db_url, tmp_path
):
    with TestClient(create_app(_settings(empty_db_url, tmp_path), FakeProviderFactory())) as client:
        _, doc_id = _ready_chat(client)

        idle = client.post(f"{API}/documents/{doc_id}/cancel")
        missing = client.post(f"{API}/documents/{uuid4()}/cancel")
        doc = client.get(f"{API}/documents/{doc_id}").json()

    assert (idle.status_code, idle.json()["code"]) == (409, "document_not_busy")
    assert missing.status_code == 404
    assert doc["progress"] is None
