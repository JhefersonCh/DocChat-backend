from uuid import uuid4

from fakes import FakeProviderFactory
from fastapi.testclient import TestClient
from test_flow import API, _ready_chat, _settings, parse_sse

from chat_pdf.main import create_app


def _client(empty_db_url, tmp_path):
    return TestClient(create_app(_settings(empty_db_url, tmp_path), FakeProviderFactory()))


def test_chat_settings_default_patch_persist_and_validate(empty_db_url, tmp_path):
    with _client(empty_db_url, tmp_path) as client:
        chat = client.post(f"{API}/chats").json()
        patched = client.patch(
            f"{API}/chats/{chat['id']}", json={"style": "custom", "custom_instructions": "Viñetas"}
        )
        partial = client.patch(f"{API}/chats/{chat['id']}", json={"length": "larga"})
        fetched = client.get(f"{API}/chats/{chat['id']}").json()
        bad = client.patch(f"{API}/chats/{chat['id']}", json={"style": "poetic"})
        missing = client.patch(f"{API}/chats/{uuid4()}", json={"length": "corta"})

    assert (chat["style"], chat["length"], chat["custom_instructions"]) == (
        "normal",
        "normal",
        None,
    )
    assert patched.status_code == 200 and patched.json()["style"] == "custom"
    assert partial.json()["custom_instructions"] == "Viñetas"  # untouched by the second patch
    assert (fetched["style"], fetched["length"]) == ("custom", "larga")
    assert bad.status_code == 422
    assert missing.status_code == 404


def test_search_returns_scored_hits_without_calling_the_chat_model(empty_db_url, tmp_path):
    with _client(empty_db_url, tmp_path) as client:
        _, doc_id = _ready_chat(client)

        res = client.get(f"{API}/documents/{doc_id}/search", params={"q": "plazo", "k": 5})
        empty = client.get(f"{API}/documents/{doc_id}/search", params={"q": ""})
        missing = client.get(f"{API}/documents/{uuid4()}/search", params={"q": "x"})

    [hit] = res.json()["hits"]
    assert hit["chunk_id"] and hit["snippet"].startswith("El plazo")
    assert {"page_start", "page_end", "section", "score"} <= hit.keys()
    assert 0 <= hit["score"] <= 1
    assert empty.status_code == 422
    assert missing.status_code == 404


def test_done_event_and_message_carry_the_weak_evidence_flag(empty_db_url, tmp_path):
    with _client(empty_db_url, tmp_path) as client:
        chat_id, _ = _ready_chat(client)
        events = parse_sse(
            client.post(f"{API}/chats/{chat_id}/messages", json={"content": "plazo"}).text
        )
        messages = client.get(f"{API}/chats/{chat_id}/messages").json()

    assert events[-1][1]["weak_evidence"] is False  # the fake embedder gives similarity 1.0
    assert messages[1]["weak_evidence"] is False
    assert messages[1]["citations"][0]["score"] == 1.0
