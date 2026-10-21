"""Phase 7 over HTTP with a real Postgres (migration 014): add, mute, ask and remove sources."""

from uuid import uuid4

from fakes import FakeProviderFactory
from fastapi.testclient import TestClient
from test_flow import API, _ready_chat, _settings, _wait_ready, parse_sse

from chat_pdf.main import create_app


def _add(client, chat_id, name, text):
    res = client.post(
        f"{API}/chats/{chat_id}/sources", files={"file": (name, text.encode(), "text/plain")}
    )
    assert res.status_code == 202
    _wait_ready(client, res.json()["document_id"])
    return res.json()["document_id"]


def test_sources_lifecycle_and_selection(empty_db_url, tmp_path):
    with TestClient(create_app(_settings(empty_db_url, tmp_path), FakeProviderFactory())) as client:
        chat_id, first = _ready_chat(client)
        solo = client.get(f"{API}/chats/{chat_id}").json()
        second = _add(client, chat_id, "otro.txt", "El presupuesto es de mil dolares.")
        detail = client.get(f"{API}/chats/{chat_id}").json()
        listed = next(c for c in client.get(f"{API}/chats").json() if c["id"] == chat_id)

        client.patch(f"{API}/chats/{chat_id}", json={"active_source_ids": [first]})
        muted = client.get(f"{API}/chats/{chat_id}").json()
        only_first = parse_sse(
            client.post(f"{API}/chats/{chat_id}/messages", json={"content": "plazo"}).text
        )
        override = parse_sse(
            client.post(
                f"{API}/chats/{chat_id}/messages",
                json={"content": "presupuesto", "source_ids": [second]},
            ).text
        )
        client.patch(f"{API}/chats/{chat_id}", json={"active_source_ids": []})
        none_left = client.post(f"{API}/chats/{chat_id}/messages", json={"content": "plazo"})
        client.patch(f"{API}/chats/{chat_id}", json={"active_source_ids": [first, second]})
        stranger = client.patch(
            f"{API}/chats/{chat_id}", json={"active_source_ids": [str(uuid4())]}
        )

        gone = client.delete(f"{API}/chats/{chat_id}/sources/{second}")
        again = client.delete(f"{API}/chats/{chat_id}/sources/{second}")
        after = client.get(f"{API}/chats/{chat_id}").json()
        file_gone = client.get(f"{API}/documents/{second}/file")

    # a chat with one document looks the same as before: S1 is also `document`
    assert solo["document"]["id"] == first and [s["alias"] for s in solo["sources"]] == ["S1"]
    assert solo["title"] == "contrato.txt" and solo["overview"] is None
    assert [s["alias"] for s in detail["sources"]] == ["S1", "S2"]
    assert detail["document"]["id"] == first and detail["title"] == "2 fuentes"
    assert detail["overview"]["sources"][1]["alias"] == "S2" and listed["source_count"] == 2
    assert muted["active_source_ids"] == [first] and muted["disabled_sources"] == [second]
    assert [s["active"] for s in muted["sources"]] == [True, False]
    citations = next(d for n, d in only_first if n == "citations")
    assert {c["source_alias"] for c in citations} == {"S1"}
    assert citations[0]["source_id"] == first and citations[0]["source_name"] == "contrato.txt"
    assert {c["source_alias"] for c in next(d for n, d in override if n == "citations")} == {"S2"}
    assert none_left.status_code == 409 and none_left.json()["code"] == "no_active_sources"
    assert stranger.status_code == 422
    assert gone.status_code == 204 and again.status_code == 404
    assert [s["alias"] for s in after["sources"]] == ["S1"] and after["title"] == "contrato.txt"
    assert file_gone.status_code == 404
