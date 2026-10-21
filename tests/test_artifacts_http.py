import time
from uuid import uuid4

from fakes import FakeChat, FakeProviderFactory
from fastapi.testclient import TestClient
from test_flow import API, _ready_chat, _settings

from chat_pdf.main import create_app

CARDS = '{"cards": [{"front": "plazo", "back": "30 días", "page": null}]}'


def test_artifact_lifecycle_over_http_and_cascade_with_chat(empty_db_url, tmp_path):
    factory = FakeProviderFactory(FakeChat(answer=CARDS))
    with TestClient(create_app(_settings(empty_db_url, tmp_path), factory)) as client:
        chat_id, _ = _ready_chat(client)
        base = f"{API}/chats/{chat_id}/artifacts"

        bad = client.post(base, json={"kind": "extraction"})
        res = client.post(base, json={"kind": "flashcards"})
        for _ in range(50):
            items = client.get(base).json()
            if items[0]["status"] != "pending":
                break
            time.sleep(0.1)
        gone = client.delete(f"{base}/{res.json()['id']}")
        again = client.delete(f"{base}/{uuid4()}")
        after = client.get(base).json()

    assert bad.status_code == 422 and res.status_code == 202
    assert res.json()["status"] == "pending"
    assert items[0]["status"] == "ready" and items[0]["data"]["cards"][0]["front"] == "plazo"
    assert gone.status_code == 204 and again.status_code == 404 and after == []
