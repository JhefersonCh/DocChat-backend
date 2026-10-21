from fakes import FakeProviderFactory
from fastapi.testclient import TestClient
from test_flow import API, _ready_chat, _settings, parse_sse

from chat_pdf.main import create_app


def test_search_hits_are_streamed_and_persisted_for_replay(empty_db_url, tmp_path):
    with TestClient(create_app(_settings(empty_db_url, tmp_path), FakeProviderFactory())) as client:
        chat_id, _ = _ready_chat(client)

        events = parse_sse(
            client.post(f"{API}/chats/{chat_id}/messages", json={"content": "plazo"}).text
        )
        [_, assistant] = client.get(f"{API}/chats/{chat_id}/messages").json()

    streamed = next(d for n, d in events if n == "search")
    [search] = assistant["searches"]
    assert streamed["found"] == search["found"] and search["panorama"] is False
    assert search["found"][0].keys() >= {"n", "chunk_id", "page_start", "page_end", "score"}
