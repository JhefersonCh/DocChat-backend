from uuid import uuid4

from fakes import FakeProviderFactory
from fastapi.testclient import TestClient

from chat_pdf.config import Settings
from chat_pdf.main import create_app


def test_every_error_has_the_same_code_and_message_shape(empty_db_url):
    settings = Settings(database_url=empty_db_url, app_secret_key="any string")
    with TestClient(create_app(settings, FakeProviderFactory())) as client:
        not_found = client.get(f"/api/v1/chats/{uuid4()}")
        invalid = client.get("/api/v1/chats/not-a-uuid")

    assert not_found.status_code == 404
    assert set(not_found.json()) == {"code", "message"}
    assert not_found.json()["code"] == "chat_not_found"
    assert invalid.status_code == 422
    assert set(invalid.json()) == {"code", "message"}
    assert invalid.json()["code"] == "invalid_input"
