import psycopg
from fastapi.testclient import TestClient

from chat_pdf.config import Settings
from chat_pdf.main import create_app

RAW_KEY = "sk-secret-1234567890"


def test_api_key_is_encrypted_at_rest_and_never_returned(empty_db_url):
    settings = Settings(database_url=empty_db_url, app_secret_key="any string")
    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/v1/providers",
            json={"kind": "chat", "name": "x", "base_url": "http://x/v1", "model": "m",
                  "api_key": RAW_KEY},
        )  # fmt: skip
        listing = client.get("/api/v1/providers")

    assert RAW_KEY not in created.text + listing.text
    assert created.json()["api_key_masked"] == "sk-****7890"
    with psycopg.connect(empty_db_url) as conn:
        stored = conn.execute(
            "select api_key_enc from provider_profiles where name = 'x'"
        ).fetchone()[0]
    assert RAW_KEY.encode() not in bytes(stored)
