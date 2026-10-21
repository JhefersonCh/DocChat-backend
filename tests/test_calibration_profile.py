"""Calibration stored on the embedding profile (migration 013): validation, API, persistence."""

import asyncio

import psycopg
import pytest
from fakes import FakeProviderFactory, FakeProviderRepository, chat_profile, embedding_profile
from fastapi.testclient import TestClient
from test_flow import API, _settings

from chat_pdf.adapters.persistence.migrations import run_migrations
from chat_pdf.application.providers import ManageProviders
from chat_pdf.domain.errors import InvalidInput
from chat_pdf.main import create_app

CALIBRATION = {
    "image": {"floor": 0.6, "threshold": 0.72},
    "audio": {"floor": 0.5, "threshold": 0.8},
}


def test_manage_providers_validates_and_stores_the_calibration():
    profile = embedding_profile()
    providers = FakeProviderRepository(profile, chat_profile())
    manage = ManageProviders(providers)

    asyncio.run(
        manage.update(profile.id, {"calibration": {"image": {"floor": "0.6", "threshold": 0.7}}})
    )
    assert providers.profiles[profile.id].calibration == {"image": {"floor": 0.6, "threshold": 0.7}}

    with pytest.raises(InvalidInput):
        asyncio.run(
            manage.update(profile.id, {"calibration": {"image": {"floor": 0.9, "threshold": 0.5}}})
        )
    chat = next(p for p in providers.profiles.values() if p.kind == "chat")
    with pytest.raises(InvalidInput):
        asyncio.run(manage.update(chat.id, {"calibration": CALIBRATION}))

    asyncio.run(manage.update(profile.id, {"calibration": None}))  # back to the code defaults
    assert providers.profiles[profile.id].calibration is None


def test_013_adds_calibration_and_the_verdict_cache(empty_db_url):
    async def run():
        async with await psycopg.AsyncConnection.connect(empty_db_url) as conn:
            await run_migrations(conn)
            await conn.execute("select calibration from provider_profiles")
            await conn.execute("select chunk_id, query_key, verdict from segment_verdicts")

    asyncio.run(run())


def test_calibration_round_trips_through_the_api(empty_db_url, tmp_path):
    with TestClient(create_app(_settings(empty_db_url, tmp_path), FakeProviderFactory())) as client:
        created = client.post(
            f"{API}/providers",
            json={"kind": "embedding", "name": "e", "base_url": "http://x/v1", "model": "m"},
        ).json()
        assert created["calibration"] is None

        patched = client.patch(
            f"{API}/providers/{created['id']}", json={"calibration": CALIBRATION}
        )
        listed = next(p for p in client.get(f"{API}/providers").json() if p["id"] == created["id"])
        bad = client.patch(
            f"{API}/providers/{created['id']}",
            json={"calibration": {"image": {"floor": 0.9, "threshold": 0.1}}},
        )
        cleared = client.patch(f"{API}/providers/{created['id']}", json={"calibration": None})

    assert patched.status_code == 200 and patched.json()["calibration"] == CALIBRATION
    assert listed["calibration"] == CALIBRATION
    assert bad.status_code == 422 and bad.json()["code"] == "invalid_calibration"
    assert cleared.json()["calibration"] is None
