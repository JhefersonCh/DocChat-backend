import os
import uuid

import psycopg
import pytest

ADMIN_URL = os.environ.get("TEST_ADMIN_DATABASE_URL")


@pytest.fixture
def empty_db_url():
    if not ADMIN_URL:
        pytest.skip(
            "TEST_ADMIN_DATABASE_URL is not set (admin URL of a throwaway Postgres, "
            "e.g. postgresql://USER:PASS@localhost:5433/postgres)"
        )
    name = f"chatpdf_test_{uuid.uuid4().hex[:8]}"
    try:
        admin = psycopg.connect(ADMIN_URL, autocommit=True)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres unreachable (run `docker compose up -d db`): {exc}")
    with admin:
        admin.execute(f'create database "{name}"')
        try:
            yield ADMIN_URL.rsplit("/", 1)[0] + f"/{name}"
        finally:
            admin.execute(f'drop database "{name}" with (force)')
