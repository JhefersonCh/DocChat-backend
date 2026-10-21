import os
import uuid

import psycopg
import pytest

ADMIN_URL = os.environ.get(
    "TEST_ADMIN_DATABASE_URL", "postgresql://chatpdf:chatpdf@localhost:5433/postgres"
)


@pytest.fixture
def empty_db_url():
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
