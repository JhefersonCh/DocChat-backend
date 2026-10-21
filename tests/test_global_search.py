"""Global keyword search: accents/case, partial words, AND, quoted phrases, snippets, HTTP."""

import psycopg
from fakes import FakeProviderFactory
from fastapi.testclient import TestClient
from test_flow import API, _settings

from chat_pdf.domain.chat_search import make_snippet, parse_query
from chat_pdf.main import create_app


def test_snippet_and_query_rules():
    assert parse_query('Presu "plan Anual" presu') == ["presu", "plan anual"]
    text = "x " * 80 + "La **Inversión** inicial [1] fue alta. " + "y " * 80
    snip = make_snippet(text, parse_query("inversion"))
    s, e = snip.highlights[0]
    assert snip.text[s:e] == "Inversión" and "[1]" not in snip.text and snip.text.startswith("…")


def test_global_search_over_http(empty_db_url, tmp_path):
    with TestClient(create_app(_settings(empty_db_url, tmp_path), FakeProviderFactory())) as client:
        project = client.post(f"{API}/projects", json={"name": "Café"}).json()
        pchat = client.post(f"{API}/projects/{project['id']}/chats").json()
        loose = client.post(f"{API}/chats").json()
        with psycopg.connect(empty_db_url, autocommit=True) as db:
            for chat, role, text in [
                (loose["id"], "user", "¿Cuál es el presupuesto del proyecto?"),
                (pchat["id"], "assistant", "El **Presupuesto** anual es de mil pesos [1]."),
                (pchat["id"], "user", "El zorro vio las uvas"),
            ]:
                db.execute(
                    "insert into messages (chat_id, role, content) values (%s, %s, %s)",
                    (chat, role, text),
                )
        partial = client.get(f"{API}/search", params={"q": "PRESU"}).json()
        both = client.get(f"{API}/search", params={"q": "zorro uvas"}).json()
        phrase = client.get(f"{API}/search", params={"q": '"vio las"'}).json()
        wrong_order = client.get(f"{API}/search", params={"q": '"las vio"'}).json()
        proj = client.get(f"{API}/search", params={"q": "cafe"}).json()
        empty = client.get(f"{API}/search", params={"q": "  "}).json()

    hits = {m["role"]: m for m in partial["messages"]}
    assert len(hits) == 2 and hits["assistant"]["project_name"] == "Café"
    assert hits["user"]["project_id"] is None
    h = hits["assistant"]
    s, e = h["highlights"][0]
    assert h["snippet"][s:e] == "Presu" and "[1]" not in h["snippet"]
    assert len(both["messages"]) == 1 and len(phrase["messages"]) == 1
    assert wrong_order["messages"] == [] and empty["messages"] == []
    assert [p["name"] for p in proj["projects"]] == ["Café"]
