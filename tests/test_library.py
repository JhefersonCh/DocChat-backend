"""Projects, categories and the library: pure clustering rules + the flow over HTTP (real PG)."""

from uuid import uuid4

from fakes import FakeProviderFactory
from fastapi.testclient import TestClient
from test_flow import API, _settings, _wait_ready

from chat_pdf.domain.library import blend, cluster_documents, parse_names
from chat_pdf.main import create_app


def test_clusters_separate_obvious_groups_and_names_fall_back():
    near_x = {uuid4(): [1.0, 0.05 * i, 0.0, 0.0] for i in range(4)}
    near_y = {uuid4(): [0.0, 0.0, 1.0, 0.05 * i] for i in range(3)}
    groups = cluster_documents({**near_x, **near_y})
    assert sorted(sorted(g) for g in groups) == sorted([sorted(near_x), sorted(near_y)])
    assert cluster_documents({uuid4(): [1.0, 0.0]}) == []  # too few documents
    names = parse_names('[{"grupo": 1, "nombre": "Finanzas"}, {"grupo": 1, "nombre": "x"}]', 3)
    assert names == ["Finanzas", "Grupo 2", "Grupo 3"]
    assert blend(None, 0.5) == 0.5 and blend(0.5, None) < 0.5


def _upload(client, project_id, name, text):
    res = client.post(
        f"{API}/projects/{project_id}/sources", files={"file": (name, text.encode(), "text/plain")}
    )
    assert res.status_code == 202
    _wait_ready(client, res.json()["document_id"])
    return res.json()["document_id"]


def test_project_categories_and_global_search(empty_db_url, tmp_path):
    with TestClient(create_app(_settings(empty_db_url, tmp_path), FakeProviderFactory())) as client:
        project = client.post(f"{API}/projects", json={"name": "Café"}).json()
        pid = project["id"]
        budget = _upload(client, pid, "presupuesto.txt", "El presupuesto anual es de mil pesos.")
        _upload(client, pid, "fabula.txt", "El zorro vio las uvas en la parra.")
        chat = client.post(f"{API}/projects/{pid}/chats").json()
        detail = client.get(f"{API}/projects/{pid}").json()
        chat_detail = client.get(f"{API}/chats/{chat['id']}").json()
        loose = client.post(f"{API}/chats").json()

        cat = client.post(f"{API}/categories", json={"name": "Finanzas"}).json()
        dup = client.post(f"{API}/categories", json={"name": "finanzas"})
        client.post(f"{API}/categories/{cat['id']}/assign", json={"document_ids": [budget]})
        auto = client.post(f"{API}/library/categorize").json()
        by_cat = client.get(f"{API}/library/documents", params={"category_id": cat["id"]}).json()
        found = client.post(f"{API}/library/search", json={"query": "presupuesto"}).json()
        ask = client.post(f"{API}/library/ask", json={"document_ids": [budget]}).json()

        deleted = client.delete(f"{API}/projects/{pid}")
        gone = client.get(f"{API}/projects/{pid}")
        chat_gone = client.get(f"{API}/chats/{chat['id']}")
        loose_ok = client.get(f"{API}/chats/{loose['id']}")

    assert detail["source_count"] == 2 and [s["alias"] for s in detail["sources"]] == ["S1", "S2"]
    assert chat["project_id"] == pid and detail["chats"][0]["id"] == chat["id"]
    assert len(chat_detail["sources"]) == 2  # a project chat reads the whole project
    assert dup.status_code == 409 and [d["id"] for d in by_cat] == [budget]
    assert auto["categories"] == [] and "documentos" in (auto["note"] or "")  # only 2 documents
    top = next(r for r in found["results"] if r["document"]["id"] == budget)
    assert top["document"]["project_name"] == "Café"
    assert top["hits"] and top["document"]["categories"][0]["name"] == "Finanzas"
    assert ask["chat_id"] and len(ask["document_ids"]) == 1 and ask["document_ids"][0] != budget
    assert deleted.status_code == 204 and gone.status_code == 404 and chat_gone.status_code == 404
    assert loose_ok.status_code == 200  # other chats are untouched
