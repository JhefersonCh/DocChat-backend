import asyncio
import json
from uuid import uuid4

import pytest
from fakes import (
    FakeChat,
    FakeChatRepository,
    FakeChunkStore,
    FakeDb,
    FakeDocumentRepository,
    FakeProviderFactory,
    FakeProviderRepository,
    chat_profile,
)
from fakes_artifacts import FakeArtifactRepository

from chat_pdf.application.artifacts import (
    DeleteArtifact,
    FailInterruptedArtifacts,
    GenerateArtifact,
    RequestArtifact,
)
from chat_pdf.domain.artifact_parsers import NO_NUMERIC_DATA, parse_artifact
from chat_pdf.domain.artifacts import artifact_messages, clean_options, select_chunks
from chat_pdf.domain.errors import Conflict, InvalidInput, NotFound
from chat_pdf.domain.models import Chunk

CHUNKS = [Chunk(i, f"texto {i} 12{i}", i + 1, i + 1, 0, 1, id=uuid4()) for i in range(6)]


def parse(kind, payload, options=None):
    text = payload if isinstance(payload, str) else "```json\n" + json.dumps(payload) + "\n```"
    return parse_artifact(kind, text, CHUNKS, options)


def test_page_refs_resolve_to_a_chunk_and_unknown_pages_become_null():
    data = parse("faq", {"items": [
        {"question": "¿A?", "answer": "B", "page": 3},
        {"question": "¿C?", "answer": "D", "page": 99},
        {"question": "", "answer": "x"},
    ]})  # fmt: skip
    assert [i["page"] for i in data["items"]] == [3, None]
    assert data["items"][0]["chunk_id"] == str(CHUNKS[2].id)
    assert data["items"][1]["chunk_id"] is None


def test_quiz_drops_invalid_questions():
    good = {"question": "¿Q?", "options": ["a", "b", "c", "d"], "answer_index": 2, "page": 1}
    bad = [
        {**good, "options": ["a", "b", "c"]},
        {**good, "answer_index": 4},
        {**good, "options": ["a", "a", "c", "d"]},
    ]
    assert len(parse("quiz", {"questions": [good, *bad]})["questions"]) == 1
    assert parse("quiz", {"questions": bad}) is None
    assert parse("quiz", "no es json") is None


def test_flashcards_study_guide_and_mind_map():
    assert (
        parse("flashcards", [{"front": "f", "back": "b", "page": 2}, {"front": "x"}])["cards"][0][
            "page"
        ]
        == 2
    )
    guide = parse("study_guide", {"concepts": [{"term": "t", "definition": "d"}], "glossary": 5})
    assert guide["concepts"] and guide["glossary"] == [] and guide["sections"] == []
    tree = {"title": "Raíz", "page": 1, "children": [{"title": "Hijo", "page": 2}, {"title": ""}]}
    root = parse("mind_map", tree)["root"]
    assert [c["title"] for c in root["children"]] == ["Hijo"] and root["children"][0]["page"] == 2
    assert parse("mind_map", {"children": []}) is None


def test_chart_validates_series_and_reports_missing_numeric_data():
    chart = {
        "type": "bar",
        "title": "T",
        "labels": ["a", "b"],
        "page": 2,
        "series": [{"name": "s", "values": [1, 2.5]}, {"name": "bad", "values": [1]}],
    }
    out = parse("chart", {"chart": chart})["chart"]
    assert out["series"] == [{"name": "s", "values": [1.0, 2.5]}] and out["page"] == 2
    assert parse("chart", {"chart": None}) == {"chart": None, "message": NO_NUMERIC_DATA}
    assert parse("chart", {"chart": {**chart, "type": "radar"}}) is None
    nan = {**chart, "series": [{"name": "s", "values": ["x", 1]}]}
    assert parse("chart", {"chart": nan})["chart"] is None


def test_extraction_keeps_declared_columns_with_quotes_and_drops_empty_rows():
    options = {"columns": ["Nombre", "Total"]}
    rows = {"rows": [
        {"Nombre": {"value": "Ana", "quote": "Ana Pérez", "page": 1}, "Total": 30, "Otra": 1},
        {"Nombre": {"value": None}, "Total": {"value": ""}},
    ]}  # fmt: skip
    data = parse("extraction", rows, options)
    assert data["columns"] == ["Nombre", "Total"] and len(data["rows"]) == 1
    cells = data["rows"][0]
    assert cells["Nombre"]["quote"] == "Ana Pérez" and cells["Total"]["value"] == "30"
    assert "Otra" not in cells


def test_options_are_validated():
    assert clean_options("extraction", {"columns": [" a ", "a", "", 3, "b"]}) == {
        "columns": ["a", "b"]
    }
    assert clean_options("chart", {"request": "  ventas  "}) == {"request": "ventas"}
    assert clean_options("quiz", {"x": 1}) == {}
    with pytest.raises(InvalidInput):
        clean_options("extraction", {})
    with pytest.raises(InvalidInput):
        clean_options("podcast", None)


def test_selection_stays_in_budget_and_prefers_numbers_for_charts():
    many = [Chunk(i, "x" * 2000 if i != 40 else "ventas 2020 1500 2021 1800", i + 1, i + 1, 0, 1)
            for i in range(100)]  # fmt: skip
    assert len(select_chunks(many, 16000)) <= 16000 // 900
    assert 40 in [c.ordinal for c in select_chunks(many, 4000, numeric=True)]
    user = artifact_messages("chart", {"request": "ventas"}, many)[1]["content"]
    assert "[pág. 41]" in user and "ventas" in user


# --- use cases --------------------------------------------------------------------------------


def _setup(answer, profile=True, status="ready"):
    db, artifacts = FakeDb(), FakeArtifactRepository()
    chat_id = db.add_chat().id
    doc = db.add_document(chat_id, uuid4(), status)
    db.chunks[doc.id] = CHUNKS
    cp = chat_profile()
    providers = FakeProviderRepository(cp, active={"chat": cp} if profile else {})
    chat = FakeChat(answer=answer)
    request = RequestArtifact(
        FakeChatRepository(db), FakeDocumentRepository(db), providers, artifacts
    )
    generate = GenerateArtifact(
        artifacts, FakeDocumentRepository(db), FakeChunkStore(db), providers,
        FakeProviderFactory(chat),
    )  # fmt: skip
    return chat_id, artifacts, request, generate, chat


def _run(kind, answer, options=None):
    chat_id, artifacts, request, generate, chat = _setup(answer)
    created = asyncio.run(request.execute(chat_id, kind, options))
    assert created.status == "pending"
    asyncio.run(generate.execute(created.id))
    return artifacts.items[created.id], chat


def test_generation_stores_ready_artifact_with_one_llm_call():
    artifact, chat = _run("flashcards", '{"cards": [{"front": "a", "back": "b", "page": 1}]}')
    assert artifact.status == "ready" and artifact.data["cards"][0]["page"] == 1
    assert chat.completions == 1


def test_invalid_model_output_marks_failed_with_clear_error():
    artifact, _ = _run("quiz", "lo siento, no puedo")
    assert artifact.status == "failed" and artifact.data is None and artifact.error


def test_request_requires_ready_document_and_chat_profile():
    chat_id, _, request, _, _ = _setup("{}", profile=False)
    with pytest.raises(Conflict) as exc:
        asyncio.run(request.execute(chat_id, "faq", None))
    assert exc.value.code == "no_chat_profile"
    chat_id, _, request, _, _ = _setup("{}", status="processing")
    with pytest.raises(Conflict):
        asyncio.run(request.execute(chat_id, "faq", None))
    with pytest.raises(NotFound):
        asyncio.run(request.execute(uuid4(), "faq", None))


def test_delete_is_scoped_to_chat_and_startup_fails_pending():
    chat_id, artifacts, request, _, _ = _setup("{}")
    a = asyncio.run(request.execute(chat_id, "faq", None))
    with pytest.raises(NotFound):
        asyncio.run(DeleteArtifact(artifacts).execute(uuid4(), a.id))
    asyncio.run(FailInterruptedArtifacts(artifacts).execute())
    assert artifacts.items[a.id].status == "failed"
    asyncio.run(DeleteArtifact(artifacts).execute(chat_id, a.id))
    assert not artifacts.items
