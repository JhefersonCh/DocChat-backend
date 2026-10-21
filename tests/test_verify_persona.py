import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

import psycopg
from fakes import (
    FakeChat,
    FakeChatRepository,
    FakeDb,
    FakeProviderFactory,
    FakeProviderRepository,
    chat_profile,
)
from fastapi.testclient import TestClient
from test_ask import _ask
from test_flow import API, _ready_chat, _settings

from chat_pdf.application.verification import VerifyAnswer
from chat_pdf.domain.models import Chunk, Citation, Message
from chat_pdf.domain.verification import (
    check_claims,
    lexical_overlap,
    parse_claims,
    split_sentences,
)
from chat_pdf.main import create_app

CITE = Citation(
    1, uuid4(), "x.pdf", 2, "snip", content="El plazo de entrega del pedido es de 30 días hábiles."
)


def _claim(**kw):
    return {
        "text": "El plazo de entrega es de 30 días",
        "sentence_index": 0,
        "status": "respaldada",
        "source_ns": [1],
        "reason": "lo dice",
        **kw,
    }


def test_split_sentences_drops_citations_and_markdown():
    answer = "**Plazo:** es de 30 días [1]. ¿Hay prórroga? No [2][3].\n- Pago a 60 días [1]"
    assert split_sentences(answer) == [
        "Plazo: es de 30 días.",
        "¿Hay prórroga?",
        "No.",
        "Pago a 60 días",
    ]


def test_lexical_overlap_ignores_short_words_accents_and_inflection():
    assert lexical_overlap("El plazo de entrega es de 30 días", CITE.content) == 1.0
    assert lexical_overlap("Los zorros odian las rosas", CITE.content) == 0.0
    assert lexical_overlap("de la y el", CITE.content) == 0.0  # nothing to compare


def test_check_claims_downgrades_lenient_supported_and_drops_invented_quotes():
    ok, lenient, made_up = check_claims(
        [
            _claim(quote="plazo de entrega del pedido"),
            _claim(text="La empresa fue fundada en Marte", status="respaldada"),
            _claim(status="parcial", quote="frase que no está"),
        ],
        [CITE],
    )

    assert ok["status"] == "respaldada" and ok["quote"] == "plazo de entrega del pedido"
    assert lenient["status"] == "parcial" and "coincidencia" in lenient["reason"]
    assert made_up["status"] == "parcial" and "quote" not in made_up


def test_parse_claims_drops_invalid_items_and_unknown_sources():
    raw = json.dumps(
        [_claim(source_ns=[1, 9]), {"text": "x", "status": "quizás"}, _claim(sentence_index=7)]
    )

    [claim] = parse_claims(f"```json\n{raw}\n```", 2, [CITE])

    assert claim["source_ns"] == [1]
    assert parse_claims("no json", 2, [CITE]) is None


def test_verify_answer_is_idempotent_and_persists_on_the_message():
    db, chat_prof = FakeDb(), chat_profile()
    chat_row = db.add_chat()
    message = Message(
        uuid4(), "assistant", "El plazo es de 30 días [1].", datetime.now(UTC), citations=[CITE]
    )
    db.messages[chat_row.id] = [message]
    chat = FakeChat(answer=json.dumps([_claim(quote="plazo de entrega")]))
    use_case = VerifyAnswer(
        FakeChatRepository(db),
        FakeProviderRepository(chat_prof, active={"chat": chat_prof}),
        FakeProviderFactory(chat),
    )

    first = asyncio.run(use_case.execute(chat_row.id, message.id))
    second = asyncio.run(use_case.execute(chat_row.id, message.id))

    assert first == second and first["claims"][0]["status"] == "respaldada"
    assert first["sentences"] == ["El plazo es de 30 días."]
    assert chat.completions == 1  # the second call reads the stored verification
    assert db.messages[chat_row.id][0].verification == first


def test_persona_is_injected_into_the_answer_prompt_and_stored_with_the_answer():
    chat = FakeChat()

    asked = _ask(chat, style={"persona": "el zorro"})

    system = chat.streamed[0][0]["content"]
    assert "«el zorro»" in system and "primera persona" in system and "[n]" in system
    assert asked.db.exchanges[0]["persona"] == "el zorro"
    assert "primera persona" not in _prompt(_ask(FakeChat()))


def _prompt(asked) -> str:
    return asked.factory.chat(None).streamed[0][0]["content"]


PAGED = [Chunk(i, f"chunk {i}", i + 1, i + 1, 0, 1, id=uuid4()) for i in range(10)]  # page i+1


def test_max_page_limits_retrieval_panorama_and_warns_the_model():
    chat = FakeChat()
    asked = _ask(chat, chat_profile(default_k=10), corpus=PAGED, max_page=3)

    citations = asked.events[1][1]
    assert [c.page for c in citations] == [1, 2, 3]
    system = chat.streamed[0][0]["content"]
    assert "hasta la página 3" in system and "todavía no lo leíste" in system
    unrestricted = _ask(FakeChat(), chat_profile(default_k=10), corpus=PAGED)
    assert len(unrestricted.events[1][1]) == 10


def test_max_page_also_filters_the_panorama_chunks():
    corpus = [replace(c, section=f"S{i}") for i, c in enumerate(PAGED)]
    asked = _ask(
        FakeChat(), chat_profile(default_k=2), corpus=corpus, max_page=4,
        question="Dame un resumen general",
    )  # fmt: skip

    pages = {c.page for c in asked.events[-3][1]}
    assert pages <= {1, 2, 3, 4} and len(pages) == 4


def test_http_persona_verify_and_max_page_against_postgres(empty_db_url, tmp_path):
    claim = _claim(text="El plazo de entrega es de 30 dias", quote="plazo de entrega")
    answer = json.dumps([claim])
    factory = FakeProviderFactory(FakeChat(answer=answer))
    with TestClient(create_app(_settings(empty_db_url, tmp_path), factory)) as client:
        chat_id, _ = _ready_chat(client)
        with psycopg.connect(empty_db_url, autocommit=True) as conn:
            conn.execute("update chunks set page_start = 5, page_end = 5")
        patched = client.patch(f"{API}/chats/{chat_id}", json={"persona": " el autor "})
        ask = f"{API}/chats/{chat_id}/messages"
        client.post(ask, json={"content": "plazo", "max_page": 3})
        client.post(ask, json={"content": "plazo", "max_page": 5})
        bad = client.post(ask, json={"content": "plazo", "max_page": 0})
        messages = client.get(ask).json()
        blocked, allowed = messages[1], messages[3]
        url = f"{ask}/{allowed['id']}/verify"
        before = factory.chat(None).completions  # the overview job also completes once
        first, second = client.post(url), client.post(url)
        stored = client.get(ask).json()[3]["verification"]
        cleared = client.patch(f"{API}/chats/{chat_id}", json={"persona": "  "})

    assert patched.json()["persona"] == "el autor" and cleared.json()["persona"] is None
    assert blocked["citations"] == [] and allowed["citations"][0]["page"] == 5
    assert bad.status_code == 422
    assert allowed["persona"] == "el autor" and blocked["verification"] is None
    assert first.status_code == 200 and first.json() == second.json() == stored
    assert first.json()["claims"][0]["status"] == "respaldada"
    assert factory.chat(None).completions == before + 1  # idempotent: the LLM was called once
