import json
from uuid import uuid4

from fakes import FakeChat, FakeProviderFactory
from fastapi.testclient import TestClient
from test_flow import API, _ready_chat, _settings

from chat_pdf.application.ports import ChatResult
from chat_pdf.main import create_app

QUESTIONS = json.dumps(
    {
        "questions": [
            {
                "question": "¿Cómo se fija el plazo de entrega?",
                "key_points": [
                    {
                        "point": "Es de 30 días",
                        "quote": "plazo de entrega es de 30 dias",
                        "page": 1,
                    },
                    {"point": "Se cuenta en días", "quote": "de 30 dias", "page": 1},
                    {"point": "Inventado", "quote": "no está en el texto para nada", "page": 1},
                ],
            }
        ]
    }
)


class TutorChat(FakeChat):
    def __init__(self, grade: str):
        super().__init__()
        self.grade = grade

    async def complete(self, messages, tools=None, tool_choice=None):
        system = messages[0]["content"]
        if "tutor socrático" in system:
            return ChatResult(QUESTIONS)
        if "Evaluás" in system:
            return ChatResult(self.grade)
        return await super().complete(messages, tools, tool_choice)


def test_tutor_generates_grades_and_reschedules(empty_db_url, tmp_path):
    grade = json.dumps({
        "points": [{"n": 1, "covered": True}, {"n": 2, "covered": False}],
        "feedback": "Vas bien.", "follow_up": "¿En qué unidad se mide?",
    })  # fmt: skip
    factory = FakeProviderFactory(TutorChat(grade))
    with TestClient(create_app(_settings(empty_db_url, tmp_path), factory)) as client:
        chat_id, _ = _ready_chat(client)
        base = f"{API}/chats/{chat_id}/tutor/questions"

        created = client.post(base)
        [q] = created.json()
        graded = client.post(f"{base}/{q['id']}/answer", json={"text": "La entrega es en 30 días"})
        empty = client.post(f"{base}/{q['id']}/answer", json={"text": ""})
        missing = client.post(f"{base}/{uuid4()}/answer", json={"text": "x"})
        listed = client.get(base).json()

    assert created.status_code == 201
    assert q["points"] == 2 and q["due"] is True and "rubric" not in q  # hidden until answered
    body = graded.json()
    assert body["score"] == 0.5 and body["follow_up_question"] == "¿En qué unidad se mide?"
    first, second = body["points"]
    assert first["covered"] and first["student_quote"] == "La entrega es en 30 días"
    assert not second["covered"] and second["source_quote"] == "de 30 dias"
    assert first["page"] is None  # TXT has no pages
    assert empty.status_code == 422 and missing.status_code == 404
    assert listed[0]["last_score"] == 0.5 and listed[0]["due"] is False  # partial: tomorrow


def test_tutor_rejects_invalid_model_output(empty_db_url, tmp_path):
    factory = FakeProviderFactory(TutorChat("not json"))
    with TestClient(create_app(_settings(empty_db_url, tmp_path), factory)) as client:
        chat_id, _ = _ready_chat(client)
        base = f"{API}/chats/{chat_id}/tutor/questions"
        [q] = client.post(base).json()

        res = client.post(f"{base}/{q['id']}/answer", json={"text": "algo"})

    assert res.status_code == 502 and res.json()["code"] == "invalid_provider_response"
