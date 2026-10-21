import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

from chat_pdf.domain.models import Chunk
from chat_pdf.domain.tutor import (
    Judgment,
    Sched,
    due_at,
    grade,
    lexical_support,
    next_schedule,
    order_for_review,
    parse_grade,
    parse_questions,
)

CHUNKS = [
    Chunk(
        0, "El plazo de entrega es de 30 días corridos.", 2, 2, 0, 10, id=uuid4(), section="Plazos"
    ),
    Chunk(1, "La multa por retraso asciende al 5% del contrato.", 3, 3, 0, 10, id=uuid4()),
]


def _q(points, question="¿Qué se pacta sobre plazos?"):
    return json.dumps({"questions": [{"question": question, "key_points": points}]})


def test_parse_questions_verifies_quotes_and_takes_page_from_the_chunk():
    text = _q(
        [
            {"point": "Plazo de 30 días", "quote": "plazo de entrega es de 30 dias", "page": 9},
            {"point": "Multa del 5%", "quote": "multa por retraso asciende", "page": 3},
            {"point": "Inventado", "quote": "esto no figura en el documento", "page": 1},
        ]
    )

    [q] = parse_questions(text, CHUNKS)

    assert [p["page"] for p in q["rubric"]] == [2, 3]  # claimed 9 is ignored
    assert q["rubric"][0]["chunk_id"] == str(CHUNKS[0].id) and q["rubric"][0]["section"] == "Plazos"


def test_parse_questions_drops_questions_with_fewer_than_two_verified_points_and_duplicates():
    one = _q([{"point": "a", "quote": "plazo de entrega es de 30", "page": 2}])
    assert parse_questions(one, CHUNKS) == []
    two = _q(
        [
            {"point": "a", "quote": "plazo de entrega es de 30", "page": 2},
            {"point": "b", "quote": "multa por retraso", "page": 3},
        ]
    )
    assert parse_questions(two, CHUNKS, existing=["¿qué se pacta sobre plazos?"]) == []
    assert parse_questions("not json", CHUNKS) == []


def test_parse_grade_maps_points_by_number_and_rejects_garbage():
    j = parse_grade('{"points":[{"n":2,"covered":true},{"n":9,"covered":true}],"feedback":"ok"}', 2)

    assert j == Judgment([False, True], "ok", "")
    assert parse_grade("nope", 2) is None and parse_grade('{"points": 3}', 2) is None


def test_lexical_support_quotes_the_matching_student_sentence():
    student = "No sé mucho. Creo que la entrega debe hacerse en treinta días. Fin."

    assert lexical_support(student, "Plazo de entrega de 30 días", "") == (
        "Creo que la entrega debe hacerse en treinta días"
    )
    assert lexical_support("Hablo de otra cosa", "Plazo de entrega de 30 días") is None


def test_grade_downgrades_a_yes_the_student_text_never_touches():
    rubric = [
        {
            "point": "Plazo de entrega",
            "quote": "plazo de entrega es de 30",
            "page": 2,
            "chunk_id": "a",
        },
        {"point": "Multa del contrato", "quote": "multa por retraso", "page": 3, "chunk_id": "b"},
    ]

    result = grade(
        rubric, Judgment([True, True], "bien", "¿y la multa?"), "El plazo de entrega es corto"
    )

    assert [p["covered"] for p in result["points"]] == [True, False]
    assert result["points"][0]["student_quote"] == "El plazo de entrega es corto"
    assert result["points"][1]["source_quote"] == "multa por retraso"
    assert result["score"] == 0.5 and result["follow_up_question"] == "¿y la multa?"


def test_scheduler_fail_resets_pass_grows_partial_is_tomorrow():
    s = Sched()
    failed = next_schedule(Sched(2.5, 6, 3, 0), 0.0)
    one = next_schedule(s, 1.0)
    two = next_schedule(one, 1.0)
    three = next_schedule(two, 1.0)

    assert (failed.reps, failed.interval_days, failed.lapses, failed.ease) == (0, 0, 1, 2.3)
    assert [one.interval_days, two.interval_days, three.interval_days] == [1, 3, 8.1]
    assert next_schedule(two, 0.6).interval_days == 1 and next_schedule(two, 0.6).reps == 2
    assert next_schedule(Sched(1.3, 1, 1, 0), 0.0).ease == 1.3  # floor


def test_review_order_puts_failed_and_due_before_upcoming():
    now = datetime(2026, 1, 10, tzinfo=UTC)
    mk = lambda name, days, lapses=0: SimpleNamespace(  # noqa: E731
        name=name, due_at=due_at(now, Sched(interval_days=days)), lapses=lapses
    )
    past = SimpleNamespace(name="old", due_at=now - timedelta(days=2), lapses=0)
    items = [mk("later", 3), mk("failed", 0, 1), past, mk("soon", 1)]

    assert [q.name for q in order_for_review(items, now + timedelta(seconds=1))] == [
        "old",
        "failed",
        "soon",
        "later",
    ]
