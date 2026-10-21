from chat_pdf.domain.generation import (
    SAMPLE_BUDGET,
    parse_overview,
    parse_questions,
    sample_text,
)
from chat_pdf.domain.models import Chunk


def test_parse_questions_tolerates_fences_prose_and_junk():
    fenced = '```json\n["¿Uno?", "¿Dos?", "¿Uno?", "", 7, "¿Tres?", "¿Cuatro?"]\n```'
    assert parse_questions(fenced) == ["¿Uno?", "¿Dos?", "¿Tres?"]  # deduped, junk out, max 3
    assert parse_questions('Claro: ["¿A?", "¿B?"] espero que sirva') == ["¿A?", "¿B?"]
    assert parse_questions('{"questions": ["¿A?"]}') == ["¿A?"]
    assert parse_questions("no sé") == []


def test_parse_overview_needs_a_summary_but_not_all_questions():
    text = 'Aquí va:\n```json\n{"summary": " Un libro. ", "questions": ["¿A?", null]}\n```'
    overview = parse_overview(text)
    assert overview and (overview.summary, overview.questions) == ("Un libro.", ["¿A?"])
    assert parse_overview('{"summary": "", "questions": ["¿A?"]}') is None
    assert parse_overview('["¿A?"]') is None


def test_sampling_stays_in_budget_and_spreads_over_the_whole_document():
    chunks = [
        Chunk(i, f"<{i}>" + "x" * 1400, None, None, 0, 1, section=f"S{i}") for i in range(500)
    ]

    sections, excerpts = sample_text(chunks)

    assert len(sections) + len(excerpts) <= SAMPLE_BUDGET
    used = [int(t.split(">")[0]) for t in (b[1:] for b in excerpts.split("\n\n"))]
    assert used[:3] == [0, 1, 2]  # the start is always there
    assert used == sorted(used) and used[-1] == 499 and used[len(used) // 2] > 150
