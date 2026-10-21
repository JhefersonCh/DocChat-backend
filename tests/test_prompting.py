from uuid import uuid4

from chat_pdf.domain.models import Chunk, Source
from chat_pdf.domain.prompting import build_answer_messages, parse_cited, strip_citations


def _sources():
    doc = uuid4()
    return [Source(n, Chunk(n, f"text {n}", n + 3, n + 3, 0, 1), doc, "x.pdf") for n in (1, 2, 3)]


def test_prompt_numbers_sources():
    user = build_answer_messages([], "why?", _sources())[-1]["content"]

    assert "[1] (pág. 4) text 1" in user
    assert "[3] (pág. 6) text 3" in user


def test_parser_maps_used_numbers_and_ignores_out_of_range():
    assert parse_cited("see [3] and [1, 2], also [9]", _sources()) == [1, 2, 3]


def test_strip_citations_removes_markers_with_their_leading_space():
    assert (
        strip_citations("El plazo es 30 días [1], según [2, 3].") == "El plazo es 30 días, según."
    )


def test_cite_carries_the_chunk_section():
    chunk = Chunk(1, "text", 2, 3, 0, 4, section="9. ÁREA")
    cite = Source(1, chunk, uuid4(), "x.pdf").cite(200)
    assert (cite.section, cite.page_end) == ("9. ÁREA", 3)
