from uuid import uuid4

import pytest

from chat_pdf.domain.models import Chunk, Source
from chat_pdf.domain.panorama import document_index, is_global_question, panorama_chunks


def _chunks(sections, size=100):
    return [
        Chunk(i, "x" * size, 1, 1, 0, 1, id=uuid4(), section=sec) for i, sec in enumerate(sections)
    ]


@pytest.mark.parametrize(
    "question",
    [
        "Dame un resumen general del proyecto",
        "RESUMÍ el documento",
        "¿De qué trata el libro?",
        "de que se trata este plan",
        "¿Cuál es el panorama?",
        "¿Cuáles son los temas principales?",
        "Dame la idea general",
        "Una sinopsis, por favor",
        "¿Qué conclusiones tiene?",
        "En general, ¿es viable?",
    ],
)
def test_global_questions_are_detected(question):
    assert is_global_question(question)


@pytest.mark.parametrize("question", ["¿Qué le enseña el zorro?", "¿Cuánto cuesta el plan?", ""])
def test_specific_questions_are_not_global(question):
    assert not is_global_question(question)


def test_index_has_summary_and_distinct_sections_in_order():
    chunks = _chunks(["A", "A", "B", "C", "B"])

    index = document_index("  Un plan\nde negocio. ", chunks)

    assert "Resumen: Un plan de negocio." in index
    assert "Secciones (3): A; B; C" in index


def test_index_without_real_sections_says_so():
    assert "no tiene secciones detectadas" in document_index(None, _chunks(["Título"] * 4))
    assert "no tiene secciones detectadas" in document_index(None, _chunks([None] * 4))
    assert "Resumen" not in document_index(None, _chunks([None]))


def test_index_is_capped_to_60_titles_spread_over_the_document():
    index = document_index(None, _chunks([f"Sec {i}" for i in range(200)]))

    shown = index.split(": ", 2)[-1].split("; ")
    assert "Secciones (200)" in index and len(shown) == 60
    assert shown[0] == "Sec 0" and shown[-1] == "Sec 199"


def test_panorama_takes_one_chunk_per_uncovered_section_within_budget():
    chunks = _chunks(["A", "A", "B", "B", "C", "C", "D", "D"])
    sources = [Source(1, chunks[0], uuid4(), "x")]  # section A already covered

    chosen = panorama_chunks(chunks, sources, budget_chars=10_000)
    assert [c.ordinal for c in chosen] == [2, 4, 6]

    tight = panorama_chunks(chunks, sources, budget_chars=250)  # 100 used, 150 left -> 1 chunk
    assert len(tight) == 1


def test_panorama_without_sections_spreads_evenly_and_skips_existing_hits():
    chunks = _chunks([None] * 10)
    sources = [Source(1, chunks[0], uuid4(), "x")]

    chosen = panorama_chunks(chunks, sources, budget_chars=500)  # 400 left -> 4 chunks

    assert len(chosen) == 4 and chunks[0] not in chosen
    assert [c.ordinal for c in chosen] == sorted(c.ordinal for c in chosen)


def test_panorama_skips_media_that_has_no_description_yet():
    text = Chunk(0, "x" * 100, 1, 1, 0, 1, id=uuid4())
    undescribed = Chunk(1, "", 1, 1, None, None, id=uuid4(), modality="image")
    described = Chunk(2, "[foto] un perro", 2, 2, None, None, id=uuid4(), modality="image")

    chosen = panorama_chunks([text, undescribed, described], [], budget_chars=10_000)

    assert chosen == [text, described]
