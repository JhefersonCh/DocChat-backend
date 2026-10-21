import pytest

from chat_pdf.domain.chunking import chunk_pages
from chat_pdf.domain.config import RagConfig
from chat_pdf.domain.models import Page

SENTENCES = " ".join(f"La oración número {i} habla del cacao amazónico." for i in range(40))


def _joined(pages: list[Page]) -> str:
    return "\n\n".join(p.text for p in pages)


def test_chunks_keep_page_range_and_offsets_across_page_boundary():
    pages = [Page(1, "alpha " * 10), Page(2, "beta " * 30)]

    chunks = chunk_pages(pages, RagConfig(chunk_size=100, chunk_overlap=10))

    text = _joined(pages)
    assert len(chunks) > 1
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert any(c.page_start == 1 and c.page_end == 2 for c in chunks)
    assert all(text[c.char_start : c.char_end] == c.content for c in chunks)


def test_chunks_respect_size_cut_on_sentences_and_overlap():
    pages = [Page(1, SENTENCES)]

    chunks = chunk_pages(pages, RagConfig(chunk_size=300, chunk_overlap=60))

    assert all(len(c.content) <= 300 for c in chunks)
    assert all(c.content.startswith("La oración") and c.content.endswith(".") for c in chunks)
    assert all(b.char_start < a.char_end for a, b in zip(chunks, chunks[1:], strict=False))


@pytest.mark.parametrize(
    ("page", "section"),
    [  # layout headings (PDF font size) win; without them, text rules (markdown / numbered)
        (
            Page(
                1,
                "intro.\nS E C C I Ó N 0 9 \n9. Área geográfica\nSerá en Mocoa.",
                ("9. Área geográfica",),
            ),
            "9. Área geográfica",
        ),  # noqa: E501
        (Page(None, "intro.\n## Área geográfica\nSerá en Mocoa."), "Área geográfica"),
        (Page(None, "intro.\n9. Área geográfica\nSerá en Mocoa."), "9. Área geográfica"),
    ],
)
def test_headings_start_a_section_that_titles_its_chunks(page, section):
    chunks = chunk_pages([page], RagConfig(chunk_size=200, chunk_overlap=20))

    first, last = chunks[0], chunks[-1]
    assert (first.content, first.section) == ("intro.", None)
    assert last.section == section and last.content.endswith("Será en Mocoa.")
    assert "Mocoa" not in first.content
    if page.headings:  # the letter-spaced kicker belongs to its heading; content stays verbatim
        assert last.content.startswith("S E C C I Ó N 0 9")
        assert last.index_text == "SECCIÓN 09 9. Área geográfica Será en Mocoa."


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (80, -1), (80, 40), (80, 90)])
def test_config_rejects_sizes_that_would_loop_forever(size, overlap):
    with pytest.raises(ValueError):
        RagConfig(chunk_size=size, chunk_overlap=overlap)


def test_roman_numeral_chapters_start_sections_even_when_the_title_is_a_layout_heading():
    body = "Texto del capítulo con varias palabras. " * 3
    page = Page(
        1,
        f"Título del libro\nI.\nIntroducción\n{body}\nII\nMétodos\n{body}\nCAPÍTULO 3\n{body}",
        ("Título del libro", "Introducción", "Métodos"),
    )

    chunks = chunk_pages([page], RagConfig(chunk_size=200, chunk_overlap=20))

    sections = list(dict.fromkeys(c.section for c in chunks))
    assert sections == ["Título del libro", "I. Introducción", "II. Métodos", "CAPÍTULO 3"]
    assert (
        chunks[0].content.startswith("Título del libro") and "Introducción" not in chunks[0].content
    )
    assert chunks[-1].section == "CAPÍTULO 3"
