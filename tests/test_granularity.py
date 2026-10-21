"""Granularity pass (video / audio): context parents chain the children's transcripts and the
answer reads the parent's text (small-to-big) while the hit keeps its own time range."""

from dataclasses import replace
from uuid import uuid4

from chat_pdf.domain.audio import Window, context_text, plan_parents
from chat_pdf.domain.models import Chunk, Source
from chat_pdf.domain.prompting import format_source
from chat_pdf.domain.segments import with_parent_text


def _chunk(**kw):
    base = dict(ordinal=0, content="", page_start=None, page_end=None, char_start=None,
                char_end=None, modality="audio", origin="window", id=uuid4())  # fmt: skip
    return Chunk(**{**base, **kw})


def test_parent_transcript_chains_overlapping_children_without_repeating_speech():
    windows = [Window(0, 9), Window(4.5, 13.5), Window(9, 18), Window(13.5, 22.5)]
    assert [p.children for p in plan_parents(windows)] == [(0, 1, 2, 3)]
    parts = [(0, 9, "hola"), (4.5, 13.5, "a todos"), (9, 18, "bienvenidos"), (13.5, 22.5, "")]
    assert context_text(parts) == "hola bienvenidos"  # the 50 % overlap is not said twice


def test_the_answer_reads_the_parent_text_but_cites_the_fine_window():
    parent = _chunk(origin="context", content="hola bienvenidos al curso", t_start=0, t_end=30)
    hit = _chunk(content="bienvenidos", t_start=9, t_end=18, parent_id=parent.id)
    twin = replace(hit, id=uuid4())
    [expanded, repeated] = with_parent_text([hit, twin], [parent, hit])
    assert expanded.parent_text == "hola bienvenidos al curso" and repeated.parent_text is None
    line = format_source(Source(1, expanded, uuid4(), "clase.mp3"))
    assert "hola bienvenidos al curso" in line and "00:09" in line  # the hit's own range


def test_a_picture_hit_lists_its_regions_but_not_the_ones_already_hit():
    parent = _chunk(modality="image", origin="image", content="un dibujo")
    blue = _chunk(modality="image", origin="region", content="arriba a la izquierda",
                  parent_id=parent.id, region_label="círculo azul")  # fmt: skip
    red = replace(blue, id=uuid4(), region_label="cuadrado rojo", content="")
    [out, _] = with_parent_text([parent, red], [parent, blue, red])
    assert out.regions_text == "círculo azul (arriba a la izquierda)"
