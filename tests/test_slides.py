import json
from uuid import uuid4

from chat_pdf.domain.artifacts import clean_options
from chat_pdf.domain.models import Chunk
from chat_pdf.domain.slides import parse_assessment, parse_deck, slide_batches

DOC = uuid4()
TEXTS = [
    "Las ventas crecieron: 2023 fue de 120 millones y 2024 de 150 millones de pesos.",
    "El equipo de ingeniería lo lidera María Gómez desde marzo.",
]
IMG = Chunk(5, "Gráfico de barras de ventas", 1, 1, None, None, id=uuid4(), modality="image",
            origin="figure", asset_key="a.webp", text_status="ready", document_id=DOC)  # fmt: skip
CHUNKS = [
    *(Chunk(i, t, i + 1, i + 1, 0, 1, id=uuid4(), document_id=DOC) for i, t in enumerate(TEXTS)),
    IMG,
]
ALIASES = {DOC: "S1"}


def deck(slides):
    return json.dumps({"title": "Ventas", "subtitle": "2024", "slides": slides})


def test_deck_parser_verifies_refs_visuals_and_charts():
    ref = {"quote": "2024 de 150 millones de pesos", "source": "S1", "page": 1}
    chart = {"type": "bar", "title": "Ventas", "labels": ["2023", "2024"],
             "series": [{"name": "COP", "values": [120, 150]}]}  # fmt: skip
    fake = {**chart, "series": [{"name": "s", "values": [7, 9]}]}
    slides = [
        {"layout": "title", "title": "Ventas"},
        {"layout": "chart", "title": "Crecimiento", "bullets": ["Subió"], "refs": [ref],
         "visual": {"kind": "chart", "spec": chart}},
        {"layout": "image", "title": "Figura", "bullets": ["Ventas"], "refs": [ref],
         "visual": {"kind": "image", "segment_id": "V1"}},
        {"layout": "bullets", "title": "Inventada", "bullets": ["x"],
         "refs": [{"quote": "cita que no existe en ningún lado"}]},
        {"layout": "chart", "title": "Cifras falsas", "bullets": ["y"], "refs": [ref],
         "visual": {"kind": "chart", "spec": fake}},
        {"layout": "bullets", "title": "Equipo", "bullets": ["Lidera María"],
         "refs": [{"quote": "lo lidera Maria Gomez", "page": 9}, {"quote": "inexistente xyz"}]},
        {"layout": "closing", "title": "Gracias"},
    ]  # fmt: skip
    out = parse_deck([deck(slides)], CHUNKS, ALIASES, {"slides": 8})
    titles = [s["title"] for s in out["slides"]]
    assert titles == ["Ventas", "Crecimiento", "Figura", "Cifras falsas", "Equipo", "Gracias"]
    by = {s["title"]: s for s in out["slides"]}
    assert by["Crecimiento"]["layout"] == "chart" and by["Crecimiento"]["visual"]["kind"] == "chart"
    assert by["Figura"]["visual"]["segment_id"] == str(IMG.id)
    assert by["Cifras falsas"]["layout"] == "bullets" and by["Cifras falsas"]["visual"] is None
    team = by["Equipo"]["refs"]
    assert len(team) == 1 and team[0]["page"] == 2 and team[0]["chunk_id"] == str(CHUNKS[1].id)
    assert parse_deck([deck(slides[3:4])], CHUNKS, ALIASES) is None  # nothing verifiable left


def test_options_assessment_and_batches():
    assert clean_options("slides", {"slides": 99, "style": "didáctica", "audience": " Jefes "}) == {
        "slides": 24, "style": "didáctica", "audience": "Jefes"}  # fmt: skip
    assert clean_options("slides", {"style": "otro"}) == {}
    ok = parse_assessment('{"suitable": true, "score": 0.8, "reasons": ["Hay estructura"],'
                          ' "recommended_slides": 40, "outline_hint": ["Intro"]}', 8)  # fmt: skip
    assert ok["suitable"] and ok["recommended_slides"] == 24 and ok["audience"] is None
    assert not parse_assessment('{"suitable": true, "score": 0.1}', 8)["suitable"]
    assert parse_assessment("sin json", 8) is None
    calls = slide_batches({}, CHUNKS, ALIASES, "S1 · a.pdf")
    assert len(calls) == 1 and "V1" in calls[0][1]["content"]
