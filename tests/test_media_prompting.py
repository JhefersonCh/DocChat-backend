from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from chat_pdf.domain.catalog import source_catalog
from chat_pdf.domain.find_all import FindAllResult
from chat_pdf.domain.models import Chunk, Document, Source
from chat_pdf.domain.prompting import (
    build_answer_messages,
    find_all_tool,
    format_find_all,
    format_sources,
    media_locator,
    pictures_to_attach,
    search_tool,
)
from chat_pdf.domain.prompts import search_system

DOC = uuid4()


def _src(n, content="d", modality="image", **kw):
    chunk = Chunk(n, content, kw.pop("page", None), None, None, None, modality=modality, **kw)
    return Source(n, chunk, DOC, "a.pdf")


def test_text_sources_are_formatted_exactly_as_before():
    text = Source(1, Chunk(0, "hola", 3, 3, 0, 4, section="Intro"), DOC, "a.pdf")
    assert format_sources([text]) == "[1] (pág. 3) [sección: Intro] hola"


def test_image_source_shows_kind_page_zone_and_caption():
    src = _src(2, "Un perro marrón", page=3, bbox=[0.1, 0.05, 0.5, 0.3], asset_key="a.webp")
    assert format_sources([src]) == "[2] (imagen · pág. 3, arriba a la izquierda) Un perro marrón"


def test_loose_image_names_the_zone_but_never_the_whole_image():
    assert media_locator(_src(1, bbox=[0, 0, 1, 1]).chunk) == ""
    assert media_locator(_src(1, bbox=[0, 0, 0.55, 0.55]).chunk) == "arriba a la izquierda"
    assert format_sources([_src(1, "gato", bbox=[0, 0, 1, 1])]) == "[1] (imagen) gato"


def test_audio_is_quoted_with_its_time_range_and_a_frame_is_a_moment():
    audio = _src(7, "hola mundo", modality="audio", t_start=80.0, t_end=100.0)
    frame = _src(9, "pizarra", modality="frame", t_start=725.0, t_end=740.0)
    assert format_sources([audio]) == "[7] (audio · 01:20–01:40) «hola mundo»"
    assert format_sources([frame]) == "[9] (video · 12:05) pizarra"


def test_undescribed_media_says_so():
    assert (
        format_sources([_src(1, "", page=2)])
        == "[1] (imagen · pág. 2) (sin descripción disponible)"
    )


def test_pictures_to_attach_takes_the_top_three_with_files():
    sources = [
        _src(1, modality="text", asset_key=None),
        _src(2, asset_key=None),
        *[_src(n, asset_key=f"{n}.webp") for n in (3, 4, 5, 6)],
    ]
    assert [s.n for s in pictures_to_attach(sources)] == [3, 4, 5]


def test_images_travel_only_in_the_user_message():
    sources = [_src(1, "perro", page=1, asset_key="1.webp")]
    messages = build_answer_messages(
        [{"role": "assistant", "content": "antes"}], "¿hay un perro?", sources,
        images=[(1, "data:image/webp;base64,AAA")],
    )  # fmt: skip

    system, history, user = messages
    assert isinstance(system["content"], str) and "adjuntan" in system["content"]
    assert history["content"] == "antes"
    parts = user["content"]
    assert parts[0]["type"] == "text" and "¿hay un perro?" in parts[0]["text"]
    assert parts[-1] == {"type": "image_url", "image_url": {"url": "data:image/webp;base64,AAA"}}
    assert "[1]" in parts[1]["text"]
    assert sum(isinstance(m["content"], list) for m in messages) == 1


def test_without_images_or_media_the_prompt_is_unchanged():
    text = Source(1, Chunk(0, "hola", 1, 1, 0, 4), DOC, "a.pdf")
    [system, user] = build_answer_messages([], "q", [text])
    assert isinstance(user["content"], str) and "imagen" not in system["content"].lower()

    [system, user] = build_answer_messages([], "q", [_src(1, "perro")])
    assert isinstance(user["content"], str)
    assert "no podés ver" in system["content"] and "adjuntan" not in system["content"]


def test_search_tool_exposes_modality_only_for_documents_with_media():
    plain = search_tool(5)["function"]["parameters"]["properties"]
    media = search_tool(5, media=True)["function"]["parameters"]["properties"]
    assert "modality" not in plain
    assert media["modality"]["enum"] == ["auto", "texto", "imagen", "audio", "video"]
    assert find_all_tool()["function"]["parameters"]["required"] == ["query"]


def test_planner_prompt_mentions_find_all_only_with_media():
    assert "find_all" not in search_system(3, 5)
    assert "find_all" in search_system(3, 5, media=True)


def test_find_all_result_lists_occurrences_with_locator_and_verification():
    dog = _src(1, "perro", page=1, bbox=[0.1, 0.6, 0.4, 0.9])
    result = FindAllResult([dog], {}, checked=1, discarded=2)
    text = format_find_all("un perro", result)
    assert "1 ocurrencias de «un perro»" in text and "verificadas una por una" in text
    assert "[1] (imagen · pág. 1, abajo a la izquierda) perro" in text


def test_find_all_labels_confidence_when_nothing_was_verified():
    cat = _src(1, "gato", page=2)
    text = format_find_all("gato", FindAllResult([cat], {1: "media"}))
    assert "sin verificación visual" in text and text.endswith("[confianza media]")


def test_find_all_says_nothing_was_found_and_how_many_were_discarded():
    text = format_find_all("perro", FindAllResult([], {}, discarded=3))
    assert "ninguna ocurrencia" in text and "descartaron 3" in text
    assert "descartaron" not in format_find_all("perro", FindAllResult([], {}))


def test_find_all_reports_truncation():
    text = format_find_all("x", FindAllResult([_src(1, "a", page=1)], {}, checked=1, extra=7))
    assert "7 ocurrencias más" in text and "tope 40" in text


def _doc(**kw):
    return Document(
        DOC, DOC, "informe.pdf", "application/pdf", "ready", None, 4, DOC, datetime.now(UTC), **kw
    )


def test_catalog_counts_media_and_is_empty_for_text_only_sources():
    text = Chunk(0, "t", 1, 1, 0, 1)
    assert source_catalog(_doc(), [text]) == ""
    image = replace(_src(1, page=1).chunk, origin="embedded_image", text_status="ready")
    undescribed = replace(image, text_status="pending")
    tile = replace(image, origin="tile")
    line = source_catalog(_doc(), [text, image, undescribed, tile])
    assert line.splitlines()[0] == "Fuentes del chat:"
    assert "S1 · informe.pdf · PDF 4 págs · 2 imágenes (1 descrita)" in line


def test_catalog_for_a_video_shows_duration_frames_and_audio():
    frame = _src(1, modality="frame", text_status="ready").chunk
    audio = _src(2, modality="audio", text_status="pending").chunk
    doc = replace(_doc(), kind="video", filename="clase.mp4", page_count=None, duration_s=3480.0)
    line = source_catalog(doc, [frame, audio]).splitlines()[1]
    assert line == (
        "S1 · clase.mp4 · video · 58:00 · 1 frame de video (1 descrita)"
        " · 1 fragmento de audio (0 descritas)"
    )


def test_verify_prompt_labels_media_sources_as_descriptions_or_transcripts():
    from chat_pdf.domain.models import Citation
    from chat_pdf.domain.verification import verify_messages

    cites = [
        Citation(1, DOC, "a.pdf", 1, "s", content="texto", kind="text"),
        Citation(2, DOC, "a.pdf", 1, "s", content="un perro", kind="image"),
        Citation(3, DOC, "a.mp3", None, "s", content="hola", kind="audio"),
    ]

    user = verify_messages(["Hay un perro."], cites)[1]["content"]

    assert "[1] texto" in user
    assert "[2] (descripción de una imagen) un perro" in user
    assert "[3] (transcripción de audio) hola" in user


def test_an_empty_search_tells_the_model_to_say_it_does_not_appear():
    [system, user] = build_answer_messages([], "¿hay un perro?", [])

    assert "ninguna" in user["content"] and "no aparece" in user["content"]
    assert user["content"].endswith("Pregunta: ¿hay un perro?")


def test_the_speech_of_a_video_is_cited_as_video_with_its_time_range():
    from uuid import uuid4

    from chat_pdf.domain.models import Chunk, Source
    from chat_pdf.domain.prompting import format_source

    speech = Chunk(0, "hola mundo", None, None, None, None, modality="audio", t_start=80, t_end=100)
    frame = Chunk(1, "un perro", None, None, None, None, modality="frame", t_start=725)

    assert (
        format_source(Source(1, speech, uuid4(), "v.mp4", "video"))
        == "[1] (video · 01:20–01:40) «hola mundo»"
    )
    assert (
        format_source(Source(2, frame, uuid4(), "v.mp4", "video")) == "[2] (video · 12:05) un perro"
    )
    assert format_source(Source(3, speech, uuid4(), "a.mp3", "audio")).startswith(
        "[3] (audio · 01:20"
    )
