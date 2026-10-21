from dataclasses import replace
from uuid import uuid4

from chat_pdf.domain.calibration import calibration_table
from chat_pdf.domain.find_all import (
    MediaMatch,
    collapse_regions,
    confidence,
    is_candidate,
    is_clear,
    normalize_query,
    parse_verdict,
    reading_order,
    strength,
    verify_prompt,
)
from chat_pdf.domain.models import Chunk

TABLE = calibration_table()  # image threshold 0.66, audio 0.68, text 0.60


def _image(native=None, similarity=None, page=1, bbox=None, ordinal=0, modality="image"):
    return Chunk(
        ordinal, "d", page, page, None, None, id=uuid4(), modality=modality, bbox=bbox,
        native_similarity=native, similarity=similarity,
    )  # fmt: skip


def test_normalize_query_ignores_accents_case_and_punctuation_only():
    assert (
        normalize_query("¿Un PERRO, aquí?") == normalize_query("un perro aqui") == "un perro aqui"
    )
    assert normalize_query("perro") != normalize_query("gato")


def test_verdict_reads_the_leading_yes_or_no_and_refuses_to_guess():
    assert parse_verdict("Sí, se ve un perro abajo") is True
    assert parse_verdict("  si.") is True
    assert parse_verdict("No, es un gato") is False
    assert parse_verdict("NO se observa") is False
    assert parse_verdict("Quizás") is None
    assert parse_verdict("") is None
    assert parse_verdict("sinceramente...") is None  # a word merely starting with "si"


def test_verify_prompt_names_the_query():
    assert "«perro grande»" in verify_prompt("perro  grande")


def test_recall_takes_native_near_threshold_text_close_or_keyword_hits():
    assert is_candidate(MediaMatch(_image(native=0.63)), TABLE)  # 0.66 - 0.04 slack
    assert not is_candidate(MediaMatch(_image(native=0.61)), TABLE)  # modality-gap noise
    assert is_candidate(MediaMatch(_image(native=0.40, similarity=0.62)), TABLE)  # caption close
    assert is_candidate(MediaMatch(_image(native=0.40, similarity=0.30), fts=True), TABLE)
    assert not is_candidate(MediaMatch(_image(native=0.40, similarity=0.30)), TABLE)
    audio = MediaMatch(_image(native=0.65, modality="audio"))
    assert is_candidate(audio, TABLE)  # audio threshold 0.68 - 0.04
    assert not is_candidate(MediaMatch(_image(native=0.62, modality="audio")), TABLE)


def test_clear_band_needs_native_margin_and_a_keyword_hit():
    assert is_clear(MediaMatch(_image(native=0.75), fts=True), TABLE)  # 0.75 >= 0.66 + 0.08
    assert not is_clear(MediaMatch(_image(native=0.75), fts=False), TABLE)
    assert not is_clear(MediaMatch(_image(native=0.70), fts=True), TABLE)
    assert not is_clear(MediaMatch(_image(native=None, similarity=0.9), fts=True), TABLE)
    assert confidence(MediaMatch(_image(native=0.75), fts=True), TABLE) == "alta"
    assert confidence(MediaMatch(_image(native=0.70)), TABLE) == "media"


def test_strength_orders_the_strongest_candidate_first():
    weak = MediaMatch(_image(native=0.64))
    strong = MediaMatch(_image(native=0.80))
    keyword = MediaMatch(_image(native=0.64), fts=True)
    ranked = sorted([weak, strong, keyword], key=lambda m: -strength(m, TABLE))
    assert ranked == [strong, keyword, weak]


def test_reading_order_is_page_then_time_then_top_to_bottom():
    p2 = _image(page=2, ordinal=0)
    top = _image(page=1, bbox=[0.1, 0.1, 0.4, 0.3], ordinal=5)
    bottom = _image(page=1, bbox=[0.1, 0.6, 0.4, 0.9], ordinal=1)
    f1 = replace(_image(page=None, modality="frame", ordinal=9), t_start=30.0)
    f0 = replace(_image(page=None, modality="frame", ordinal=8), t_start=4.0)
    assert reading_order([p2, bottom, top]) == [top, bottom, p2]
    assert reading_order([f1, f0]) == [f0, f1]


def test_collapse_keeps_the_smallest_confirmed_region_of_a_page():
    page = _image(page=1, bbox=[0, 0, 1, 1], ordinal=0)
    tile = _image(page=1, bbox=[0, 0, 0.55, 0.55], ordinal=1)
    inner = _image(page=1, bbox=[0.1, 0.1, 0.3, 0.3], ordinal=2)
    elsewhere = _image(page=2, bbox=[0, 0, 1, 1], ordinal=3)

    assert collapse_regions([page, tile, inner, elsewhere]) == [inner, elsewhere]
    assert collapse_regions([page, tile]) == [tile]


def test_collapse_keeps_distinct_regions_and_regions_without_bbox():
    left = _image(page=1, bbox=[0, 0, 0.5, 1], ordinal=0)
    right = _image(page=1, bbox=[0.5, 0, 1, 1], ordinal=1)
    frame = _image(page=None, modality="frame", ordinal=2)
    same = replace(left, ordinal=3, id=uuid4())  # an equal region is not "clearly bigger"

    assert collapse_regions([left, right, frame]) == [left, right, frame]
    assert collapse_regions([left, same]) == [left, same]
