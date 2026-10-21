import pytest

from chat_pdf.domain.calibration import DEFAULT_CALIBRATION, above_threshold, modality_relevance
from chat_pdf.domain.errors import InvalidInput
from chat_pdf.domain.locator import (
    format_locator,
    format_range,
    format_time,
    region_label,
    validate_bbox,
    validate_range,
)


@pytest.mark.parametrize(
    ("seconds", "expected"), [(0, "00:00"), (200, "03:20"), (59.9, "00:59"), (3723, "1:02:03")]
)
def test_format_time(seconds, expected):
    assert format_time(seconds) == expected


def test_format_range_uses_an_en_dash():
    assert format_range(200, 220) == "03:20–03:40"


@pytest.mark.parametrize(
    ("bbox", "label"),
    [
        ([0.0, 0.0, 0.2, 0.2], "arriba a la izquierda"),
        ([0.4, 0.0, 0.6, 0.2], "arriba"),
        ([0.8, 0.0, 1.0, 0.2], "arriba a la derecha"),
        ([0.0, 0.4, 0.2, 0.6], "a la izquierda"),
        ([0.4, 0.4, 0.6, 0.6], "centro"),
        ([0.8, 0.4, 1.0, 0.6], "a la derecha"),
        ([0.0, 0.8, 0.2, 1.0], "abajo a la izquierda"),
        ([0.4, 0.8, 0.6, 1.0], "abajo"),
        ([0.8, 0.8, 1.0, 1.0], "abajo a la derecha"),
        ([0.0, 0.0, 1.0, 1.0], "toda la página"),
    ],
)
def test_region_label_uses_page_thirds(bbox, label):
    assert region_label(bbox) == label


def test_format_locator_combines_page_region_and_time():
    assert format_locator(page=12, bbox=[0, 0, 0.2, 0.2]) == "pág. 12, arriba a la izquierda"
    assert format_locator(page=3) == "pág. 3"
    assert format_locator(t_start=200, t_end=220) == "03:20–03:40"
    assert format_locator() == ""


@pytest.mark.parametrize(
    "bbox", [[0, 0, 1], [0.5, 0, 0.4, 1], [-0.1, 0, 1, 1], [0, 0, 1.2, 1], [0, 0, 0, 1]]
)
def test_validate_bbox_rejects_malformed_regions(bbox):
    with pytest.raises(InvalidInput):
        validate_bbox(bbox)


def test_validate_bbox_returns_floats():
    assert validate_bbox([0, 0, 1, 1]) == [0.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize("span", [(-1, 5), (5, 5), (6, 5)])
def test_validate_range_rejects_empty_or_negative(span):
    with pytest.raises(InvalidInput):
        validate_range(*span)


def test_relevance_is_normalized_by_each_modalitys_floor():
    floor = DEFAULT_CALIBRATION["image"].floor
    assert modality_relevance(floor, "image") == 0.0
    assert modality_relevance(0.2, "image") == 0.0  # below the floor: unrelated
    assert modality_relevance(1.0, "image") == 1.0
    mid = modality_relevance((floor + 1) / 2, "image")
    assert mid == pytest.approx(0.5, abs=1e-3)


def test_same_cosine_means_less_for_media_than_for_text():
    assert modality_relevance(0.7, "image") < modality_relevance(0.7, "text")


def test_threshold_is_per_modality():
    # 0.62 clears the text threshold but not the image one: the modality gap
    assert above_threshold(0.62, "text") and not above_threshold(0.62, "image")
