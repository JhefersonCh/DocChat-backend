from dataclasses import replace

import pytest

from chat_pdf.domain.calibration import (
    Calibration,
    calibration_table,
    chunk_margin,
    chunk_relevance,
    fit_calibration,
    modality_relevance,
    parse_calibration,
)
from chat_pdf.domain.errors import InvalidInput
from chat_pdf.domain.models import Chunk

TEXT = Chunk(0, "t", 1, 1, 0, 1)
IMAGE = Chunk(1, "d", 1, 1, None, None, modality="image")


def test_table_layers_defaults_then_app_text_threshold_then_profile():
    assert calibration_table()["image"] == Calibration(0.55, 0.66)
    assert calibration_table(text_threshold=0.7)["text"].threshold == 0.7
    table = calibration_table(
        {"image": {"floor": 0.6, "threshold": 0.7}, "text": {"floor": 0.5, "threshold": 0.65}},
        text_threshold=0.9,
    )
    assert table["image"] == Calibration(0.6, 0.7)
    assert table["text"].threshold == 0.65  # the profile wins over the app default
    assert table["audio"] == calibration_table()["audio"]


def test_relevance_uses_the_profile_floor():
    table = calibration_table({"image": {"floor": 0.8, "threshold": 0.9}})
    assert modality_relevance(0.7, "image", table) == 0.0
    assert modality_relevance(0.9, "image", table) == pytest.approx(0.5)
    assert modality_relevance(0.9, "frame", table) == pytest.approx(0.5)  # frame = image space


def test_parse_calibration_validates_and_normalizes():
    assert parse_calibration({"image": {"floor": "0.5", "threshold": 0.7}}) == {
        "image": {"floor": 0.5, "threshold": 0.7}
    }
    for bad in (
        {"video": {"floor": 0.5, "threshold": 0.7}},
        {"image": {"floor": 0.8, "threshold": 0.7}},
        {"image": {"floor": 0.5}},
        {"image": {"floor": 1.0, "threshold": 1.0}},
        {"image": {"floor": "x", "threshold": 0.7}},
    ):
        with pytest.raises(InvalidInput):
            parse_calibration(bad)


def test_text_chunk_score_and_margin_are_judged_as_text():
    chunk = replace(TEXT, similarity=0.8)
    assert chunk_relevance(chunk) == 0.6  # (0.8 - 0.5) / 0.5
    assert chunk_margin(chunk) == pytest.approx(0.2)
    assert chunk_relevance(TEXT) is None and chunk_margin(TEXT) is None


def test_media_chunk_takes_the_best_of_native_and_caption_space():
    chunk = replace(IMAGE, native_similarity=0.80, similarity=0.55)
    native = modality_relevance(0.80, "image")
    assert chunk_relevance(chunk) == round(native, 3)
    assert chunk_margin(chunk) == pytest.approx(0.80 - 0.66)  # vs the image threshold, not 0.60
    unrelated_image = replace(IMAGE, native_similarity=0.62)  # above the text 0.60 but not image
    assert chunk_margin(unrelated_image) < 0


def test_fit_separable_data_puts_the_threshold_between_the_classes():
    scores = [0.67, 0.72, 0.81, 0.50, 0.55, 0.65]
    labels = [True, True, True, False, False, False]
    fit = fit_calibration(scores, labels)
    assert fit.floor == 0.55  # median of 0.50 / 0.55 / 0.65
    assert fit.threshold == 0.66  # halfway between 0.65 and 0.67


def test_fit_overlapping_data_maximizes_f1():
    scores = [0.9, 0.8, 0.7, 0.6, 0.75, 0.5, 0.4]
    labels = [True, True, False, True, False, False, False]
    # F1 by cutoff: 0.9 -> .5, 0.8 -> .8, 0.75 -> .667, 0.6 -> .75
    fit = fit_calibration(scores, labels)
    assert fit.threshold == pytest.approx(0.775)  # halfway to 0.75, the next score below 0.8


def test_fit_threshold_never_falls_below_the_floor():
    fit = fit_calibration([0.9, 0.9, 0.1], [True, False, False])
    assert fit.threshold >= fit.floor


def test_fit_needs_both_classes():
    with pytest.raises(InvalidInput):
        fit_calibration([0.9, 0.8], [True, True])
    with pytest.raises(ValueError):
        fit_calibration([0.9], [True, False])
