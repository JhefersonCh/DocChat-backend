from uuid import uuid4

from chat_pdf.domain.models import Chunk, Source
from chat_pdf.domain.retrieval import (
    MODALITY_HINTS,
    fit_sources,
    is_weak_evidence,
    parse_hint,
    plan_rankings,
    reciprocal_rank_fusion,
)


def test_fit_sources_keeps_whole_sources_in_order_within_budget_and_at_least_one():
    doc = uuid4()
    sources = [Source(n, Chunk(n, "x" * 100, 1, 1, 0, 1), doc, "a.pdf") for n in (1, 2, 3)]

    assert [s.n for s in fit_sources(sources, 250)] == [1, 2]
    assert [s.n for s in fit_sources(sources, 10)] == [1]
    assert fit_sources([], 10) == []


def test_rrf_rewards_chunks_both_rankings_agree_on_and_breaks_ties_by_first_ranking():
    a, b, c, d = (Chunk(i, str(i), 1, 1, 0, 1, id=uuid4()) for i in range(4))

    fused = reciprocal_rank_fusion([[a, b, c], [d, c]], k=60)

    assert fused == [c, a, d, b]  # c is in both; a and d tie at rank 1, vector list first


# --- Phase 3: modality hints, weighted native rankings, weak evidence per modality --------------


def _chunks(n):
    return [Chunk(i, str(i), 1, 1, None, None, id=uuid4()) for i in range(n)]


def test_hint_parsing_is_lenient_and_defaults_to_auto():
    assert [parse_hint(h) for h in ("Imagen", "audio", "TEXTO", "video", "image", "foto")] == [
        "imagen", "audio", "texto", "video", "imagen", "imagen"
    ]  # fmt: skip
    assert parse_hint(None) == parse_hint("xyz") == parse_hint("") == "auto"


def test_plan_rankings_weights_and_modality_filters():
    plan = {h: plan_rankings(h, 0.3, 0.7, 1.5) for h in MODALITY_HINTS}

    assert plan["auto"].weights == (1.0, 0.3, 0.7, 0.7) and plan["auto"].modalities is None
    assert plan["imagen"].weights == (1.0, 0.3, 1.5, 0.0)
    assert plan["imagen"].modalities == ("image", "frame")
    assert plan["audio"].weights == (1.0, 0.3, 0.0, 1.5) and plan["audio"].modalities == ("audio",)
    assert plan["video"].weights == (1.0, 0.3, 1.5, 1.5)
    assert plan["texto"].weights == (1.0, 0.3, 0.0, 0.0) and plan["texto"].modalities == ("text",)


def test_native_ranking_weight_decides_whether_an_image_beats_a_text_hit():
    text_hit, image_hit = _chunks(2)
    rankings = [[text_hit], [], [image_hit], []]  # each is rank 1 of its own ranking

    default = reciprocal_rank_fusion(rankings, k=10, weights=[1.0, 0.3, 0.7, 0.7])
    boosted = reciprocal_rank_fusion(rankings, k=10, weights=[1.0, 0.3, 1.5, 0.0])

    assert default == [text_hit, image_hit]  # 1/11 > 0.7/11
    assert boosted == [image_hit, text_hit]  # the "imagen" hint lifts the visual ranking


def test_a_segment_found_by_both_native_and_text_rankings_outranks_single_hits():
    both, only_text, only_image = _chunks(3)

    fused = reciprocal_rank_fusion(
        [[only_text, both], [], [only_image, both], []], k=10, weights=[1.0, 0.3, 0.7, 0.7]
    )

    assert fused[0] == both


def _media_source(native=None, similarity=None, modality="image"):
    chunk = Chunk(
        0, "d", 1, 1, None, None, modality=modality, similarity=similarity, native_similarity=native
    )
    return Source(1, chunk, uuid4(), "a.pdf")


def test_weak_evidence_uses_each_modalitys_threshold():
    # 0.63 would pass the text threshold (0.60) but is under the image one (0.66): weak
    assert is_weak_evidence([_media_source(native=0.63)], 0.6)
    assert not is_weak_evidence([_media_source(native=0.70)], 0.6)
    assert not is_weak_evidence([_media_source(native=0.5, similarity=0.7)], 0.6)  # caption rescues
    assert not is_weak_evidence([_media_source(native=0.80, modality="audio")], 0.6)
    assert is_weak_evidence([_media_source(native=0.66, modality="audio")], 0.6)  # audio needs .68


def test_weak_evidence_honors_the_profile_calibration():
    calibration = {"image": {"floor": 0.5, "threshold": 0.6}}
    assert not is_weak_evidence([_media_source(native=0.63)], 0.6, calibration)
