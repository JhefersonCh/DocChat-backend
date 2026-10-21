"""ExhaustiveFinder in isolation: collapsing nested confirmed regions, the cap, vision budget."""

import asyncio
from uuid import uuid4

from fakes import FakeChunkStore, FakeDb, FakeVision
from test_ask_media import KeyedVision, _image

from chat_pdf.adapters.storage.local import LocalFileStorage
from chat_pdf.application.find_all import MAX_VERIFICATIONS, ExhaustiveFinder
from chat_pdf.domain.calibration import calibration_table
from chat_pdf.domain.find_all import MAX_OCCURRENCES
from chat_pdf.domain.models import Chunk


def _find(tmp_path, chunks, vision, query="perro"):
    db = FakeDb()
    doc_id = uuid4()
    db.chunks[doc_id] = chunks
    storage = LocalFileStorage(tmp_path)
    for c in chunks:
        asyncio.run(
            storage.save_asset(doc_id, "assets", c.asset_key, b"IMG" + c.asset_key.encode())
        )
    store = FakeChunkStore(db)
    finder = ExhaustiveFinder(store, store, storage, vision, calibration_table())
    return asyncio.run(finder.find(doc_id, query, [1.0, 0, 0])), store


def test_a_confirmed_tile_replaces_the_confirmed_page_that_contains_it(tmp_path):
    page = _image(1, "página escaneada", 0.70, bbox=[0.0, 0.0, 1.0, 1.0], origin="page")
    tile = _image(2, "cuadrante", 0.78, bbox=[0.0, 0.5, 0.55, 1.0], origin="tile")
    other_page = _image(3, "otra", 0.70, page=2, bbox=[0.0, 0.0, 1.0, 1.0], origin="page")
    vision = KeyedVision(**{"1": "sí", "2": "sí", "3": "no"})

    found, _ = _find(tmp_path, [page, tile, other_page], vision)

    assert [c.id for c in found.chunks] == [tile.id]
    assert (found.checked, found.discarded) == (2, 1) and found.asked == 3


def test_results_are_capped_and_the_overflow_is_counted(tmp_path):
    many = [_image(i, f"foto {i}", 0.70, page=i) for i in range(1, MAX_OCCURRENCES + 6)]

    found, _ = _find(tmp_path, many, None)  # no vision: all unverified

    assert len(found.chunks) == MAX_OCCURRENCES and found.extra == 5
    assert [c.page_start for c in found.chunks] == list(range(1, MAX_OCCURRENCES + 1))
    assert set(found.confidence.values()) == {"media"}


def test_vision_calls_are_budgeted_and_the_rest_comes_back_unverified(tmp_path):
    n = MAX_VERIFICATIONS + 4
    many = [_image(i, f"foto {i}", 0.70, page=i) for i in range(1, n + 1)]
    vision = FakeVision(*["sí"] * n)

    found, store = _find(tmp_path, many, vision)

    assert len(vision.prompts) == MAX_VERIFICATIONS == found.asked
    assert found.checked == MAX_VERIFICATIONS and len(found.confidence) == 4
    assert len(store.verdict_cache) == MAX_VERIFICATIONS


def test_candidates_below_recall_are_never_checked(tmp_path):
    noise = [_image(i, f"foto {i}", 0.55, page=i) for i in range(1, 4)]
    vision = FakeVision()

    found, _ = _find(tmp_path, noise, vision)

    assert found.chunks == [] and vision.prompts == []


def test_audio_is_searched_through_its_transcripts_without_any_vision_call(tmp_path):
    def window(i, text, similarity):
        return Chunk(
            i, text, None, None, None, None, id=uuid4(), modality="audio", origin="window",
            t_start=i * 16.0, t_end=i * 16.0 + 20, text_status="ready", similarity=similarity,
        )  # fmt: skip

    hit = window(3, "el presupuesto total es de cuarenta y cuatro millones", 0.82)
    miss = window(1, "buenos días a todos", 0.30)
    vision = FakeVision()
    db = FakeDb()
    doc_id = uuid4()
    db.chunks[doc_id] = [miss, hit]
    store = FakeChunkStore(db)
    finder = ExhaustiveFinder(store, store, None, vision, calibration_table())

    found = asyncio.run(finder.find(doc_id, "presupuesto", [1.0, 0, 0], hint="audio"))

    assert [c.id for c in found.chunks] == [hit.id]
    assert vision.prompts == [] and found.asked == 0
    assert found.confidence[hit.id] in ("alta", "media")
