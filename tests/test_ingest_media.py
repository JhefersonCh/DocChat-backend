"""Stage 1 of image ingest: regions -> crops, thumbs, native embeddings, then `ready`."""

import asyncio
from dataclasses import replace

import pytest
from fakes import (
    FakeDb,
    FakeFileStorage,
    FakeLoader,
    FakeMediaExtractor,
    FakeMultimodalEmbedder,
    FakeProviderFactory,
    FakeProviderRepository,
    FakeUnitOfWork,
    embedding_profile,
    vision_profile,
)

from chat_pdf.application.documents import NO_IMAGE_EMBEDDINGS, SCANNED_PDF, IngestDocument
from chat_pdf.application.ingest_media import MediaIngest
from chat_pdf.domain.config import RagConfig
from chat_pdf.domain.errors import ProviderError
from chat_pdf.domain.figures import MediaRegion
from chat_pdf.domain.models import Page

PDF = "application/pdf"
TEXT = Page(1, "Un texto de cuerpo suficientemente largo para generar un chunk de prueba.")


def _region(page=1, sha=None, origin="embedded_image"):
    return MediaRegion(origin, page, [0.1, 0.1, 0.5, 0.5], sha)


def _setup(
    *regions,
    pages=(TEXT,),
    modalities=("text", "image"),
    mime=PDF,
    kind="pdf",
    vision=False,
    multimodal=None,
    **extractor_kw,
):
    profile = replace(embedding_profile(), modalities=list(modalities))
    db, storage = FakeDb(), FakeFileStorage()
    doc = db.add_document(db.add_chat().id, profile.id, status="pending")
    db.docs[doc.id] = replace(doc, mime=mime, kind=kind)
    active = {"chat": profile, "vision": vision_profile()} if vision else {}
    providers = FakeProviderRepository(profile, active=active)
    factory = FakeProviderFactory(multimodal=multimodal or FakeMultimodalEmbedder(("image",)))
    extractor = FakeMediaExtractor(*regions, **extractor_kw)
    media = MediaIngest({mime: extractor}, factory, storage)
    ingest = IngestDocument(
        lambda: FakeUnitOfWork(db),
        providers,
        factory,
        {mime: FakeLoader(*pages)} if pages is not None and kind != "image" else {},
        storage,
        RagConfig(),
        media=media,
    )
    return db, storage, factory, ingest, doc.id, extractor


def _run(ingest, doc_id):
    asyncio.run(ingest.execute(doc_id))


def test_pdf_images_become_segments_after_the_text_chunks():
    db, storage, factory, ingest, doc_id, _ = _setup(_region(1), _region(1, origin="figure"))

    _run(ingest, doc_id)

    chunks = db.chunks[doc_id]
    text, *images = chunks
    assert db.docs[doc_id].status == "ready" and text.modality == "text"
    assert [(c.modality, c.origin, c.page_start) for c in images] == [
        ("image", "embedded_image", 1),
        ("image", "figure", 1),
    ]
    first = images[0]
    assert (first.bbox, first.content, first.text_status) == ([0.1, 0.1, 0.5, 0.5], "", "pending")
    assert first.asset_key == f"{first.ordinal}.webp"
    assert (doc_id, "assets", first.asset_key) in storage.assets
    assert (doc_id, "thumbs", first.asset_key) in storage.assets
    assert [kind for kind, _ in factory.multimodal.received] == ["image", "image"]


def test_a_picture_repeated_on_every_page_keeps_one_segment_per_occurrence():
    db, _, factory, ingest, doc_id, _ = _setup(_region(1, "dog"), _region(2, "dog"))

    _run(ingest, doc_id)

    assert len(factory.multimodal.received) == 2  # each page has its own context
    assert [c.page_start for c in db.chunks[doc_id] if c.modality == "image"] == [1, 2]


def test_ocr_text_length_per_page_reaches_the_extractor():
    _, _, _, ingest, doc_id, extractor = _setup(_region(1), pages=(TEXT, Page(2, "ab")))

    _run(ingest, doc_id)

    assert extractor.plan_calls == [{1: len(TEXT.text), 2: 2}]


def test_without_a_vision_profile_descriptions_are_skipped_and_summary_too():
    db, _, _, ingest, doc_id, _ = _setup(_region(1), kind="image", mime="image/png", pages=None)

    _run(ingest, doc_id)

    doc = db.docs[doc_id]
    assert (doc.status, doc.enrich_status, doc.summary_status) == ("ready", "skipped", "skipped")


def test_with_a_vision_profile_descriptions_wait_pending():
    db, _, _, ingest, doc_id, _ = _setup(_region(1), vision=True)

    _run(ingest, doc_id)

    assert (db.docs[doc_id].status, db.docs[doc_id].enrich_status) == ("ready", "pending")


def test_plain_text_document_never_asks_for_descriptions():
    db, _, _, ingest, doc_id, _ = _setup(vision=True)

    _run(ingest, doc_id)

    assert db.docs[doc_id].enrich_status == "skipped"


def test_standalone_image_records_its_size():
    db, _, _, ingest, doc_id, _ = _setup(
        _region(None, origin="whole_image"),
        kind="image", mime="image/png", pages=None, width=800, height=600,
    )  # fmt: skip

    _run(ingest, doc_id)

    doc = db.docs[doc_id]
    assert (doc.status, doc.width, doc.height) == ("ready", 800, 600)
    assert [c.origin for c in db.chunks[doc_id]] == ["whole_image"]


def test_profile_without_image_support_indexes_a_pdf_as_text_only():
    db, storage, factory, ingest, doc_id, _ = _setup(_region(1), modalities=("text",))

    _run(ingest, doc_id)

    assert db.docs[doc_id].status == "ready"
    assert [c.modality for c in db.chunks[doc_id]] == ["text"]
    assert factory.multimodal.received == [] and storage.assets == {}


def test_profile_without_image_support_fails_an_image_upload_with_a_clear_error():
    db, _, _, ingest, doc_id, _ = _setup(
        _region(None, origin="whole_image"),
        modalities=("text",), kind="image", mime="image/png", pages=None,
    )  # fmt: skip

    _run(ingest, doc_id)

    assert db.docs[doc_id].status == "failed"
    assert db.docs[doc_id].error == NO_IMAGE_EMBEDDINGS


def test_scanned_pdf_without_ocr_text_is_ready_through_its_page_segments():
    db, _, _, ingest, doc_id, _ = _setup(_region(1, origin="page"), pages=(Page(1, ""),))

    _run(ingest, doc_id)

    assert db.docs[doc_id].status == "ready"
    assert [(c.modality, c.origin) for c in db.chunks[doc_id]] == [("image", "page")]


def test_pdf_with_neither_text_nor_images_is_still_the_scanned_error():
    db, _, _, ingest, doc_id, _ = _setup(pages=(Page(1, ""),))

    _run(ingest, doc_id)

    assert (db.docs[doc_id].status, db.docs[doc_id].error) == ("failed", SCANNED_PDF)


def test_unrenderable_regions_are_skipped_not_fatal():
    db, _, _, ingest, doc_id, _ = _setup(_region(1), _region(2), unrenderable=(0,))

    _run(ingest, doc_id)

    assert db.docs[doc_id].status == "ready"
    assert [c.page_start for c in db.chunks[doc_id] if c.modality == "image"] == [2]


def test_a_dead_embeddings_server_aborts_instead_of_skipping_every_image():
    class Down(FakeMultimodalEmbedder):
        async def embed_image(self, image, context=None):
            raise ProviderError("embeddings caído")

    db, _, _, ingest, doc_id, _ = _setup(
        _region(1, "a"), _region(1, "b"), _region(1, "c"), _region(1, "d"), multimodal=Down()
    )

    _run(ingest, doc_id)

    assert (db.docs[doc_id].status, db.chunks) == ("failed", {})
    assert "embeddings caído" in db.docs[doc_id].error


def test_cancel_between_images_fails_the_ingest_like_any_other_cancel():
    class Cancelling(FakeMultimodalEmbedder):
        async def embed_image(self, image, context=None):
            db.docs[doc_id] = replace(db.docs[doc_id], cancel_requested=True)
            return await super().embed_image(image, context)

    db, _, _, ingest, doc_id, _ = _setup(
        _region(1, "a"), _region(1, "b"), multimodal=Cancelling(("image",))
    )

    _run(ingest, doc_id)

    assert db.docs[doc_id].status == "failed" and db.docs[doc_id].error == "Ingesta cancelada"
    assert db.chunks == {}


@pytest.mark.parametrize("mime", ["image/jpeg", "image/png", "image/webp", "image/gif"])
def test_upload_accepts_image_types_and_records_the_kind(mime):
    from chat_pdf.domain.rules import kind_for_mime, resolve_upload_mime

    ext = {
        "image/jpeg": "a.JPG",
        "image/png": "a.png",
        "image/webp": "a.webp",
        "image/gif": "a.gif",
    }
    assert resolve_upload_mime(ext[mime], mime) == mime
    assert kind_for_mime(mime) == "image"
