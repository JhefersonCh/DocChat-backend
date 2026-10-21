import asyncio
from dataclasses import replace
from uuid import uuid4

import pytest
from fakes import (
    FakeDb,
    FakeDocumentRepository,
    FakeEmbedder,
    FakeFileStorage,
    FakeLoader,
    FakeProviderFactory,
    FakeProviderRepository,
    FakeUnitOfWork,
    embedding_profile,
)

from chat_pdf.application.documents import (
    SCANNED_PDF,
    CancelIngest,
    FailInterruptedIngests,
    IngestDocument,
    OpenDocumentFile,
    ReindexDocument,
    UploadDocument,
)
from chat_pdf.domain.config import RagConfig
from chat_pdf.domain.errors import Conflict, NotFound, ProviderError
from chat_pdf.domain.models import Chunk, Page


def test_failed_ingest_marks_document_failed_and_leaves_no_chunks():
    profile = embedding_profile()
    db, storage = FakeDb(), FakeFileStorage()
    doc = db.add_document(db.add_chat().id, profile.id, status="pending")
    ingest = IngestDocument(
        lambda: FakeUnitOfWork(db),
        FakeProviderRepository(profile),
        FakeProviderFactory(embedder=FakeEmbedder(ProviderError("embeddings caído"))),
        {"text/plain": FakeLoader(Page(None, "some text"))},
        storage,
        RagConfig(),
    )

    asyncio.run(ingest.execute(doc.id))

    assert db.docs[doc.id].status == "failed"
    assert "embeddings caído" in (db.docs[doc.id].error or "")
    assert db.chunks == {}


def test_upload_replaces_failed_document_and_its_file():
    profile = embedding_profile()
    db, storage = FakeDb(), FakeFileStorage()
    chat = db.add_chat()
    db.chats[chat.id] = replace(chat, title="old.pdf")  # titled from the failed upload
    failed = db.add_document(chat.id, profile.id, status="failed")
    storage.files[failed.id] = b"old"
    upload = UploadDocument(
        lambda: FakeUnitOfWork(db),
        FakeProviderRepository(profile, active={"embedding": profile}),
        storage,
    )

    async def content():
        yield b"new"

    new_id = asyncio.run(upload.execute(chat.id, "new.txt", "text/plain", content()))

    assert list(db.docs) == [new_id]
    assert list(storage.files) == [new_id]
    assert db.chats[chat.id].title == "new.txt"


@pytest.mark.parametrize(
    ("filename", "mime", "ext", "kind"),
    [("A.PDF", "application/pdf", ".PDF", "pdf"), ("n.md", "text/markdown", ".md", "text")],
)
def test_upload_records_source_kind_and_keeps_the_extension(filename, mime, ext, kind):
    profile = embedding_profile()
    db, storage = FakeDb(), FakeFileStorage()
    upload = UploadDocument(
        lambda: FakeUnitOfWork(db),
        FakeProviderRepository(profile, active={"embedding": profile}),
        storage,
    )

    async def content():
        yield b"x"

    doc_id = asyncio.run(upload.execute(db.add_chat().id, filename, mime, content()))

    assert db.docs[doc_id].kind == kind
    assert storage.exts[doc_id] == ext


def test_open_file_fails_when_document_or_stored_file_is_missing():
    db = FakeDb()
    doc = db.add_document(db.add_chat().id, embedding_profile().id)
    open_file = OpenDocumentFile(FakeDocumentRepository(db), FakeFileStorage())

    for document_id, code in [(uuid4(), "document_not_found"), (doc.id, "file_not_found")]:
        with pytest.raises(NotFound) as exc:
            asyncio.run(open_file.execute(document_id))
        assert exc.value.code == code


def test_restart_fails_documents_left_pending_or_processing():
    db = FakeDb()
    chat_id, profile_id = db.add_chat().id, uuid4()
    docs = [
        db.add_document(chat_id, profile_id, status=s) for s in ("pending", "processing", "ready")
    ]

    asyncio.run(FailInterruptedIngests(FakeDocumentRepository(db)).execute())

    assert [db.docs[d.id].status for d in docs] == ["failed", "failed", "ready"]
    assert "reinicio" in (db.docs[docs[0].id].error or "")


def test_reindex_replaces_chunks_and_embeds_each_with_its_section_title(tmp_path):
    profile = replace(embedding_profile(), document_prefix="title: {title} | text: ")
    db, embedder = FakeDb(), FakeEmbedder()
    doc = db.add_document(db.add_chat().id, profile.id)
    db.chunks[doc.id] = [Chunk(0, "viejo", None, None, 0, 5)]
    stored = tmp_path / "doc.txt"
    stored.write_text("x")
    storage = FakeFileStorage()
    storage.path = lambda _id: stored  # type: ignore[method-assign]
    ingest = IngestDocument(
        lambda: FakeUnitOfWork(db),
        FakeProviderRepository(profile),
        FakeProviderFactory(embedder=embedder),
        {"text/plain": FakeLoader(Page(None, "Portada.\n## Precios\nC A J A  de regalo."))},
        storage,
        RagConfig(chunk_size=20, chunk_overlap=0),
    )
    reindex = ReindexDocument(FakeDocumentRepository(db), storage)

    asyncio.run(reindex.execute(doc.id))
    with pytest.raises(Conflict):  # a second reindex while it is processing
        asyncio.run(reindex.execute(doc.id))
    asyncio.run(ingest.execute(doc.id))

    assert db.docs[doc.id].status == "ready"
    contents = [c.content for c in db.chunks[doc.id]]
    assert contents == ["Portada.", "## Precios", "C A J A  de regalo."]  # old chunk gone
    assert embedder.texts == [
        "title: none | text: Portada.",
        "title: Precios | text: ## Precios",
        "title: Precios | text: CAJA de regalo.",
    ]


def _ingest_setup(db, embedder, pages, config):
    profile = embedding_profile()
    doc = db.add_document(db.add_chat().id, profile.id, status="processing")
    ingest = IngestDocument(
        lambda: FakeUnitOfWork(db),
        FakeProviderRepository(profile),
        FakeProviderFactory(embedder=embedder),
        {"text/plain": FakeLoader(*pages)},
        FakeFileStorage(),
        config,
    )
    return doc, ingest


def test_ingest_reports_progress_per_embedding_batch_and_clears_it_when_ready():
    db, seen = FakeDb(), []

    class Spy(FakeEmbedder):
        async def embed(self, texts):
            seen.append(db.docs[doc.id].progress)
            return await super().embed(texts)

    doc, ingest = _ingest_setup(
        db, Spy(), [Page(None, "uno dos.\n\ntres cuatro.\n\ncinco seis.")],
        RagConfig(chunk_size=12, chunk_overlap=0, embed_batch=2),
    )  # fmt: skip

    asyncio.run(ingest.execute(doc.id))

    assert [p for p in seen if p] == [{"done": 0, "total": 4}, {"done": 2, "total": 4}]
    assert db.docs[doc.id].status == "ready" and db.docs[doc.id].progress is None


def test_cancel_stops_ingest_between_batches_and_fails_it_without_chunks():
    db = FakeDb()
    embedder = FakeEmbedder()
    doc, ingest = _ingest_setup(
        db, embedder, [Page(None, "uno dos.\n\ntres cuatro.\n\ncinco seis.")],
        RagConfig(chunk_size=12, chunk_overlap=0, embed_batch=1),
    )  # fmt: skip
    real_embed = embedder.embed

    async def cancel_after_first(texts):
        out = await real_embed(texts)
        await CancelIngest(FakeDocumentRepository(db)).execute(doc.id)
        return out

    embedder.embed = cancel_after_first  # type: ignore[method-assign]

    asyncio.run(ingest.execute(doc.id))

    assert len(embedder.texts) == 1  # stopped before the second batch
    assert (db.docs[doc.id].status, db.docs[doc.id].error) == ("failed", "Ingesta cancelada")
    assert db.chunks == {}


def test_cancel_requires_a_processing_document():
    db = FakeDb()
    ready = db.add_document(db.add_chat().id, uuid4(), status="ready")
    cancel = CancelIngest(FakeDocumentRepository(db))

    with pytest.raises(Conflict) as busy:
        asyncio.run(cancel.execute(ready.id))
    with pytest.raises(NotFound):
        asyncio.run(cancel.execute(uuid4()))

    assert busy.value.code == "document_not_busy"


def test_pdf_without_text_layer_fails_as_scanned_when_no_ocr_is_available():
    db = FakeDb()
    profile = embedding_profile()
    doc = db.add_document(db.add_chat().id, profile.id, status="processing")
    db.docs[doc.id] = replace(doc, mime="application/pdf")
    ingest = IngestDocument(
        lambda: FakeUnitOfWork(db),
        FakeProviderRepository(profile),
        FakeProviderFactory(),
        {"application/pdf": FakeLoader(Page(1, " "), Page(2, ""))},
        FakeFileStorage(),
        RagConfig(),
    )

    asyncio.run(ingest.execute(doc.id))

    assert db.docs[doc.id].status == "failed"
    assert db.docs[doc.id].error == SCANNED_PDF
