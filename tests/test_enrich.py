"""Stage 2: descriptions with the vision profile; resumable, cancellable, never un-readies."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from fakes import (
    FakeDb,
    FakeEmbedder,
    FakeFileStorage,
    FakeProviderFactory,
    FakeProviderRepository,
    FakeUnitOfWork,
    FakeVision,
    embedding_profile,
    vision_profile,
)

from chat_pdf.application.enrich import (
    CANCELLED,
    NO_VISION,
    EnrichDocument,
    RequestEnrichment,
)
from chat_pdf.domain.errors import Conflict, EmptyCompletion, ProviderError
from chat_pdf.domain.models import Chunk

GOOD = json.dumps({"tipo": "foto", "descripcion": "Un perro marrón", "objetos": ["perro"]})


class DiskStorage(FakeFileStorage):
    """Assets as real files so the use case can read the crop it describes."""

    def __init__(self, root: Path):
        super().__init__()
        self.root = root

    def asset_path(self, document_id, kind, name) -> Path:
        return self.root / kind / name


class Env:
    def __init__(self, tmp_path, n=3, vision: bool = True, **vision_kw):
        self.profile = embedding_profile()
        self.vision_profile = vision_profile()
        self.db = FakeDb()
        self.storage = DiskStorage(tmp_path)
        (tmp_path / "assets").mkdir()
        doc = self.db.add_document(self.db.add_chat().id, self.profile.id)
        self.doc_id = doc.id
        self.db.docs[doc.id] = replace(doc, enrich_status="pending")
        text = Chunk(0, "Texto de la página uno.", 1, 1, 0, 20, id=uuid4())
        media = [
            Chunk(
                i + 1, "", 1, 1, None, None, id=uuid4(), modality="image", origin="embedded_image",
                asset_key=f"{i + 1}.webp", text_status="pending",
            )
            for i in range(n)
        ]  # fmt: skip
        for m in media:
            (tmp_path / "assets" / m.asset_key).write_bytes(b"RIFF....WEBPdata")
        self.db.chunks[doc.id] = [text, *media]
        self.providers = FakeProviderRepository(
            self.profile, active={"vision": self.vision_profile} if vision else {}
        )
        self.embedder = FakeEmbedder()
        self.vision = FakeVision(*vision_kw.get("answers", ()))
        self.factory = FakeProviderFactory(embedder=self.embedder, vision=self.vision)
        self.enrich = EnrichDocument(
            lambda: FakeUnitOfWork(self.db), self.providers, self.factory, self.storage, 0
        )

    def run(self):
        asyncio.run(self.enrich.execute(self.doc_id))
        return self.db.docs[self.doc_id]

    def statuses(self):
        return [c.text_status for c in self.db.chunks[self.doc_id] if c.modality != "text"]


def test_describes_every_segment_and_stores_text_embedding_and_fts_text(tmp_path):
    env = Env(tmp_path, answers=(GOOD,) * 3)

    doc = env.run()

    media = [c for c in env.db.chunks[env.doc_id] if c.modality == "image"]
    assert (doc.status, doc.enrich_status, doc.enrich_error) == ("ready", "ready", None)
    assert env.statuses() == ["ready"] * 3
    assert media[0].content.startswith("[foto] Un perro marrón")
    assert "perro" in media[0].index_text
    assert len(env.embedder.texts) == 3  # one text embedding per caption
    assert "Un perro marrón" in env.embedder.texts[0]


def test_the_nearby_page_text_is_sent_as_context(tmp_path):
    env = Env(tmp_path, n=1, answers=(GOOD,))

    env.run()

    assert "Texto de la página uno." in env.vision.prompts[0]
    assert env.vision.images[0].startswith(b"RIFF")


def test_empty_content_is_retried_and_then_succeeds(tmp_path):
    env = Env(tmp_path, n=1, answers=(EmptyCompletion("vacía"), GOOD))

    doc = env.run()

    assert len(env.vision.prompts) == 2
    assert (doc.enrich_status, env.statuses()) == ("ready", ["ready"])


def test_a_segment_that_keeps_failing_makes_the_stage_partial_not_the_source_failed(tmp_path):
    env = Env(tmp_path, answers=(GOOD, ProviderError("x"), ProviderError("y"), GOOD))

    doc = env.run()

    assert doc.status == "ready"  # the source stays searchable
    assert doc.enrich_status == "partial" and "1 de 3" in doc.enrich_error
    assert env.statuses() == ["ready", "failed", "ready"]


def test_unparseable_empty_json_counts_as_a_failure(tmp_path):
    env = Env(tmp_path, n=1, answers=('{"tipo": "foto", "descripcion": ""}',) * 2)

    doc = env.run()

    assert (doc.enrich_status, env.statuses()) == ("failed", ["failed"])


def test_consecutive_failures_stop_early_and_leave_the_rest_for_a_retry(tmp_path):
    env = Env(tmp_path, n=5, answers=(ProviderError("401"),) * 6)

    doc = env.run()

    assert doc.enrich_status == "failed" and "401" in doc.enrich_error
    assert env.statuses() == ["failed"] * 3 + ["pending"] * 2
    assert len(env.vision.prompts) == 6  # 3 segments x 2 attempts, not all 5


def test_no_vision_profile_skips_and_keeps_segments_pending(tmp_path):
    env = Env(tmp_path, vision=False)

    doc = env.run()

    assert (doc.status, doc.enrich_status, doc.enrich_error) == ("ready", "skipped", NO_VISION)
    assert env.statuses() == ["pending"] * 3 and env.vision.prompts == []


def test_only_runs_for_a_pending_document(tmp_path):
    env = Env(tmp_path)
    env.db.docs[env.doc_id] = replace(env.db.docs[env.doc_id], enrich_status="ready")

    env.run()

    assert env.vision.prompts == [] and env.statuses() == ["pending"] * 3


def test_cancel_stops_between_segments_leaves_the_source_ready_and_resumes_the_rest(tmp_path):
    env = Env(tmp_path, n=3)
    original = env.vision.describe

    async def cancel_after_first(image, prompt):
        text = await original(image, prompt)
        env.db.docs[env.doc_id] = replace(env.db.docs[env.doc_id], cancel_requested=True)
        return text

    env.vision.describe = cancel_after_first

    doc = env.run()

    assert (doc.status, doc.enrich_status, doc.enrich_error) == ("ready", "partial", CANCELLED)
    assert doc.cancel_requested is False
    assert env.statuses() == ["ready", "pending", "pending"]

    env.vision.describe = original
    asyncio.run(_retry(env))
    doc = env.db.docs[env.doc_id]

    assert (doc.enrich_status, env.statuses()) == ("ready", ["ready"] * 3)
    assert len(env.vision.prompts) == 3  # the first segment was NOT described again


async def _retry(env):
    await RequestEnrichment(FakeUnitOfWork(env.db).docs, env.providers).execute(env.doc_id)
    await env.enrich.execute(env.doc_id)


def test_retry_requeues_failed_segments_but_not_the_described_ones(tmp_path):
    env = Env(tmp_path, answers=(GOOD, ProviderError("a"), ProviderError("b"), GOOD))
    env.run()
    assert env.statuses() == ["ready", "failed", "ready"]

    asyncio.run(_retry(env))

    assert env.db.docs[env.doc_id].enrich_status == "ready"
    assert env.statuses() == ["ready"] * 3
    assert len(env.vision.prompts) == 5  # 4 first run + only the failed one again


def test_progress_resumes_from_the_already_described_segments(tmp_path):
    env = Env(tmp_path, n=3)
    asyncio.run(_mark_ready(env, 0))
    seen = []
    original = FakeUnitOfWork(env.db).docs.set_progress

    async def spy(document_id, done, total, stage=None):
        seen.append((done, total))
        return await original(document_id, done, total, stage)

    env.enrich._uow = lambda: _UowWith(env.db, spy)

    env.run()

    assert seen == [(1, 3), (2, 3)]


async def _mark_ready(env, index):
    media = [c for c in env.db.chunks[env.doc_id] if c.modality != "text"]
    await FakeUnitOfWork(env.db).chunks.save_caption(media[index].id, "ya descripta", "ya", [1.0])


class _UowWith(FakeUnitOfWork):
    def __init__(self, db, set_progress):
        super().__init__(db)
        self.docs.set_progress = set_progress


def test_request_enrichment_rules(tmp_path):
    env = Env(tmp_path)
    docs = FakeUnitOfWork(env.db).docs
    request = RequestEnrichment(docs, env.providers)

    with pytest.raises(Conflict) as busy:  # `pending` already
        asyncio.run(request.execute(env.doc_id))
    assert busy.value.code == "document_busy"

    env.db.docs[env.doc_id] = replace(env.db.docs[env.doc_id], enrich_status="partial")
    asyncio.run(request.execute(env.doc_id))
    assert env.db.docs[env.doc_id].enrich_status == "pending"

    env.db.docs[env.doc_id] = replace(env.db.docs[env.doc_id], enrich_status="skipped")
    env.providers._active.pop("vision")
    with pytest.raises(Conflict) as no_vision:
        asyncio.run(request.execute(env.doc_id))
    assert no_vision.value.code == "no_vision_profile"


def test_interrupted_descriptions_become_partial_on_restart(tmp_path):
    env = Env(tmp_path)
    env.db.docs[env.doc_id] = replace(env.db.docs[env.doc_id], enrich_status="running")

    asyncio.run(FakeUnitOfWork(env.db).docs.fail_unfinished("reinicio"))

    doc = env.db.docs[env.doc_id]
    assert (doc.status, doc.enrich_status, doc.enrich_error) == ("ready", "partial", "reinicio")
