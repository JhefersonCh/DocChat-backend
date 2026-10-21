"""Stage 2 of audio ingest: a transcript per window; resumable, cancellable, never un-readies."""

import asyncio
from dataclasses import replace
from uuid import uuid4

from fakes import (
    FakeDb,
    FakeEmbedder,
    FakeFileStorage,
    FakeProviderFactory,
    FakeProviderRepository,
    FakeTranscriber,
    FakeUnitOfWork,
    embedding_profile,
    stt_profile,
)
from fakes_media import FakeAudioProbe

from chat_pdf.application.enrich import NO_STT, EnrichDocument, RequestEnrichment
from chat_pdf.application.transcribe import CANCELLED, TranscribeWindows
from chat_pdf.domain.errors import Conflict, ProviderError
from chat_pdf.domain.models import Chunk


class ScriptedStt(FakeTranscriber):
    """Answers in order; an Exception is raised instead of answered."""

    def __init__(self, *answers):
        super().__init__()
        self.answers = list(answers)

    async def transcribe(self, wav):
        self.received.append(wav)
        answer = self.answers.pop(0) if self.answers else "texto de relleno"
        if isinstance(answer, Exception):
            raise answer
        return answer


class Env:
    def __init__(self, n=4, stt=True, answers=(), kind="audio"):
        self.profile = embedding_profile()
        self.db = FakeDb()
        doc = self.db.add_document(self.db.add_chat().id, self.profile.id)
        self.doc_id = doc.id
        self.db.docs[doc.id] = replace(doc, kind=kind, enrich_status="pending")
        self.db.chunks[doc.id] = [
            Chunk(
                i, "", None, None, None, None, id=uuid4(), modality="audio", origin="window",
                t_start=i * 16.0, t_end=i * 16.0 + 20, text_status="pending",
            )
            for i in range(n)
        ]  # fmt: skip
        self.providers = FakeProviderRepository(
            self.profile, active={"stt": stt_profile()} if stt else {}
        )
        self.stt = ScriptedStt(*answers)
        self.embedder = FakeEmbedder()
        self.probe = FakeAudioProbe(n * 16 + 4)
        self.factory = FakeProviderFactory(embedder=self.embedder, transcriber=self.stt)
        self.storage = FakeFileStorage()
        self.enrich = EnrichDocument(
            lambda: FakeUnitOfWork(self.db), self.providers, self.factory, self.storage, 0,
            audio=TranscribeWindows(
                lambda: FakeUnitOfWork(self.db), self.providers, self.factory, self.storage,
                self.probe, 0,
            ),
        )  # fmt: skip

    def run(self):
        asyncio.run(self.enrich.execute(self.doc_id))
        return self.db.docs[self.doc_id]

    def chunks(self):
        return self.db.chunks[self.doc_id]

    def statuses(self):
        return [c.text_status for c in self.chunks()]


def test_every_window_is_transcribed_with_its_own_slice_and_indexed_as_text():
    env = Env(n=3, answers=("uno dos", "tres cuatro", "cinco"))

    doc = env.run()

    assert (doc.status, doc.enrich_status, doc.enrich_error) == ("ready", "ready", None)
    assert [c.content for c in env.chunks()] == ["uno dos", "tres cuatro", "cinco"]
    assert [c.index_text for c in env.chunks()] == ["uno dos", "tres cuatro", "cinco"]
    assert env.statuses() == ["ready"] * 3
    assert env.probe.slices == [(0.0, 20.0), (16.0, 36.0), (32.0, 52.0)]
    assert env.stt.received[1] == b"wav:16.0-36.0"
    assert len(env.embedder.texts) == 3 and "tres cuatro" in env.embedder.texts[1]


def test_whitespace_in_a_transcript_is_normalized():
    env = Env(n=1, answers=("  hola\n  mundo  ",))

    env.run()

    assert env.chunks()[0].content == "hola mundo"


def test_progress_is_reported_per_window_in_the_transcribing_stage(monkeypatch):
    from fakes_store import FakeDocumentRepository

    env = Env(n=3)
    seen = []
    set_progress = FakeDocumentRepository.set_progress
    set_enrich = FakeDocumentRepository.set_enrich

    async def progress(self, document_id, done, total, stage=None):
        seen.append(("progress", done, total))
        return await set_progress(self, document_id, done, total, stage)

    async def enrich(self, document_id, status, error=None, stage="describing"):
        seen.append((status, stage if status == "running" else None))
        return await set_enrich(self, document_id, status, error, stage)

    monkeypatch.setattr(FakeDocumentRepository, "set_progress", progress)
    monkeypatch.setattr(FakeDocumentRepository, "set_enrich", enrich)

    env.run()

    assert seen[0] == ("running", "transcribing")
    assert [e for e in seen if e[0] == "progress"] == [("progress", i, 3) for i in range(3)]
    assert seen[-1] == ("ready", None)


def test_a_window_without_speech_is_skipped_not_failed_and_has_no_text_vector():
    env = Env(n=3, answers=("hola", "", "adiós"))

    doc = env.run()

    assert doc.enrich_status == "ready"
    assert env.statuses() == ["ready", "skipped", "ready"]
    assert env.chunks()[1].content == "" and len(env.embedder.texts) == 2


def test_without_an_stt_profile_the_stage_is_skipped_and_windows_stay_pending():
    env = Env(stt=False)

    doc = env.run()

    assert (doc.status, doc.enrich_status, doc.enrich_error) == ("ready", "skipped", NO_STT)
    assert env.statuses() == ["pending"] * 4


def test_a_flaky_window_is_retried_once_and_then_succeeds():
    env = Env(n=1, answers=(ProviderError("blip"), "hola"))

    doc = env.run()

    assert len(env.stt.received) == 2
    assert (doc.enrich_status, env.statuses()) == ("ready", ["ready"])


def test_a_window_that_keeps_failing_makes_the_stage_partial_not_the_source_failed():
    env = Env(n=3, answers=("uno", ProviderError("x"), ProviderError("y"), "tres"))

    doc = env.run()

    assert doc.status == "ready"  # the source stays searchable by its native vectors
    assert doc.enrich_status == "partial" and "1 de 3 transcripciones" in doc.enrich_error
    assert env.statuses() == ["ready", "failed", "ready"]


def test_a_failed_window_is_retried_by_the_next_run_and_only_it():
    env = Env(n=3, answers=("uno", ProviderError("x"), ProviderError("y"), "tres"))
    env.run()
    asyncio.run(RequestEnrichment(FakeUnitOfWork(env.db).docs, env.providers).execute(env.doc_id))
    env.stt.received.clear()
    env.stt.answers = ["dos"]

    doc = env.run()

    assert (doc.enrich_status, env.statuses()) == ("ready", ["ready"] * 3)
    assert len(env.stt.received) == 1 and env.stt.received[0] == b"wav:16.0-36.0"
    assert [c.content for c in env.chunks()] == ["uno", "dos", "tres"]


def test_resuming_after_an_interruption_continues_from_the_pending_windows():
    env = Env(n=4, answers=("a", "b"))
    store = FakeUnitOfWork(env.db).chunks
    first = env.chunks()[0]
    asyncio.run(store.save_caption(first.id, "ya hecho", "ya hecho", [1.0]))  # earlier run

    env.run()

    assert [c.content for c in env.chunks()][0] == "ya hecho"
    assert len(env.stt.received) == 3  # windows 1..3 only
    assert env.statuses() == ["ready"] * 4


def test_consecutive_failures_stop_early_and_leave_the_rest_for_a_retry():
    env = Env(n=6, answers=(ProviderError("401"),) * 6)

    doc = env.run()

    assert doc.enrich_status == "failed" and "401" in doc.enrich_error
    assert env.statuses() == ["failed"] * 3 + ["pending"] * 3


def test_a_slice_that_cannot_be_cut_fails_only_that_window():
    env = Env(n=3)
    env.probe.slice_errors = {16.0}

    doc = env.run()

    assert env.statuses() == ["ready", "failed", "ready"] and doc.enrich_status == "partial"


def test_cancel_stops_between_windows_and_the_rest_stays_pending():
    env = Env(n=3, answers=("uno",))
    original = env.stt.transcribe

    async def cancel_after_first(wav):
        text = await original(wav)
        env.db.docs[env.doc_id] = replace(env.db.docs[env.doc_id], cancel_requested=True)
        return text

    env.stt.transcribe = cancel_after_first

    doc = env.run()

    assert (doc.status, doc.enrich_status, doc.enrich_error) == ("ready", "partial", CANCELLED)
    assert env.statuses() == ["ready", "pending", "pending"]


def test_requesting_the_transcripts_of_an_audio_needs_an_stt_profile_not_a_vision_one():
    env = Env(stt=False)
    env.db.docs[env.doc_id] = replace(env.db.docs[env.doc_id], enrich_status="skipped")
    request = RequestEnrichment(FakeUnitOfWork(env.db).docs, env.providers)

    try:
        asyncio.run(request.execute(env.doc_id))
    except Conflict as err:
        assert err.code == "no_stt_profile"
    else:
        raise AssertionError("expected Conflict")


def test_an_unconfigured_audio_stage_fails_the_stage_not_the_source():
    env = Env()
    env.enrich._audio = None

    doc = env.run()

    assert (doc.status, doc.enrich_status) == ("ready", "failed")
