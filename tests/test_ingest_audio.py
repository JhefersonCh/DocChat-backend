"""Stage 1 of audio ingest: windows, native embeddings per window, waveform, `ready`."""

import asyncio
import json
from dataclasses import replace

import pytest
from fakes import (
    FakeDb,
    FakeFileStorage,
    FakeMultimodalEmbedder,
    FakeProviderFactory,
    FakeProviderRepository,
    FakeUnitOfWork,
    chat_profile,
    embedding_profile,
    stt_profile,
)
from fakes_media import FakeAudioProbe
from fakes_store import FakeChunkStore, FakeDocumentRepository

from chat_pdf.application.documents import (
    NO_AUDIO_EMBEDDINGS,
    NO_AUDIO_INDEX,
    IngestDocument,
)
from chat_pdf.application.ingest_audio import WAVEFORM_ASSET, AudioIngest
from chat_pdf.domain.config import RagConfig
from chat_pdf.domain.errors import InvalidInput, ProviderError
from chat_pdf.domain.rules import kind_for_mime, resolve_upload_mime

FRAME = 0.05


class FlakyAudio(FakeMultimodalEmbedder):
    """Embeds audio except the calls (1-based) in `fail_on`."""

    def __init__(self, fail_on=()):
        super().__init__(("audio",))
        self.fail_on, self.calls = set(fail_on), 0

    async def embed_audio(self, wav):
        self.calls += 1
        if self.calls in self.fail_on:
            raise ProviderError("audio no soportado")
        return await super().embed_audio(wav)


def _setup(probe, modalities=("text", "audio"), stt=True, chat=True, multimodal=None):
    profile = replace(embedding_profile(), modalities=list(modalities))
    db, storage = FakeDb(), FakeFileStorage()
    doc = db.add_document(db.add_chat().id, profile.id, status="pending")
    db.docs[doc.id] = replace(doc, mime="audio/mpeg", kind="audio")
    active = {}
    if stt:
        active["stt"] = stt_profile()
    if chat:
        active["chat"] = chat_profile()
    providers = FakeProviderRepository(profile, active=active)
    factory = FakeProviderFactory(multimodal=multimodal or FakeMultimodalEmbedder(("audio",)))
    ingest = IngestDocument(
        lambda: FakeUnitOfWork(db), providers, factory, {}, storage, RagConfig(),
        audio=AudioIngest(probe, factory, storage),
    )  # fmt: skip
    return db, storage, factory, ingest, doc.id


def _run(ingest, doc_id):
    asyncio.run(ingest.execute(doc_id))


def test_a_90_second_file_becomes_audio_windows_with_time_ranges_and_native_vectors():
    probe = FakeAudioProbe(90)
    db, storage, factory, ingest, doc_id = _setup(probe)

    _run(ingest, doc_id)

    doc, chunks = db.docs[doc_id], db.chunks[doc_id]
    assert (doc.status, doc.duration_s) == ("ready", 90)
    fine = [c for c in chunks if c.origin == "window"]
    parents = [c for c in chunks if c.origin == "context"]
    assert len(fine) == 18 and len(parents) == 3 and all(c.modality == "audio" for c in chunks)
    assert (fine[0].t_start, fine[0].t_end) == (0.0, 9.0)  # ~9 s windows, 50 % overlap
    assert fine[1].t_start == 4.5 and fine[-1].t_end == 90
    assert [c.ordinal for c in chunks] == list(range(21))
    assert all((c.content, c.text_status, c.asset_key) == ("", "pending", None) for c in chunks)
    # small-to-big: every fine window points to the ~30 s parent that holds it (never embedded)
    assert {c.parent_ordinal for c in fine} == {18, 19, 20} and all(
        c.parent_ordinal is None for c in parents
    )
    assert [kind for kind, _ in factory.multimodal.received] == ["audio"] * 18
    assert factory.multimodal.received[0][1] == b"wav:0.0-9.0"  # each window is its own slice


def test_stage_two_is_pending_with_stt_and_the_overview_waits_for_the_transcripts():
    db, _, _, ingest, doc_id = _setup(FakeAudioProbe(30))

    _run(ingest, doc_id)

    doc = db.docs[doc_id]
    assert doc.enrich_status == "pending" and doc.summary_status == "pending"


def test_without_an_stt_profile_stage_two_is_skipped_but_the_audio_is_still_searchable():
    db, _, _, ingest, doc_id = _setup(FakeAudioProbe(30), stt=False)

    _run(ingest, doc_id)

    doc = db.docs[doc_id]
    assert (doc.status, doc.enrich_status, doc.summary_status) == ("ready", "skipped", "skipped")


def test_the_waveform_is_stored_as_an_asset_json():
    db, storage, _, ingest, doc_id = _setup(FakeAudioProbe(30))

    _run(ingest, doc_id)

    payload = json.loads(storage.assets[(doc_id, "assets", WAVEFORM_ASSET)])
    assert (
        payload["duration_s"] == 30 and len(payload["peaks"]) == 600
    )  # one bar per 50 ms frame when there are fewer than 800
    assert max(payload["peaks"]) == 1.0


def test_silent_stretches_are_dropped_and_windows_snap_to_the_pauses():
    # 0-10 s speech, a pause centered at 19 s, speech until 60 s, then 30 s of digital silence
    levels = (
        [-20.0] * round(10 / FRAME)
        + [-20.0] * round(8.6 / FRAME)
        + [-70.0] * round(0.8 / FRAME)  # pause 18.6-19.4
        + [-20.0] * round(41 / FRAME)
        + [-80.0] * round(30 / FRAME)
    )
    probe = FakeAudioProbe(90, levels)
    db, _, _, ingest, doc_id = _setup(probe)

    _run(ingest, doc_id)

    chunks = db.chunks[doc_id]
    assert any(c.t_end == pytest.approx(19.0, abs=0.1) for c in chunks)  # snapped from 18
    assert all(c.t_start < 60 for c in chunks)  # the all-silent tail has no segments
    assert db.docs[doc_id].duration_s == 90


def test_a_file_of_pure_silence_fails_with_a_clear_message():
    db, _, _, ingest, doc_id = _setup(FakeAudioProbe(30, [-90.0] * 600))

    _run(ingest, doc_id)

    doc = db.docs[doc_id]
    assert doc.status == "failed" and "no tiene sonido" in doc.error


def test_audio_longer_than_three_hours_is_refused_before_decoding_it():
    probe = FakeAudioProbe(3 * 3600 + 1, [-20.0])
    db, _, _, ingest, doc_id = _setup(probe)

    _run(ingest, doc_id)

    doc = db.docs[doc_id]
    assert doc.status == "failed" and "3 h" in doc.error


def test_a_window_whose_embedding_fails_stays_as_a_text_only_segment(monkeypatch):
    db, _, _, ingest, doc_id = _setup(FakeAudioProbe(90), multimodal=FlakyAudio(fail_on={2}))
    stored = {}
    original = FakeChunkStore.replace

    async def spy(self, document_id, chunks, vectors):
        stored["vectors"] = vectors
        return await original(self, document_id, chunks, vectors)

    monkeypatch.setattr(FakeChunkStore, "replace", spy)

    _run(ingest, doc_id)

    assert db.docs[doc_id].status == "ready" and len(db.chunks[doc_id]) == 21
    assert [v is None for v in stored["vectors"][:18]] == [False, True] + [False] * 16
    assert all(v is None for v in stored["vectors"][18:])  # context parents have no vector


def test_without_native_audio_support_and_without_stt_there_is_nothing_to_index():
    db, _, _, ingest, doc_id = _setup(FakeAudioProbe(30), modalities=("text",), stt=False)

    _run(ingest, doc_id)

    assert db.docs[doc_id].status == "failed" and db.docs[doc_id].error == NO_AUDIO_EMBEDDINGS


def test_without_native_audio_support_the_transcripts_alone_index_the_audio():
    db, _, factory, ingest, doc_id = _setup(FakeAudioProbe(30), modalities=("text",))

    _run(ingest, doc_id)

    assert db.docs[doc_id].status == "ready" and factory.multimodal.received == []


def test_when_every_embedding_fails_and_there_is_no_stt_the_ingest_fails():
    db, _, _, ingest, doc_id = _setup(
        FakeAudioProbe(90), stt=False, multimodal=FlakyAudio(fail_on=set(range(1, 99)))
    )

    _run(ingest, doc_id)

    assert db.docs[doc_id].status == "failed" and db.docs[doc_id].error == NO_AUDIO_INDEX


def test_embedding_stops_after_three_consecutive_failures():
    multimodal = FlakyAudio(fail_on=set(range(1, 99)))
    db, _, _, ingest, doc_id = _setup(FakeAudioProbe(120), multimodal=multimodal)

    _run(ingest, doc_id)

    assert multimodal.calls == 3 and db.docs[doc_id].status == "ready"


def test_progress_counts_windows(monkeypatch):
    db, _, _, ingest, doc_id = _setup(FakeAudioProbe(60))
    seen = []
    original = FakeDocumentRepository.set_progress

    async def spy(self, document_id, done, total, stage=None):
        seen.append((done, total, stage))
        return await original(self, document_id, done, total, stage)

    monkeypatch.setattr(FakeDocumentRepository, "set_progress", spy)

    _run(ingest, doc_id)

    assert seen[0] == (0, 0, "extracting")
    assert seen[-1] == (12, 12, "embedding") and (6, 12, "embedding") in seen
    assert db.docs[doc_id].status == "ready"


def test_probe_errors_fail_the_ingest_with_their_message():
    class Broken(FakeAudioProbe):
        async def duration(self, path):
            raise InvalidInput("audio_undecodable", "El archivo no es un audio válido")

    db, _, _, ingest, doc_id = _setup(Broken(10))

    _run(ingest, doc_id)

    assert db.docs[doc_id].error == "El archivo no es un audio válido"


@pytest.mark.parametrize(
    "name,content_type,mime",
    [
        ("a.mp3", "audio/mpeg", "audio/mpeg"),
        ("a.wav", "audio/x-wav", "audio/wav"),
        ("a.m4a", "audio/x-m4a", "audio/mp4"),
        ("a.ogg", "audio/ogg", "audio/ogg"),
        ("a.opus", "audio/opus", "audio/ogg"),
        ("a.webm", "audio/webm", "audio/webm"),
        ("a.flac", "audio/flac", "audio/flac"),
    ],
)
def test_audio_formats_are_accepted_and_classified_as_audio(name, content_type, mime):
    assert resolve_upload_mime(name, content_type) == mime
    assert kind_for_mime(mime) == "audio"


def test_an_audio_extension_with_a_foreign_content_type_is_refused():
    with pytest.raises(InvalidInput):
        resolve_upload_mime("a.mp3", "application/pdf")
