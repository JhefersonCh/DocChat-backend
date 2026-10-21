"""Stage 2 of a video: captions of the frames (vision) + transcripts of the audio (STT)."""

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
    FakeTranscriber,
    FakeUnitOfWork,
    FakeVision,
    embedding_profile,
    stt_profile,
    vision_profile,
)
from fakes_media import FakeAudioProbe
from fakes_store import FakeChunkStore, FakeDocumentRepository

from chat_pdf.application.enrich import NO_STT, NO_VISION, EnrichDocument, RequestEnrichment
from chat_pdf.application.media import GetStoryboard
from chat_pdf.application.transcribe import TranscribeWindows
from chat_pdf.domain.errors import Conflict, NotFound, ProviderError
from chat_pdf.domain.models import Chunk

GOOD = json.dumps({"tipo": "foto", "descripcion": "Un perro marrón", "objetos": ["perro"]})


class Disk(FakeFileStorage):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root

    def asset_path(self, document_id, kind, name) -> Path:
        return self.root / kind / name


class Env:
    def __init__(self, tmp_path, frames=3, windows=2, vision=True, stt=True):
        self.profile = embedding_profile()
        self.db = FakeDb()
        self.storage = Disk(tmp_path)
        (tmp_path / "assets").mkdir(parents=True)
        doc = self.db.add_document(self.db.add_chat().id, self.profile.id)
        self.doc_id = doc.id
        self.db.docs[doc.id] = replace(doc, kind="video", enrich_status="pending")
        chunks = []
        for i in range(frames):
            name = f"{i}.webp"
            (tmp_path / "assets" / name).write_bytes(b"RIFF....WEBPdata")
            chunks.append(
                Chunk(i, "", None, None, None, None, id=uuid4(), modality="frame", origin="frame",
                      t_start=i * 10.0, t_end=i * 10.0 + 10, asset_key=name,
                      text_status="pending")
            )  # fmt: skip
        for j in range(windows):
            chunks.append(
                Chunk(frames + j, "", None, None, None, None, id=uuid4(), modality="audio",
                      origin="window", t_start=j * 16.0, t_end=j * 16.0 + 20,
                      text_status="pending")
            )  # fmt: skip
        self.db.chunks[doc.id] = chunks
        active = {}
        if vision:
            active["vision"] = vision_profile()
        if stt:
            active["stt"] = stt_profile()
        self.providers = FakeProviderRepository(self.profile, active=active)
        self.vision, self.stt = FakeVision(), FakeTranscriber()
        self.factory = FakeProviderFactory(
            embedder=FakeEmbedder(), transcriber=self.stt, vision=self.vision
        )

        def uow():
            return FakeUnitOfWork(self.db)

        self.enrich = EnrichDocument(
            uow, self.providers, self.factory, self.storage, 0,
            audio=TranscribeWindows(
                uow, self.providers, self.factory, self.storage, FakeAudioProbe(40), 0
            ),
        )  # fmt: skip

    def run(self):
        asyncio.run(self.enrich.execute(self.doc_id))
        return self.db.docs[self.doc_id]

    def by_modality(self, modality):
        return [c for c in self.db.chunks[self.doc_id] if c.modality == modality]


def test_frames_are_captioned_and_the_audio_transcribed(tmp_path):
    env = Env(tmp_path / uuid4().hex)

    doc = env.run()

    assert (doc.status, doc.enrich_status, doc.enrich_error) == ("ready", "ready", None)
    assert [c.text_status for c in env.by_modality("frame")] == ["ready"] * 3
    assert [c.text_status for c in env.by_modality("audio")] == ["ready"] * 2
    assert len(env.vision.images) == 3 and len(env.stt.received) == 2  # each kind to its model


def test_without_an_stt_profile_the_frames_are_still_described_and_it_ends_partial(tmp_path):
    env = Env(tmp_path / uuid4().hex, stt=False)

    doc = env.run()

    assert doc.enrich_status == "partial" and doc.enrich_error.endswith(NO_STT)
    assert [c.text_status for c in env.by_modality("frame")] == ["ready"] * 3
    assert [c.text_status for c in env.by_modality("audio")] == ["pending"] * 2


def test_without_a_vision_profile_the_audio_is_still_transcribed_and_it_ends_partial(tmp_path):
    env = Env(tmp_path / uuid4().hex, vision=False)

    doc = env.run()

    assert doc.enrich_status == "partial" and doc.enrich_error.endswith(NO_VISION)
    assert [c.text_status for c in env.by_modality("frame")] == ["pending"] * 3
    assert [c.text_status for c in env.by_modality("audio")] == ["ready"] * 2


def test_with_neither_provider_it_is_skipped(tmp_path):
    env = Env(tmp_path / uuid4().hex, vision=False, stt=False)

    assert env.run().enrich_status == "skipped"


def test_a_video_without_audio_windows_needs_only_vision(tmp_path):
    env = Env(tmp_path / uuid4().hex, windows=0, stt=False)

    assert env.run().enrich_status == "ready"


def test_request_enrichment_accepts_a_video_with_only_one_of_the_providers(tmp_path):
    for vision, stt, ok in [(True, False, True), (False, True, True), (False, False, False)]:
        env = Env(tmp_path / uuid4().hex, vision=vision, stt=stt)
        env.db.docs[env.doc_id] = replace(
            env.db.docs[env.doc_id], status="ready", enrich_status="partial"
        )
        request = RequestEnrichment(FakeDocumentRepository(env.db), env.providers)
        if ok:
            asyncio.run(request.execute(env.doc_id))
        else:
            with pytest.raises(Conflict):
                asyncio.run(request.execute(env.doc_id))


def test_storyboard_is_only_for_videos(tmp_path):
    env = Env(tmp_path / uuid4().hex)
    case = GetStoryboard(FakeDocumentRepository(env.db), FakeChunkStore(env.db))
    # frames have a thumb (asset_key) so they all show up, in time order
    entries = asyncio.run(case.execute(env.doc_id))
    assert [e.t for e in entries] == [0.0, 10.0, 20.0]

    env.db.docs[env.doc_id] = replace(env.db.docs[env.doc_id], kind="audio")
    with pytest.raises(NotFound):
        asyncio.run(case.execute(env.doc_id))


def test_a_failed_frame_next_to_a_finished_transcript_is_partial_and_says_which_part(tmp_path):
    env = Env(tmp_path / uuid4().hex, frames=1, windows=2)
    env.db.chunks[env.doc_id] = [
        replace(c, text_status="ready") if c.modality == "audio" else c
        for c in env.db.chunks[env.doc_id]
    ]  # an earlier run transcribed the audio
    env.vision.answers = [ProviderError("sin saldo")] * 10

    doc = env.run()

    assert doc.enrich_status == "partial"  # not "failed": the audio is fine
    assert doc.enrich_error.startswith("Audio listo · fotogramas fallidos")
    assert [c.text_status for c in env.by_modality("frame")] == ["failed"]
