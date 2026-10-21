"""Stage 1 of video ingest: keyframes at scene cuts, duplicates dropped, audio track, playable."""

import asyncio
from dataclasses import replace
from pathlib import Path

from fakes import (
    FakeDb,
    FakeFileStorage,
    FakeProviderFactory,
    FakeProviderRepository,
    FakeUnitOfWork,
    chat_profile,
    embedding_profile,
    stt_profile,
)
from fakes_media import FakeAudioProbe, FakeVideoProbe, SceneEmbedder, vision_profile

from chat_pdf.application.documents import NO_VIDEO_EMBEDDINGS, IngestDocument
from chat_pdf.application.ingest_audio import AudioIngest
from chat_pdf.application.ingest_video import VideoIngest
from chat_pdf.domain.config import RagConfig
from chat_pdf.domain.errors import ProviderError
from chat_pdf.domain.rules import kind_for_mime, resolve_upload_mime
from chat_pdf.domain.video import VideoInfo


def _setup(video, audio=None, modalities=("text", "image", "audio"), vision=True, stt=False,
           chat=True, name="original.mp4"):  # fmt: skip
    profile = replace(embedding_profile(), modalities=list(modalities))
    db, storage = FakeDb(), FakeFileStorage()
    doc = db.add_document(db.add_chat().id, profile.id, status="pending")
    db.docs[doc.id] = replace(doc, mime="video/mp4", kind="video")
    active = {}
    for key, on, make in (("vision", vision, vision_profile), ("stt", stt, stt_profile),
                          ("chat", chat, chat_profile)):  # fmt: skip
        if on:
            active[key] = make()
    providers = FakeProviderFactory(multimodal=SceneEmbedder(("image", "audio")))
    repo = FakeProviderRepository(profile, active=active)
    audio_ingest = AudioIngest(audio, providers, storage) if audio else None
    storage.path = lambda document_id, _n=name: Path(f"/fake/{_n}")
    ingest = IngestDocument(
        lambda: FakeUnitOfWork(db), repo, providers, {}, storage, RagConfig(),
        audio=audio_ingest, video=VideoIngest(video, audio_ingest, providers, storage),
    )  # fmt: skip
    return db, storage, providers, ingest, doc.id


def _run(ingest, doc_id):
    asyncio.run(ingest.execute(doc_id))


def test_one_frame_per_second_with_representatives_as_parents_and_multi_frame_clips():
    video = FakeVideoProbe(duration=40.0, scenes=(15.0,))
    db, _, factory, ingest, doc_id = _setup(video)

    _run(ingest, doc_id)

    doc, chunks = db.docs[doc_id], db.chunks[doc_id]
    assert (doc.status, doc.duration_s, doc.width, doc.height) == ("ready", 40.0, 1280, 720)
    frames = [c for c in chunks if c.origin == "frame"]
    clips = [c for c in chunks if c.origin == "clip"]
    assert all(
        c.modality == "frame" and (c.content, c.text_status) == ("", "pending") for c in chunks
    )
    starts = [c.t_start for c in frames]
    assert 15.0 in starts and 0.0 in starts
    # identical pictures are skipped, but never leaving a hole longer than 2 s (1 fps otherwise)
    edges = [*starts, 40.0]
    assert all(b - a <= 2.0 + 1e-6 for a, b in zip(edges, edges[1:], strict=False))
    assert len(frames) < len(video.frames)  # static stretches collapsed
    # representatives (scene cuts + one every ~5 s) are the parents; the rest point to one
    reps = [c for c in frames if c.parent_ordinal is None]
    assert [c.t_start for c in reps][:2] == [0.0, 5.0] and 15.0 in [c.t_start for c in reps]
    assert all(c.parent_ordinal is not None for c in frames if c not in reps)
    # clips: ~4 s of consecutive frames embedded as ONE input, under the representative at start
    assert len(clips) == 10 and clips[0].t_end == 4.0
    assert [kind for kind, _ in factory.multimodal.received].count("clip") == len(clips)
    assert all(c.parent_ordinal is not None and c.asset_key for c in clips)


def test_a_failing_frame_is_skipped_and_clips_fall_back_to_the_mean_of_the_frames():
    video = FakeVideoProbe(duration=30.0, scenes=(10.0, 20.0), unreadable=(10.0,))
    db, _, factory, ingest, doc_id = _setup(video)
    factory.multimodal.embed_clip = _no_clips

    _run(ingest, doc_id)

    chunks = db.chunks[doc_id]
    assert 10.0 not in [c.t_start for c in chunks] and 20.0 in [c.t_start for c in chunks]
    assert db.docs[doc_id].status == "ready"
    clips = [c for c in chunks if c.origin == "clip"]
    assert clips and "clip" not in [kind for kind, _ in factory.multimodal.received]


async def _no_clips(frames):
    raise ProviderError("multi-image input not supported")


def test_the_audio_track_follows_the_frames_in_the_same_document():
    video = FakeVideoProbe(duration=40.0, scenes=(15.0,))
    audio = FakeAudioProbe(40)
    db, storage, _, ingest, doc_id = _setup(video, audio=audio, stt=True)

    _run(ingest, doc_id)

    chunks = db.chunks[doc_id]
    visual = [c for c in chunks if c.modality == "frame"]
    sound = [c for c in chunks if c.modality == "audio"]
    assert chunks == [*visual, *sound]  # frames and clips first, then the audio track
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert [c.origin for c in sound].count("window") == 7  # ~9 s windows, 50 % overlap
    assert [c.origin for c in sound].count("context") == 2  # ~30 s parents
    assert (sound[0].t_start, sound[6].t_end) == (0.0, 40)
    assert (doc_id, "thumbs", "0.webp") in storage.assets  # the audio pass kept the frames
    assert (doc_id, "assets", "waveform.json") in storage.assets
    assert db.docs[doc_id].enrich_status == "pending"


def test_a_silent_track_is_not_an_error():
    video = FakeVideoProbe(duration=20.0, scenes=())
    audio = FakeAudioProbe(20, rms_db=[-90.0] * 400)
    db, _, _, ingest, doc_id = _setup(video, audio=audio)

    _run(ingest, doc_id)

    assert db.docs[doc_id].status == "ready"
    assert {c.modality for c in db.chunks[doc_id]} == {"frame"}


def test_a_video_without_audio_stream_skips_the_audio_pipeline():
    info = VideoInfo(20.0, 640, 360, "h264", None, "mov,mp4,m4a,3gp,3g2,mj2")
    video = FakeVideoProbe(duration=20.0, scenes=(), info=info)
    db, _, _, ingest, doc_id = _setup(video, audio=FakeAudioProbe(20))

    _run(ingest, doc_id)

    assert {c.modality for c in db.chunks[doc_id]} == {"frame"}


def test_a_playable_mp4_is_not_transcoded_but_an_unplayable_codec_is_once():
    playable = FakeVideoProbe()
    _, _, _, ingest, doc_id = _setup(playable)
    _run(ingest, doc_id)
    assert playable.transcodes == []

    hevc = FakeVideoProbe(
        info=VideoInfo(40.0, 1920, 1080, "hevc", "aac", "mov,mp4,m4a,3gp,3g2,mj2")
    )
    _, storage, _, ingest, doc_id = _setup(hevc)
    _run(ingest, doc_id)
    assert [dest for _, dest in hevc.transcodes] == [storage.playable_path(doc_id)]


def test_mkv_is_transcoded_even_with_playable_codecs():
    info = VideoInfo(40.0, 1280, 720, "h264", "aac", "matroska,webm")
    video = FakeVideoProbe(info=info)
    _, _, _, ingest, doc_id = _setup(video, name="original.mkv")

    _run(ingest, doc_id)

    assert len(video.transcodes) == 1


def test_stage_two_is_pending_only_if_something_can_be_described_or_transcribed():
    cases = [  # (vision, stt, windows) -> enrich_status
        (True, False, False, "pending"),
        (False, True, True, "pending"),
        (False, True, False, "skipped"),
        (False, False, True, "skipped"),
    ]
    for vision, stt, with_audio, expected in cases:
        audio = FakeAudioProbe(40) if with_audio else None
        db, _, _, ingest, doc_id = _setup(FakeVideoProbe(), audio=audio, vision=vision, stt=stt)
        _run(ingest, doc_id)
        assert db.docs[doc_id].enrich_status == expected, (vision, stt, with_audio)
        assert db.docs[doc_id].status == "ready"


def test_without_image_embeddings_the_ingest_fails_with_a_clear_message():
    db, _, _, ingest, doc_id = _setup(FakeVideoProbe(), modalities=("text", "audio"))

    _run(ingest, doc_id)

    assert db.docs[doc_id].status == "failed" and db.docs[doc_id].error == NO_VIDEO_EMBEDDINGS


def test_a_video_over_two_hours_is_refused():
    db, _, _, ingest, doc_id = _setup(FakeVideoProbe(duration=2 * 3600 + 1))

    _run(ingest, doc_id)

    assert db.docs[doc_id].status == "failed" and "2 h" in db.docs[doc_id].error


def test_upload_types_for_video():
    for name, mime, canonical in [
        ("a.mp4", "video/mp4", "video/mp4"),
        ("a.MOV", "video/quicktime", "video/quicktime"),
        ("a.mkv", "video/x-matroska", "video/x-matroska"),
        ("a.webm", "video/webm", "video/webm"),
        ("a.webm", "audio/webm", "audio/webm"),  # a recording stays audio
    ]:
        assert resolve_upload_mime(name, mime) == canonical
    assert kind_for_mime("video/webm") == "video" and kind_for_mime("audio/webm") == "audio"
