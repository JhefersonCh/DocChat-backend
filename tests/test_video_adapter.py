"""ffprobe / ffmpeg adapter with the subprocess mocked, plus the HTTP side of video."""

import asyncio
import io
import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from fakes import FakeDb, FakeFileStorage
from fakes_store import FakeChunkStore, FakeDocumentRepository
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from chat_pdf.adapters.http.chats import router as chats_router
from chat_pdf.adapters.http.deps import get_get_storyboard, get_open_document_file
from chat_pdf.adapters.http.errors import register_error_handlers
from chat_pdf.adapters.http.media import router as media_router
from chat_pdf.adapters.video.ffmpeg import FfmpegVideoProbe, parse_info, parse_scene_times
from chat_pdf.application.documents import OpenDocumentFile
from chat_pdf.application.media import GetStoryboard
from chat_pdf.domain.errors import InvalidInput
from chat_pdf.domain.models import Chunk

PATH = Path("/x/original.mkv")
PROBE = {
    "streams": [
        {"codec_type": "video", "codec_name": "hevc", "width": 1920, "height": 1080,
         "pix_fmt": "yuv420p10le", "disposition": {"attached_pic": 0}},
        {"codec_type": "audio", "codec_name": "aac"},
    ],
    "format": {"format_name": "matroska,webm", "duration": "61.5"},
}  # fmt: skip


def scripted(code=0, out=b""):
    calls: list[list[str]] = []

    async def run(args, timeout):
        calls.append(args)
        return code, out

    return run, calls


def test_info_reads_codecs_size_and_duration():
    run, calls = scripted(0, json.dumps(PROBE).encode())

    info = asyncio.run(FfmpegVideoProbe(run).info(PATH))

    assert (info.video_codec, info.audio_codec, info.container) == ("hevc", "aac", "matroska,webm")
    assert (info.width, info.height, info.duration_s, info.pix_fmt) == (
        1920, 1080, 61.5, "yuv420p10le",
    )  # fmt: skip
    assert calls[0][0] == "ffprobe"


def test_a_cover_art_stream_is_not_the_video_and_no_audio_means_none():
    cover = {
        "streams": [
            {"codec_type": "video", "codec_name": "mjpeg", "width": 5, "height": 5,
             "disposition": {"attached_pic": 1}},
            {"codec_type": "video", "codec_name": "h264", "width": 640, "height": 360},
        ],
        "format": {"format_name": "mov,mp4", "duration": "3"},
    }  # fmt: skip

    info = parse_info(json.dumps(cover).encode())

    assert info.video_codec == "h264" and info.audio_codec is None and info.has_audio is False


@pytest.mark.parametrize(
    "raw",
    [b"", b"not json", b"{}", json.dumps({"streams": [{"codec_type": "audio"}]}).encode()],
)
def test_something_that_is_not_a_video_is_invalid(raw):
    with pytest.raises(InvalidInput) as err:
        parse_info(raw)
    assert err.value.code == "video_undecodable"


def test_ffprobe_failure_is_invalid():
    run, _ = scripted(1, b"")
    with pytest.raises(InvalidInput):
        asyncio.run(FfmpegVideoProbe(run).info(PATH))


def test_scene_times_come_from_the_select_filter_metadata():
    text = (
        "frame:0    pts:250    pts_time:10.01\nlavfi.scene_score=0.55\n"
        "frame:1    pts:600    pts_time:24\nlavfi.scene_score=0.9\n"
    )
    run, calls = scripted(0, text.encode())

    assert asyncio.run(FfmpegVideoProbe(run).scene_times(PATH, 0.3)) == [10.01, 24.0]
    assert "gt(scene,0.3)" in " ".join(calls[0])
    assert parse_scene_times("") == []


def _png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (800, 600), "red").save(out, "PNG")
    return out.getvalue()


def test_frame_is_rendered_in_the_three_sizes():
    run, calls = scripted(0, _png())

    rendered = asyncio.run(FfmpegVideoProbe(run).frame(PATH, 12.5))

    assert max(Image.open(io.BytesIO(rendered.embed)).size) <= 384
    assert rendered.asset and rendered.thumb and rendered.sha
    assert calls[0][calls[0].index("-ss") + 1] == "12.500"


@pytest.mark.parametrize("code,out", [(1, b""), (0, b""), (0, b"garbage")])
def test_a_missing_or_corrupt_frame_is_invalid(code, out):
    run, _ = scripted(code, out)
    with pytest.raises(InvalidInput) as err:
        asyncio.run(FfmpegVideoProbe(run).frame(PATH, 1.0))
    assert err.value.code == "frame_undecodable"


def test_transcode_writes_atomically_and_failure_leaves_nothing(tmp_path):
    dest = tmp_path / "play.mp4"

    async def ok(args, timeout):
        Path(args[-1]).write_bytes(b"mp4")
        return 0, b""

    asyncio.run(FfmpegVideoProbe(ok).transcode_playable(PATH, dest))
    assert dest.read_bytes() == b"mp4" and not list(tmp_path.glob("*.part"))

    dest.unlink()

    async def broken(args, timeout):
        Path(args[-1]).write_bytes(b"half")
        return 1, b""

    with pytest.raises(InvalidInput) as err:
        asyncio.run(FfmpegVideoProbe(broken).transcode_playable(PATH, dest))
    assert err.value.code == "transcode_failed" and not dest.exists()
    assert not list(tmp_path.glob("*.part"))


# --- serving: the playable variant and the storyboard -----------------------------------------


class Storage(FakeFileStorage):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root

    def path(self, document_id):
        return self.root / "original.mkv"

    def playable_path(self, document_id):
        return self.root / "play.mp4"


def _video_doc(tmp_path, transcoded: bool):
    db = FakeDb()
    doc = db.add_document(db.add_chat().id, uuid4())
    db.docs[doc.id] = replace(doc, kind="video", mime="video/x-matroska", filename="clase 1.mkv")
    (tmp_path / "original.mkv").write_bytes(b"mkv-bytes")
    if transcoded:
        (tmp_path / "play.mp4").write_bytes(b"mp4-bytes")
    return db, doc.id


def _client(db, tmp_path):
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(chats_router)
    app.include_router(media_router)
    docs = FakeDocumentRepository(db)
    app.dependency_overrides[get_open_document_file] = lambda: OpenDocumentFile(
        docs, Storage(tmp_path)
    )
    app.dependency_overrides[get_get_storyboard] = lambda: GetStoryboard(docs, FakeChunkStore(db))
    return TestClient(app)


def test_file_serves_the_transcoded_mp4_unless_the_original_is_asked_for(tmp_path):
    db, doc_id = _video_doc(tmp_path, transcoded=True)
    client = _client(db, tmp_path)

    playable = client.get(f"/api/v1/documents/{doc_id}/file")
    ranged = client.get(f"/api/v1/documents/{doc_id}/file", headers={"Range": "bytes=0-2"})
    original = client.get(f"/api/v1/documents/{doc_id}/file?variant=original")

    assert (playable.content, playable.headers["content-type"]) == (b"mp4-bytes", "video/mp4")
    assert ranged.status_code == 206 and ranged.content == b"mp4"
    assert (original.content, original.headers["content-type"]) == (
        b"mkv-bytes",
        "video/x-matroska",
    )


def test_file_serves_the_original_when_it_was_playable(tmp_path):
    db, doc_id = _video_doc(tmp_path, transcoded=False)

    res = _client(db, tmp_path).get(f"/api/v1/documents/{doc_id}/file")

    assert res.content == b"mkv-bytes"


def test_storyboard_endpoint_lists_thumbs_in_time_order(tmp_path):
    db, doc_id = _video_doc(tmp_path, transcoded=False)
    frames = [
        Chunk(i, "", None, None, None, None, id=uuid4(), modality="frame", t_start=t, t_end=t + 5,
              asset_key=f"{i}.webp")
        for i, t in enumerate([20.0, 0.0])
    ]  # fmt: skip
    db.chunks[doc_id] = frames
    client = _client(db, tmp_path)

    res = client.get(f"/api/v1/documents/{doc_id}/storyboard")

    assert res.status_code == 200
    assert res.json() == [
        {"segment_id": str(frames[1].id), "t": 0.0, "t_end": 5.0,
         "thumb_url": f"/api/v1/segments/{frames[1].id}/thumb"},
        {"segment_id": str(frames[0].id), "t": 20.0, "t_end": 25.0,
         "thumb_url": f"/api/v1/segments/{frames[0].id}/thumb"},
    ]  # fmt: skip
    assert client.get(f"/api/v1/documents/{uuid4()}/storyboard").status_code == 404


def test_signature_is_a_small_grey_thumbnail_that_sees_text_changes():
    import io

    from PIL import Image, ImageDraw

    from chat_pdf.adapters.video.ffmpeg import FfmpegVideoProbe
    from chat_pdf.application.ports import RenderedMedia
    from chat_pdf.domain.video import is_duplicate_frame, pixel_change

    def png(draw_bar: bool) -> RenderedMedia:
        image = Image.new("RGB", (384, 216), "white")
        if draw_bar:
            ImageDraw.Draw(image).rectangle((20, 100, 300, 130), fill="black")  # a line of text
        out = io.BytesIO()
        image.save(out, "PNG")
        return RenderedMedia(b"", b"", out.getvalue(), "sha")

    probe = FfmpegVideoProbe()
    blank, text, blank2 = (asyncio.run(probe.signature(png(b))) for b in (False, True, False))

    assert len(blank) == 96 * 54
    assert pixel_change(blank, blank2) == 0.0
    assert not is_duplicate_frame(0.995, pixel_change(blank, text), 5.0)


def test_seconds_come_from_one_pass_and_are_read_from_the_cache(tmp_path):
    run, calls = scripted(0, b"")
    probe = FfmpegVideoProbe(run)
    asyncio.run(probe.extract_seconds(PATH, tmp_path))
    assert len(calls) == 1 and "fps=1" in " ".join(calls[0])

    out = io.BytesIO()
    Image.new("RGB", (800, 600), "blue").save(out, "JPEG")
    (tmp_path / "000007.jpg").write_bytes(out.getvalue())
    rendered = asyncio.run(probe.frame(PATH, 7.0, tmp_path))

    assert rendered.sha and len(calls) == 1  # no extra ffmpeg spawn
    assert not (tmp_path / "000007.jpg").exists()
