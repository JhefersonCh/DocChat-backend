"""Media endpoints (thumb / asset / original with HTTP Range) and the citation contract."""

import asyncio
from dataclasses import replace
from uuid import uuid4

import pytest
from fakes import (
    FakeChunkStore,
    FakeDb,
    FakeFileStorage,
    FakeMultimodalEmbedder,
    FakeProviderFactory,
    FakeProviderRepository,
    embedding_profile,
)
from fakes_store import FakeDocumentRepository
from fastapi import FastAPI
from fastapi.testclient import TestClient

from chat_pdf.adapters.http.chats import router as chats_router
from chat_pdf.adapters.http.deps import get_open_document_file, get_open_segment_asset
from chat_pdf.adapters.http.errors import register_error_handlers
from chat_pdf.adapters.http.media import router as media_router
from chat_pdf.application.documents import StoredFile
from chat_pdf.application.media import OpenSegmentAsset
from chat_pdf.application.providers import ProbeProvider
from chat_pdf.domain.errors import NotFound
from chat_pdf.domain.models import Chunk, Source

API = "/api/v1"


class _File:
    def __init__(self, stored: StoredFile):
        self.stored = stored

    async def execute(self, document_id, variant="playable"):
        return self.stored


def _client(file: StoredFile | None = None, assets: OpenSegmentAsset | None = None):
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(chats_router)
    app.include_router(media_router)
    if file:
        app.dependency_overrides[get_open_document_file] = lambda: _File(file)
    if assets:
        app.dependency_overrides[get_open_segment_asset] = lambda: assets
    return TestClient(app)


def test_original_file_supports_http_range_for_seeking(tmp_path):
    media = tmp_path / "original.mp3"
    media.write_bytes(bytes(range(256)) * 4)  # 1024 bytes
    client = _client(StoredFile(media, "clase.mp3", "audio/mpeg"))

    full = client.get(f"{API}/documents/{uuid4()}/file")
    part = client.get(f"{API}/documents/{uuid4()}/file", headers={"Range": "bytes=100-199"})
    tail = client.get(f"{API}/documents/{uuid4()}/file", headers={"Range": "bytes=1000-"})

    assert full.status_code == 200 and full.headers["accept-ranges"] == "bytes"
    assert part.status_code == 206
    assert part.headers["content-range"] == "bytes 100-199/1024"
    assert part.content == media.read_bytes()[100:200]
    assert tail.status_code == 206 and len(tail.content) == 24


def test_waveform_endpoint_serves_the_stored_peaks_and_404s_without_them(tmp_path):
    from chat_pdf.adapters.http.deps import get_get_waveform
    from chat_pdf.application.media import GetWaveform

    db, storage = FakeDb(), FakeFileStorage()
    doc = db.add_document(db.add_chat().id, uuid4())
    storage.asset_path = lambda d, kind, name: tmp_path / kind / name  # type: ignore[method-assign]
    client = _client()
    client.app.dependency_overrides[get_get_waveform] = lambda: GetWaveform(
        FakeDocumentRepository(db), storage
    )

    assert client.get(f"{API}/documents/{doc.id}/waveform").json()["code"] == "waveform_not_found"
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "waveform.json").write_text('{"peaks": [0.1, 1.0], "duration_s": 42.5}')

    res = client.get(f"{API}/documents/{doc.id}/waveform")

    assert res.json() == {"peaks": [0.1, 1.0], "duration_s": 42.5}
    assert client.get(f"{API}/documents/{uuid4()}/waveform").json()["code"] == "document_not_found"


def _asset_setup(tmp_path, asset_key: str | None = "s1.webp"):
    db, storage = FakeDb(), FakeFileStorage()
    doc_id, chunk_id = uuid4(), uuid4()
    chunk = Chunk(0, "", 1, 1, None, None, id=chunk_id, modality="image", asset_key=asset_key)
    db.chunks[doc_id] = [chunk]
    storage.asset_path = lambda d, kind, name: tmp_path / kind / name  # type: ignore[method-assign]
    return OpenSegmentAsset(FakeChunkStore(db), storage), chunk_id


@pytest.mark.parametrize(("route", "folder"), [("thumb", "thumbs"), ("asset", "assets")])
def test_segment_images_are_served_immutable(tmp_path, route, folder):
    (tmp_path / folder).mkdir()
    (tmp_path / folder / "s1.webp").write_bytes(b"RIFFwebp")
    use_case, segment_id = _asset_setup(tmp_path)

    res = _client(assets=use_case).get(f"{API}/segments/{segment_id}/{route}")

    assert res.status_code == 200
    assert res.content == b"RIFFwebp" and res.headers["content-type"] == "image/webp"
    assert "immutable" in res.headers["cache-control"]


def test_segment_without_image_or_unknown_segment_is_404(tmp_path):
    no_key, segment_id = _asset_setup(tmp_path, asset_key=None)
    missing_file, other_id = _asset_setup(tmp_path)  # key set but nothing on disk
    client = _client(assets=no_key)

    assert client.get(f"{API}/segments/{segment_id}/thumb").status_code == 404
    assert client.get(f"{API}/segments/{uuid4()}/thumb").json()["code"] == "segment_not_found"
    with pytest.raises(NotFound):
        asyncio.run(missing_file.execute(other_id, "assets"))


# --- citation contract ------------------------------------------------------------------------


def test_text_citation_keeps_the_legacy_shape_plus_defaults():
    chunk = Chunk(0, "hola mundo", 3, 3, 0, 10, id=uuid4(), similarity=0.7)

    citation = Source(1, chunk, uuid4(), "a.pdf", "pdf").cite(100)

    assert (citation.kind, citation.bbox, citation.t_start, citation.t_end) == (
        "text", None, None, None,
    )  # fmt: skip
    assert citation.has_thumb is False and citation.source_kind == "pdf"


@pytest.mark.parametrize(
    ("modality", "kind"), [("image", "image"), ("audio", "audio"), ("frame", "video")]
)
def test_media_citation_carries_locator_and_thumb(modality, kind):
    chunk = Chunk(
        0, "descripción", 12, 12, None, None, id=uuid4(), modality=modality,
        bbox=[0.1, 0.2, 0.5, 0.6], t_start=200.0, t_end=220.0, asset_key="s.webp",
    )  # fmt: skip

    citation = Source(2, chunk, uuid4(), "clase", None).cite(100)

    assert citation.kind == kind and citation.has_thumb is True
    assert citation.bbox == [0.1, 0.2, 0.5, 0.6]
    assert (citation.t_start, citation.t_end) == (200.0, 220.0)
    assert not Source(2, replace(chunk, asset_key=None), uuid4(), "c").cite(10).has_thumb


# --- probe detects modalities -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("accepts", "expected"),
    [
        ((), ["text"]),
        (("image",), ["text", "image"]),
        (("image", "audio"), ["text", "image", "audio"]),
    ],
)
def test_embedding_probe_detects_and_persists_modalities(accepts, expected):
    profile = embedding_profile()
    providers = FakeProviderRepository(profile)
    factory = FakeProviderFactory(multimodal=FakeMultimodalEmbedder(accepts))

    result = asyncio.run(ProbeProvider(providers, factory).execute(profile.id))

    assert result.ok and result.modalities == expected
    assert providers.profiles[profile.id].modalities == expected
