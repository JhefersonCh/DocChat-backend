"""Image upload end to end over a real Postgres: stage 1, stage 2, thumbs, search."""

import io
import time

from fakes import FakeMultimodalEmbedder, FakeProviderFactory, FakeVision
from fastapi.testclient import TestClient
from PIL import Image

from chat_pdf.config import Settings
from chat_pdf.main import create_app

API = "/api/v1"


def _png() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (640, 480), (30, 120, 200)).save(out, "PNG")
    return out.getvalue()


def _wait(client, doc_id, field, done):
    for _ in range(80):
        doc = client.get(f"{API}/documents/{doc_id}").json()
        if doc[field] in done:
            return doc
        time.sleep(0.1)
    raise AssertionError(doc)


def _client(empty_db_url, tmp_path, factory):
    settings = Settings(
        database_url=empty_db_url, app_secret_key="any string", upload_dir=str(tmp_path)
    )
    return TestClient(create_app(settings, factory))


def _activate_vision(client):
    profile = client.post(
        f"{API}/providers",
        json={"kind": "vision", "name": "v", "base_url": "http://x/v1", "model": "m"},
    ).json()
    active = client.put(f"{API}/settings/active", json={"vision_profile_id": profile["id"]}).json()
    assert active["vision_profile_id"] == profile["id"]
    return profile


def _detect_image_support(client):
    embedding = client.get(f"{API}/settings/active").json()["embedding_profile_id"]
    probe = client.post(f"{API}/providers/{embedding}/test").json()
    assert probe["modalities"] == ["text", "image"]


def _upload(client, filename="foto.png", data=None, mime="image/png"):
    chat_id = client.post(f"{API}/chats").json()["id"]
    res = client.post(
        f"{API}/chats/{chat_id}/documents", files={"file": (filename, data or _png(), mime)}
    )
    assert res.status_code == 202, res.text
    return res.json()["document_id"]


def test_image_upload_is_searchable_then_described(empty_db_url, tmp_path):
    caption = '{"tipo": "foto", "descripcion": "Un perro marrón en el pasto", "objetos": ["perro"]}'
    factory = FakeProviderFactory(
        multimodal=FakeMultimodalEmbedder(("image",)), vision=FakeVision(caption)
    )
    with _client(empty_db_url, tmp_path, factory) as client:
        _detect_image_support(client)
        _activate_vision(client)
        doc_id = _upload(client)

        doc = _wait(client, doc_id, "enrich_status", {"ready", "failed", "partial"})
        chunks = client.get(f"{API}/documents/{doc_id}/chunks").json()
        segment = next(c for c in chunks["items"] if c["modality"] == "image")
        thumb = client.get(f"{API}/segments/{segment['chunk_id']}/thumb")
        asset = client.get(f"{API}/segments/{segment['chunk_id']}/asset")
        hits = client.get(f"{API}/documents/{doc_id}/search", params={"q": "perro"}).json()["hits"]
        again = client.post(f"{API}/documents/{doc_id}/enrich")

    assert (doc["kind"], doc["status"], doc["enrich_status"]) == ("image", "ready", "ready")
    assert (doc["width"], doc["height"], doc["stage"]) == (640, 480, None)
    assert segment["origin"] == "whole_image" and segment["bbox"] == [0, 0, 1, 1]
    assert "perro" in segment["content"]
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/webp"
    assert Image.open(io.BytesIO(asset.content)).format == "WEBP"
    assert {h["kind"] for h in hits} == {"image"} and hits[0]["has_thumb"] is True
    assert again.status_code == 409  # nothing left to describe


def test_without_a_vision_profile_the_image_is_ready_and_descriptions_skipped_then_resumable(
    empty_db_url, tmp_path
):
    factory = FakeProviderFactory(multimodal=FakeMultimodalEmbedder(("image",)))
    with _client(empty_db_url, tmp_path, factory) as client:
        _detect_image_support(client)
        doc_id = _upload(client)
        doc = _wait(client, doc_id, "status", {"ready", "failed"})
        no_vision = client.post(f"{API}/documents/{doc_id}/enrich")
        _activate_vision(client)
        resumed = client.post(f"{API}/documents/{doc_id}/enrich")
        doc_after = _wait(client, doc_id, "enrich_status", {"ready", "failed", "partial"})

    assert (doc["status"], doc["enrich_status"]) == ("ready", "skipped")
    assert (no_vision.status_code, no_vision.json()["code"]) == (409, "no_vision_profile")
    assert resumed.status_code == 202 and doc_after["enrich_status"] == "ready"


def test_image_upload_fails_clearly_until_the_embedding_profile_is_probed(empty_db_url, tmp_path):
    factory = FakeProviderFactory(multimodal=FakeMultimodalEmbedder(("image",)))
    with _client(empty_db_url, tmp_path, factory) as client:
        doc_id = _upload(client)  # modalities still the default ['text']
        doc = _wait(client, doc_id, "status", {"ready", "failed"})

    assert doc["status"] == "failed" and "no admite imágenes" in doc["error"]


def test_unsupported_image_type_is_rejected(empty_db_url, tmp_path):
    with _client(empty_db_url, tmp_path, FakeProviderFactory()) as client:
        chat_id = client.post(f"{API}/chats").json()["id"]
        res = client.post(
            f"{API}/chats/{chat_id}/documents", files={"file": ("a.bmp", b"BM", "image/bmp")}
        )

    assert res.status_code == 415
    assert res.json()["code"] == "unsupported_file_type"
