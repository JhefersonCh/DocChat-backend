import asyncio
import base64
import json
import math
import os
import shutil

import httpx
import pymupdf
import pytest

from chat_pdf.adapters.providers.multimodal import (
    EMBED_IMAGE_MAX_SIDE,
    OpenAIMultimodalEmbedder,
    prepare_image,
)
from chat_pdf.application.providers import probe_png, probe_wav
from chat_pdf.domain.errors import InvalidInput, ProviderError
from chat_pdf.domain.providers import ProviderProfile


def _png(width: int, height: int) -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    pix.clear_with(180)
    return pix.tobytes("png")


def _profile(base_url: str = "http://x:8080/v1") -> ProviderProfile:
    return ProviderProfile(None, "embedding", "n", base_url, "embeddinggemma-2")  # type: ignore[arg-type]


def _embedder(handler) -> OpenAIMultimodalEmbedder:
    return OpenAIMultimodalEmbedder(_profile(), transport=httpx.MockTransport(handler))


def _answer(vector):
    return lambda req: httpx.Response(200, json={"data": [{"index": 0, "embedding": vector}]})


def test_image_goes_to_v1_embeddings_as_a_typed_content_part():
    """The llama.cpp pitfall: `/embedding` + `image_data` silently answers a TEXT-only vector.
    The only request that really embeds an image is /v1/embeddings with an image_url part."""
    seen: list[httpx.Request] = []

    def handler(req):
        seen.append(req)
        return _answer([3.0, 4.0])(req)

    vector = asyncio.run(_embedder(handler).embed_image(_png(64, 64)))

    [req] = seen
    assert req.url.path == "/v1/embeddings"  # never "/embedding"
    body = json.loads(req.content)
    assert "image_data" not in body and "content" not in body
    [item] = body["input"]
    [part] = item["content"]
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")
    assert body["model"] == "embeddinggemma-2"
    assert vector == pytest.approx([0.6, 0.8])  # L2-normalized


def test_audio_goes_as_input_audio_wav():
    seen: list[httpx.Request] = []

    def handler(req):
        seen.append(req)
        return _answer([1.0, 0.0])(req)

    wav = probe_wav(0.5)
    asyncio.run(_embedder(handler).embed_audio(wav))

    [req] = seen
    assert req.url.path == "/v1/embeddings"
    [item] = json.loads(req.content)["input"]
    [part] = item["content"]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "wav"
    assert base64.b64decode(part["input_audio"]["data"]) == wav


def test_media_requests_carry_no_text_prefix():
    seen: list[httpx.Request] = []
    profile = ProviderProfile(
        None, "embedding", "n", "http://x/v1", "m",  # type: ignore[arg-type]
        query_prefix="task: search result | query: ", document_prefix="title: {title} | text: ",
    )  # fmt: skip
    embedder = OpenAIMultimodalEmbedder(
        profile, transport=httpx.MockTransport(lambda r: seen.append(r) or _answer([1.0])(r))
    )

    asyncio.run(embedder.embed_image(_png(8, 8)))

    assert "task:" not in seen[0].content.decode() and "title:" not in seen[0].content.decode()


def test_images_are_downscaled_to_384px_before_embedding():
    out = prepare_image(_png(960, 720))

    pix = pymupdf.Pixmap(out)
    assert max(pix.width, pix.height) == EMBED_IMAGE_MAX_SIDE
    assert pix.width / pix.height == pytest.approx(960 / 720, abs=0.02)


def test_small_images_are_not_upscaled():
    pix = pymupdf.Pixmap(prepare_image(_png(100, 50)))

    assert (pix.width, pix.height) == (100, 50)


def test_undecodable_image_is_invalid_input_not_a_provider_error():
    with pytest.raises(InvalidInput):
        prepare_image(b"not an image")


def test_provider_http_error_is_surfaced():
    with pytest.raises(ProviderError, match="500"):
        asyncio.run(_embedder(lambda r: httpx.Response(500, text="boom")).embed_image(_png(8, 8)))


def test_malformed_response_is_a_provider_error():
    with pytest.raises(ProviderError, match="invalid"):
        asyncio.run(_embedder(lambda r: httpx.Response(200, json={"x": 1})).embed_image(_png(8, 8)))


# --- integration: needs the llama.cpp embeddings server with the mmproj (no --no-mmproj) -------

LIVE_URL = os.environ.get("MM_EMBEDDINGS_URL", "http://localhost:8080/v1")
QUERY_PREFIX = "task: search result | query: "


def _live_ok() -> bool:
    try:
        return httpx.get(f"{LIVE_URL}/models", timeout=2).status_code < 500
    except httpx.HTTPError:
        return False


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )


@pytest.mark.skipif(not _live_ok(), reason="embeddings server not reachable")
def test_live_image_vector_is_not_the_text_only_vector():
    """Regression for the `/embedding` pitfall: a real image vector must differ from the vector of
    the empty text, which is what a silently-ignored image produces."""
    from chat_pdf.adapters.providers.openai_compatible import OpenAICompatibleEmbedder

    profile = _profile(LIVE_URL)
    image = asyncio.run(OpenAIMultimodalEmbedder(profile).embed_image(probe_png(64)))
    [empty] = asyncio.run(OpenAICompatibleEmbedder(profile, 1).embed([""]))

    assert len(image) == 768
    assert _cosine(image, empty) < 0.97


def _red_circle() -> bytes:
    page = pymupdf.open().new_page(width=512, height=512)
    page.draw_circle((256, 256), 150, color=(1, 0, 0), fill=(1, 0, 0))
    return page.get_pixmap().tobytes("png")


@pytest.mark.skipif(not _live_ok(), reason="embeddings server not reachable")
def test_live_image_lands_near_its_description_in_the_text_space():
    from chat_pdf.adapters.providers.openai_compatible import OpenAICompatibleEmbedder

    profile = _profile(LIVE_URL)

    async def run():
        image = await OpenAIMultimodalEmbedder(profile).embed_image(_red_circle())
        match, other = await OpenAICompatibleEmbedder(profile, 2).embed(
            [QUERY_PREFIX + "a red circle on a white background", QUERY_PREFIX + "a green square"]
        )
        return _cosine(image, match), _cosine(image, other)

    matching, unrelated = asyncio.run(run())

    assert matching > 0.6
    assert matching > unrelated


TTS_URL = os.environ.get("MM_TTS_URL", "http://localhost:7788/v1")
SENTENCE = "Hoy el clima estará soleado con temperaturas altas durante toda la tarde."


def _tts_ok() -> bool:
    try:
        return httpx.get(f"{TTS_URL}/models", timeout=2).status_code < 500
    except httpx.HTTPError:
        return False


@pytest.mark.skipif(
    not (_live_ok() and _tts_ok() and shutil.which("ffmpeg")),
    reason="needs embeddings, tts, ffmpeg",
)
def test_live_speech_lands_near_its_transcript_in_the_text_space():
    from chat_pdf.adapters.audio.ffmpeg import FfmpegConverter
    from chat_pdf.adapters.providers.openai_compatible import OpenAICompatibleEmbedder

    profile = _profile(LIVE_URL)

    async def run():
        speech = httpx.post(
            f"{TTS_URL}/audio/speech",
            json={"model": "supertonic-3", "input": SENTENCE, "voice": "F2", "lang": "es",
                  "response_format": "wav"},
            timeout=60,
        )  # fmt: skip
        wav = await FfmpegConverter(300).to_wav16k(speech.content)
        audio = await OpenAIMultimodalEmbedder(profile).embed_audio(wav)
        match, other = await OpenAICompatibleEmbedder(profile, 2).embed(
            [QUERY_PREFIX + SENTENCE, QUERY_PREFIX + "El contrato fija un plazo de entrega."]
        )
        return _cosine(audio, match), _cosine(audio, other)

    matching, unrelated = asyncio.run(run())

    assert matching > 0.6
    assert matching > unrelated


def test_ollama_native_shape_is_detected_and_mixes_context_text_with_the_image():
    seen: list[httpx.Request] = []

    def handler(req):
        seen.append(req)
        if req.url.path == "/v1/embeddings":
            return httpx.Response(400, text="input.content must be a string")
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0]]})

    embedder = OpenAIMultimodalEmbedder(
        _profile("http://ollama-host:11434/v1"), transport=httpx.MockTransport(handler)
    )
    vector = asyncio.run(embedder.embed_image(_png(64, 64), "title: A | text: hola"))

    assert vector == pytest.approx([0.6, 0.8])
    native = seen[-1]
    assert native.url.path == "/api/embed"
    [item] = json.loads(native.content)["input"]
    assert item["text"] == "title: A | text: hola" and base64.b64decode(item["image"])
