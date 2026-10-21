import asyncio
import json

import httpx
import pytest

from chat_pdf.adapters.providers.vision import OpenAIVision, image_mime
from chat_pdf.domain.errors import EmptyCompletion, ProviderError
from chat_pdf.domain.providers import ProviderProfile

PROFILE = ProviderProfile(None, "vision", "v", "http://x/v1", "deepseek-flash", api_key="k")  # type: ignore[arg-type]
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 20


def _vision(handler) -> OpenAIVision:
    return OpenAIVision(PROFILE, transport=httpx.MockTransport(handler))


def _answer(message: dict, finish: str = "stop") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish}]})


def test_sends_the_image_as_a_data_url_with_an_ample_token_budget():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return _answer({"content": " Un perro. "})

    text = asyncio.run(_vision(handler).describe(PNG, "descríbela"))

    body = seen["body"]
    [prompt, image] = body["messages"][0]["content"]
    assert text == "Un perro."
    assert prompt == {"type": "text", "text": "descríbela"}
    assert image["image_url"]["url"].startswith("data:image/png;base64,")
    assert body["max_tokens"] >= 800 and body["model"] == "deepseek-flash"
    assert seen["auth"] == "Bearer k"


def test_reads_content_and_ignores_the_models_reasoning():
    message = {"content": "respuesta", "reasoning_content": "pensando...", "reasoning": "otra cosa"}

    text = asyncio.run(_vision(lambda r: _answer(message)).describe(PNG, "p"))

    assert text == "respuesta"


def test_empty_content_with_reasoning_is_a_retryable_error_not_the_reasoning():
    message = {"content": "", "reasoning_content": "Let me think about the image..."}

    with pytest.raises(EmptyCompletion, match="max_tokens") as exc:
        asyncio.run(_vision(lambda r: _answer(message, "length")).describe(PNG, "p"))

    assert "think" not in str(exc.value)
    assert isinstance(exc.value, ProviderError)


def test_null_content_is_also_empty():
    with pytest.raises(EmptyCompletion):
        asyncio.run(_vision(lambda r: _answer({"content": None})).describe(PNG, "p"))


def test_http_errors_and_garbage_are_provider_errors():
    with pytest.raises(ProviderError, match="HTTP 401"):
        asyncio.run(_vision(lambda r: httpx.Response(401, text="no")).describe(PNG, "p"))
    with pytest.raises(ProviderError, match="invalid"):
        asyncio.run(_vision(lambda r: httpx.Response(200, json={})).describe(PNG, "p"))


def test_mime_is_sniffed_from_the_bytes():
    assert image_mime(PNG) == "image/png"
    assert image_mime(b"\xff\xd8\xff\xe0abc") == "image/jpeg"
    assert image_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
