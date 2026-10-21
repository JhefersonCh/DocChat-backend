import asyncio

import httpx
import pytest

from chat_pdf.adapters.providers.openai_compatible import OpenAICompatibleChat
from chat_pdf.domain.errors import ProviderError, StreamingUnsupported, ToolsUnsupported
from chat_pdf.domain.providers import ProviderProfile


def test_stream_raises_when_server_answers_json_instead_of_sse():
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"choices": []}))
    profile = ProviderProfile(
        id=None,
        kind="chat",
        name="n",
        base_url="http://x/v1",
        model="m",  # type: ignore[arg-type]
    )
    chat = OpenAICompatibleChat(profile, transport=transport)

    async def consume():
        async for _ in chat.stream([{"role": "user", "content": "hi"}]):
            pass

    with pytest.raises(StreamingUnsupported):
        asyncio.run(consume())


def _chat(status: int) -> OpenAICompatibleChat:
    profile = ProviderProfile(None, "chat", "n", "http://x/v1", "m")  # type: ignore[arg-type]
    return OpenAICompatibleChat(
        profile, transport=httpx.MockTransport(lambda req: httpx.Response(status, text="no"))
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [(400, ToolsUnsupported), (422, ToolsUnsupported), (401, ProviderError), (429, ProviderError)],
)
def test_only_validation_statuses_mean_tools_unsupported(status, expected):
    with pytest.raises(ProviderError) as exc:
        asyncio.run(_chat(status).complete([], tools=[{}]))

    assert type(exc.value) is expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [(404, StreamingUnsupported), (403, ProviderError), (503, ProviderError)],
)
def test_only_validation_statuses_mean_streaming_unsupported(status, expected):
    async def consume():
        async for _ in _chat(status).stream([]):
            pass

    with pytest.raises(ProviderError) as exc:
        asyncio.run(consume())

    assert type(exc.value) is expected


def test_complete_has_hard_deadline_even_if_bytes_keep_flowing():
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json={})

    profile = ProviderProfile(None, "chat", "n", "http://x/v1", "m")  # type: ignore[arg-type]
    chat = OpenAICompatibleChat(
        profile, transport=httpx.MockTransport(slow), timeout=httpx.Timeout(0.05, connect=1)
    )
    with pytest.raises(ProviderError, match="no respondió"):
        asyncio.run(chat.complete([{"role": "user", "content": "hi"}]))
