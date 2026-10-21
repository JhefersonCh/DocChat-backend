import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from chat_pdf.application.ports import (
    AudioClip,
    ChatMessage,
    ChatResult,
    MultimodalEmbedder,
    ToolCall,
    Transcriber,
    VisionModel,
)
from chat_pdf.domain.errors import ProviderError, StreamingUnsupported, ToolsUnsupported
from chat_pdf.domain.providers import ProviderProfile

DEFAULT_TIMEOUT = httpx.Timeout(120.0, connect=5.0)  # the app overrides it from Settings
# The provider rejected the request shape (unknown param / tools / stream), vs. auth, rate limit
# or outage, which a different request would not fix.
REJECTED_STATUSES = (400, 404, 422)
ERROR_BODY_CHARS = 300  # provider error body kept in exception messages


def _headers(profile: ProviderProfile) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if profile.api_key:
        headers["Authorization"] = f"Bearer {profile.api_key}"
    return headers


def _url(profile: ProviderProfile, path: str) -> str:
    return f"{profile.base_url.rstrip('/')}{path}"


class _Base:
    def __init__(
        self,
        profile: ProviderProfile,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    ):
        self._profile = profile
        self._transport = transport
        self._timeout = timeout

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._timeout, transport=self._transport)

    async def _post(self, path: str, payload: dict[str, Any], **kwargs: Any) -> httpx.Response:
        """POST JSON (or, with `files=`/`data=` kwargs instead of a payload, multipart)."""
        # httpx's read timeout restarts on every byte: a provider that keeps the socket alive with
        # keep-alive lines (DeepSeek under load) would hang a request for minutes. Hard deadline.
        deadline = (self._timeout.read or 120.0) * 2
        try:
            async with asyncio.timeout(deadline), self._client() as client:
                headers = _headers(self._profile)
                if kwargs:  # multipart: httpx sets the Content-Type with its boundary
                    headers.pop("Content-Type")
                    return await client.post(_url(self._profile, path), headers=headers, **kwargs)
                return await client.post(_url(self._profile, path), json=payload, headers=headers)
        except TimeoutError as exc:
            raise ProviderError(f"el proveedor no respondió en {deadline:.0f}s") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc


class OpenAICompatibleEmbedder(_Base):
    """Sends `texts` to /embeddings in batches of `batch_size` (the only place that batches)."""

    def __init__(self, profile: ProviderProfile, batch_size: int, **kwargs: Any):
        super().__init__(profile, **kwargs)
        self._batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            res = await self._post("/embeddings", {"model": self._profile.model, "input": batch})
            if res.status_code >= 400:
                raise ProviderError(
                    f"embeddings HTTP {res.status_code}: {res.text[:ERROR_BODY_CHARS]}"
                )
            try:
                data = sorted(res.json()["data"], key=lambda d: d["index"])
                part = [d["embedding"] for d in data]
            except (ValueError, KeyError, TypeError) as exc:
                raise ProviderError("invalid embeddings response") from exc
            if len(part) != len(batch):
                raise ProviderError("embeddings count does not match inputs")
            vectors.extend(part)
        return vectors


def _parse_tool_calls(raw: list[dict[str, Any]] | None) -> list[ToolCall]:
    calls = []
    for item in raw or []:
        fn = item.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {}
        calls.append(
            ToolCall(
                id=item.get("id", ""),
                name=fn.get("name", ""),
                arguments=args if isinstance(args, dict) else {},
            )
        )
    return calls


class OpenAICompatibleChat(_Base):
    def _payload(self, messages: list[ChatMessage], stream: bool) -> dict[str, Any]:
        return {"model": self._profile.model, "messages": messages, "stream": stream}

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> ChatResult:
        payload = self._payload(messages, stream=False)
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        res = await self._post("/chat/completions", payload)
        if res.status_code >= 400:
            error = f"chat HTTP {res.status_code}: {res.text[:ERROR_BODY_CHARS]}"
            rejected = tools and res.status_code in REJECTED_STATUSES
            raise (ToolsUnsupported if rejected else ProviderError)(error)
        try:
            message = res.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("invalid chat response") from exc
        return ChatResult(
            content=message.get("content"), tool_calls=_parse_tool_calls(message.get("tool_calls"))
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        # Keep-alive comment lines reset httpx's read timeout: bound the wait for the first token.
        deadline = (self._timeout.read or 120.0) * 2
        started = asyncio.get_running_loop().time()
        got_text = False
        try:
            async with (
                self._client() as client,
                client.stream(
                    "POST",
                    _url(self._profile, "/chat/completions"),
                    json=self._payload(messages, stream=True),
                    headers=_headers(self._profile),
                ) as res,
            ):
                if res.status_code >= 400:
                    error = f"stream HTTP {res.status_code}"
                    raise (
                        StreamingUnsupported
                        if res.status_code in REJECTED_STATUSES
                        else ProviderError
                    )(error)
                if "text/event-stream" not in res.headers.get("content-type", ""):
                    raise StreamingUnsupported("server did not answer with an SSE stream")
                async for line in res.aiter_lines():
                    if not got_text and asyncio.get_running_loop().time() - started > deadline:
                        raise ProviderError(f"el proveedor no respondió en {deadline:.0f}s")
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        choices = json.loads(data).get("choices") or [{}]
                    except ValueError:
                        continue
                    text = (choices[0].get("delta") or {}).get("content")
                    if text:
                        got_text = True
                        yield text
        except httpx.HTTPError as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc


TRANSCRIBE_PROMPT = (
    "Transcribe this audio literally (spoken language code: {language}). "
    "Reply with only the transcription, no comments. If there is no speech, reply with nothing."
)


def _clean_transcript(text: str) -> str:
    return text.strip().strip('"\u201c\u201d').strip()


class OpenAITranscriber(_Base):
    """POST {base_url}/audio/transcriptions (multipart): OpenAI / Whisper / Ollama-compat API."""

    async def transcribe(self, wav: bytes) -> str:
        data = {"model": self._profile.model, "response_format": "json"}
        if self._profile.language:
            data["language"] = self._profile.language
        res = await self._post(
            "/audio/transcriptions", {}, data=data, files={"file": ("audio.wav", wav, "audio/wav")}
        )
        if res.status_code >= 400:
            raise ProviderError(f"stt HTTP {res.status_code}: {res.text[:ERROR_BODY_CHARS]}")
        try:
            return _clean_transcript(res.json()["text"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError("invalid transcription response") from exc


class OllamaTranscriber(_Base):
    """Ollama native /api/chat with the WAV in `images` and `think: false`: the fastest path
    for audio-capable models such as gemma4:e2b (the OpenAI-compat route re-encodes)."""

    async def transcribe(self, wav: bytes) -> str:
        prompt = TRANSCRIBE_PROMPT.format(language=self._profile.language or "es")
        payload = {
            "model": self._profile.model,
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "user", "content": prompt, "images": [base64.b64encode(wav).decode()]}
            ],
        }
        res = await self._post("/api/chat", payload)
        if res.status_code >= 400:
            raise ProviderError(f"stt HTTP {res.status_code}: {res.text[:ERROR_BODY_CHARS]}")
        try:
            return _clean_transcript(res.json()["message"]["content"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError("invalid transcription response") from exc


class OpenAICompatibleSpeaker(_Base):
    """POST {base_url}/audio/speech. `lang` is the Supertonic extension (ignored by others)."""

    async def speak(self, text: str) -> AudioClip:
        payload = {
            "model": self._profile.model,
            "input": text,
            "voice": self._profile.voice,
            "response_format": "wav",
        }
        if self._profile.language:
            payload["lang"] = self._profile.language
        res = await self._post("/audio/speech", payload)
        if res.status_code >= 400:
            raise ProviderError(f"tts HTTP {res.status_code}: {res.text[:ERROR_BODY_CHARS]}")
        if not res.content:
            raise ProviderError("empty audio response")
        return AudioClip(res.content, res.headers.get("content-type", "audio/wav"))


class HttpProviderFactory:
    def __init__(
        self,
        embed_batch: int,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        health_timeout_s: float = 3.0,
    ):
        self._embed_batch = embed_batch
        self._timeout = timeout
        self._health_timeout_s = health_timeout_s

    def embedder(self, profile: ProviderProfile) -> OpenAICompatibleEmbedder:
        return OpenAICompatibleEmbedder(profile, self._embed_batch, timeout=self._timeout)

    def multimodal_embedder(self, profile: ProviderProfile) -> MultimodalEmbedder:
        from chat_pdf.adapters.providers.multimodal import OpenAIMultimodalEmbedder

        return OpenAIMultimodalEmbedder(profile, timeout=self._timeout)

    def chat(self, profile: ProviderProfile) -> OpenAICompatibleChat:
        return OpenAICompatibleChat(profile, timeout=self._timeout)

    def transcriber(self, profile: ProviderProfile) -> Transcriber:
        cls = OllamaTranscriber if profile.protocol == "ollama" else OpenAITranscriber
        return cls(profile, timeout=self._timeout)

    def speaker(self, profile: ProviderProfile) -> OpenAICompatibleSpeaker:
        return OpenAICompatibleSpeaker(profile, timeout=self._timeout)

    def vision(self, profile: ProviderProfile) -> VisionModel:
        from chat_pdf.adapters.providers.vision import OpenAIVision

        return OpenAIVision(profile, timeout=self._timeout)

    async def reachable(self, profile: ProviderProfile) -> bool:
        """Cheap liveness check: any non-5xx answer from {base_url}/models."""
        try:
            async with httpx.AsyncClient(timeout=self._health_timeout_s) as client:
                res = await client.get(_url(profile, "/models"), headers=_headers(profile))
            return res.status_code < 500
        except httpx.HTTPError:
            return False
