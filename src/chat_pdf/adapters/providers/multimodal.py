import base64
import math
from dataclasses import replace
from typing import Any

import pymupdf

from chat_pdf.adapters.providers.openai_compatible import ERROR_BODY_CHARS, _Base
from chat_pdf.domain.errors import InvalidInput, ProviderError

# 384 px: 2.7 s vs 11-27 s per image on CPU with the same or better score (spike, plan §2.1).
EMBED_IMAGE_MAX_SIDE = 384


def prepare_image(data: bytes, max_side: int = EMBED_IMAGE_MAX_SIDE) -> bytes:
    """Decode any PNG/JPEG/... and return a PNG whose longest side is <= `max_side` (never
    upscaled, alpha dropped, always RGB/gray so every server decodes it the same way)."""
    try:
        pix = pymupdf.Pixmap(data)
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
        if pix.colorspace is not None and pix.colorspace.n not in (1, 3):
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        scale = max_side / max(pix.width, pix.height)
        if scale < 1:
            pix = pymupdf.Pixmap(
                pix, max(1, round(pix.width * scale)), max(1, round(pix.height * scale))
            )
        return pix.tobytes("png")
    except (pymupdf.mupdf.FzErrorBase, RuntimeError, ValueError) as exc:
        raise InvalidInput("invalid_image", "No se pudo decodificar la imagen") from exc


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


def _image_part(image: bytes) -> dict[str, Any]:
    url = "data:image/png;base64," + base64.b64encode(prepare_image(image)).decode()
    return {"type": "image_url", "image_url": {"url": url}}


# Wire protocol that worked for a server, remembered per (base_url, model) for the process.
_PROTOCOLS: dict[tuple[str, str], str] = {}
_REJECTED = (400, 404, 405, 415, 422)  # "I do not speak this shape": try the other protocol


class _Rejected(ProviderError):
    """The server answered 4xx to the typed-parts shape (not an outage: worth trying Ollama)."""


def _root(base_url: str) -> str:
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def _ollama_item(parts: list[dict[str, Any]]) -> dict[str, str]:
    """Typed content parts -> one Ollama `/api/embed` input item ({text?, image? | audio?})."""
    item: dict[str, str] = {}
    for part in parts:
        kind = part.get("type")
        if kind == "text":
            item["text"] = part["text"]
        elif kind == "image_url":
            if "image" in item:
                raise ProviderError("Ollama embeds one image per input")
            item["image"] = part["image_url"]["url"].split(",", 1)[-1]
        elif kind == "input_audio":
            item["audio"] = part["input_audio"]["data"]
    return item


class OpenAIMultimodalEmbedder(_Base):
    """Implements MultimodalEmbedder for the two servers that embed media.

    llama.cpp: `/v1/embeddings` with typed content parts, one media item per request (PITFALL,
    plan §7: the legacy `/embedding` endpoint silently returns a TEXT-only vector; never use it).
    Ollama: `/api/embed` (base_url without /v1) with `{"text"?, "image" | "audio"}` items; it
    answers 400 "input.content must be a string" to typed parts. The protocol is detected on the
    first call (typed parts first; a 4xx falls back to Ollama) and remembered per profile.
    """

    async def embed_image(self, image: bytes, context: str | None = None) -> list[float]:
        """With `context`, text and picture travel in ONE input (contextual embedding); a server
        that rejects the mix gets the picture alone (never worse than before)."""
        part = _image_part(image)
        text = " ".join((context or "").split())
        if text:
            try:
                return await self._embed_parts([{"type": "text", "text": text}, part])
            except ProviderError:
                pass  # unsupported mixed input: fall back to the image alone
        return await self._embed_part(part)

    async def embed_audio(self, wav: bytes) -> list[float]:
        data = base64.b64encode(wav).decode()
        return await self._embed_part(
            {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}
        )

    async def embed_clip(self, frames: list[bytes]) -> list[float]:
        """Verified live on llama.cpp: several image_url parts in ONE input item give one vector
        (cost ~ one image embedding per frame; the vector differs from any single frame's)."""
        return await self._embed_parts([_image_part(f) for f in frames])

    async def _embed_part(self, part: dict[str, Any]) -> list[float]:
        return await self._embed_parts([part])

    async def _embed_parts(self, parts: list[dict[str, Any]]) -> list[float]:
        key = (self._profile.base_url, self._profile.model)
        if _PROTOCOLS.get(key) == "ollama":
            return await self._ollama(parts)
        try:
            vector = await self._typed(parts)
        except _Rejected as rejected:
            try:
                vector = await self._ollama(parts)
            except ProviderError:
                raise rejected from None
            _PROTOCOLS[key] = "ollama"
            return vector
        _PROTOCOLS.setdefault(key, "openai")
        return vector

    async def _typed(self, parts: list[dict[str, Any]]) -> list[float]:
        payload = {"model": self._profile.model, "input": [{"content": parts}]}
        res = await self._post("/embeddings", payload)
        if res.status_code >= 400:
            error = _Rejected if res.status_code in _REJECTED else ProviderError
            raise error(f"embeddings HTTP {res.status_code}: {res.text[:ERROR_BODY_CHARS]}")
        try:
            [item] = res.json()["data"]
            return _normalize([float(v) for v in item["embedding"]])
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError("invalid embeddings response") from exc

    async def _ollama(self, parts: list[dict[str, Any]]) -> list[float]:
        native = _Base(
            replace(self._profile, base_url=_root(self._profile.base_url)),
            self._transport,
            self._timeout,
        )
        payload = {"model": self._profile.model, "input": [_ollama_item(parts)]}
        res = await native._post("/api/embed", payload)
        if res.status_code >= 400:
            raise ProviderError(f"embed HTTP {res.status_code}: {res.text[:ERROR_BODY_CHARS]}")
        try:
            [vector] = res.json()["embeddings"]
            return _normalize([float(v) for v in vector])
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError("invalid embeddings response") from exc
