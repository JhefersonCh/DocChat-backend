import base64
import logging
from typing import Any

from chat_pdf.adapters.providers.openai_compatible import ERROR_BODY_CHARS, _Base
from chat_pdf.domain.captions import CAPTION_MAX_TOKENS
from chat_pdf.domain.errors import EmptyCompletion, ProviderError

log = logging.getLogger(__name__)

MAGIC_MIMES = ((b"\x89PNG", "image/png"), (b"\xff\xd8", "image/jpeg"), (b"GIF8", "image/gif"))


def image_mime(data: bytes) -> str:
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return next((mime for magic, mime in MAGIC_MIMES if data.startswith(magic)), "image/png")


def _content_text(message: dict[str, Any]) -> str:
    """Only `content`: `reasoning` / `reasoning_content` are the model's thinking, never ours."""
    content = message.get("content")
    if isinstance(content, list):  # some servers answer with typed parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content.strip() if isinstance(content, str) else ""


class OpenAIVision(_Base):
    """Implements VisionModel on OpenAI-compatible `/chat/completions` with `image_url` data URLs.

    PITFALL: reasoning models (e.g. DeepSeek `deepseek-flash`) spend thinking tokens that count
    against `max_tokens`, returning empty `content` when the budget runs out.  `describe()` retries
    automatically with a capped `thinking.budget_tokens`; non-reasoning servers ignore that key.
    """

    def __init__(self, profile, max_tokens: int = CAPTION_MAX_TOKENS, **kwargs: Any):
        super().__init__(profile, **kwargs)
        self._max_tokens = max_tokens

    async def _call(self, payload: dict[str, Any]) -> tuple[str, str | None]:
        """POST to /chat/completions; return (content_text, finish_reason). Raises ProviderError."""
        res = await self._post("/chat/completions", payload)
        if res.status_code >= 400:
            raise ProviderError(f"vision HTTP {res.status_code}: {res.text[:ERROR_BODY_CHARS]}")
        try:
            choice = res.json()["choices"][0]
            text = _content_text(choice["message"])
            finish_reason = choice.get("finish_reason")
            usage = res.json().get("usage", {})
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("invalid vision response") from exc
        log.debug(
            "vision _call: finish_reason=%r text_len=%d usage=%s thinking_in_payload=%s",
            finish_reason,
            len(text),
            usage,
            "thinking" in payload,
        )
        return text, finish_reason

    async def describe(self, image: bytes, prompt: str) -> str:
        url = f"data:{image_mime(image)};base64,{base64.b64encode(image).decode()}"
        base_payload = {
            "model": self._profile.model,
            "stream": False,
            "max_tokens": self._max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
        }
        text, finish_reason = await self._call(base_payload)
        if text:
            return text

        if finish_reason == "length":
            log.warning(
                "vision: reasoning used all %d tokens, retrying with low effort + 2x budget",
                self._max_tokens,
            )
            capped = {
                **base_payload,
                "max_tokens": self._max_tokens * 1.5,
                "reasoning_effort": "low",
            }
            text, finish_reason = await self._call(capped)
            if text:
                return text

        why = " (agotó max_tokens pensando)" if finish_reason == "length" else ""
        raise EmptyCompletion(f"el modelo de visión devolvió una respuesta vacía{why}")
