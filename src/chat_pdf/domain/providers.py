from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

ProviderKind = Literal["chat", "embedding", "stt", "tts", "vision"]
SttProtocol = Literal["openai", "ollama"]
STT_PROTOCOLS = ("openai", "ollama")
DEFAULT_STT_PROTOCOL: SttProtocol = "ollama"  # native /api/chat: ~3x faster than /v1/audio/...
DEFAULT_LANGUAGE = "es"

# Chat-profile defaults, defined once. The SQL defaults in migrations/001_init.sql mirror them.
DEFAULT_K = 5
MAX_K = 15
MAX_TOOL_CALLS = 3


@dataclass(frozen=True)
class ProviderProfile:
    id: UUID
    kind: ProviderKind
    name: str
    base_url: str
    model: str
    api_key: str | None = None  # decrypted; never leaves the backend
    dimensions: int | None = None  # embedding only
    query_prefix: str | None = None  # embedding only
    document_prefix: str | None = None  # embedding only, supports {title}
    supports_tools: bool = False  # chat only
    default_k: int = DEFAULT_K  # chat only
    max_k: int = MAX_K  # chat only
    max_tool_calls: int = MAX_TOOL_CALLS  # chat only
    voice: str | None = None  # tts only (e.g. "F2")
    language: str | None = None  # stt / tts only (ISO code, e.g. "es")
    protocol: SttProtocol | None = None  # stt only: "openai" (/v1/audio/transcriptions) or "ollama"
    supports_vision: bool = False  # chat only: accepts image parts (set by the probe/UI)
    # embedding only: what the model can embed besides text ("image", "audio"); set by the probe
    modalities: list[str] = field(default_factory=lambda: ["text"])
    # embedding only: per-modality {"floor", "threshold"} scores (migration 013); None = defaults
    calibration: dict[str, dict[str, float]] | None = None
