"""Named business rules. Pure: they decide, the use cases act."""

from pathlib import PurePath

from chat_pdf.domain.errors import Conflict, InvalidInput
from chat_pdf.domain.models import Document
from chat_pdf.domain.sources import MAX_SOURCES

# extension -> (canonical MIME stored, MIME types accepted from the client)
UPLOAD_TYPES = {
    ".pdf": ("application/pdf", {"application/pdf"}),
    ".txt": ("text/plain", {"text/plain"}),
    ".md": ("text/markdown", {"text/markdown", "text/x-markdown", "text/plain"}),
    ".jpg": ("image/jpeg", {"image/jpeg", "image/jpg", "image/pjpeg"}),
    ".jpeg": ("image/jpeg", {"image/jpeg", "image/jpg", "image/pjpeg"}),
    ".png": ("image/png", {"image/png"}),
    ".webp": ("image/webp", {"image/webp"}),
    ".gif": ("image/gif", {"image/gif"}),
    ".mp3": ("audio/mpeg", {"audio/mpeg", "audio/mp3", "audio/x-mpeg"}),
    ".wav": ("audio/wav", {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}),
    ".m4a": ("audio/mp4", {"audio/mp4", "audio/x-m4a", "audio/m4a", "audio/aac"}),
    ".ogg": ("audio/ogg", {"audio/ogg", "application/ogg", "audio/opus"}),
    ".opus": ("audio/ogg", {"audio/ogg", "audio/opus"}),
    ".webm": ("audio/webm", {"audio/webm"}),
    ".flac": ("audio/flac", {"audio/flac", "audio/x-flac"}),
    ".mp4": ("video/mp4", {"video/mp4", "video/x-m4v", "video/mpeg4"}),
    ".m4v": ("video/mp4", {"video/mp4", "video/x-m4v"}),
    ".mov": ("video/quicktime", {"video/quicktime", "video/mov"}),
    ".mkv": (
        "video/x-matroska",
        {"video/x-matroska", "video/matroska", "video/webm", "application/octet-stream"},
    ),
}
# `.webm` holds audio recordings and video alike: the client's MIME type tells them apart.
WEBM_VIDEO = ("video/webm", {"video/webm"})


def resolve_upload_mime(filename: str, content_type: str | None) -> str:
    """Canonical MIME of an uploaded file; InvalidInput if the type is not supported."""
    suffix = PurePath(filename).suffix.lower()
    canonical, accepted = UPLOAD_TYPES.get(suffix, (None, set()))
    if suffix == ".webm" and content_type in WEBM_VIDEO[1]:
        canonical, accepted = WEBM_VIDEO
    if canonical is None or content_type not in accepted:
        raise InvalidInput(
            "unsupported_file_type",
            "Solo se admiten pdf, txt, md, imágenes (jpg, png, webp, gif), audio "
            "(mp3, wav, m4a, ogg, opus, webm, flac) y video (mp4, webm, mov, mkv)",
        )
    return canonical


def kind_for_mime(mime: str) -> str:
    """Source kind (documents.kind) of a canonical upload MIME."""
    if mime == "application/pdf":
        return "pdf"
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("video/"):
        return "video"
    return "audio" if mime.startswith("audio/") else "text"


class SourceLimit:
    """A chat holds up to MAX_SOURCES sources. A failed one does not count: the user may add
    another in its place (and remove the failed one when it suits them)."""

    @staticmethod
    def check(existing: list[Document]) -> None:
        if sum(1 for d in existing if d.status != "failed") >= MAX_SOURCES:
            raise Conflict(
                "too_many_sources",
                f"Un chat admite hasta {MAX_SOURCES} fuentes: quitá alguna para agregar otra",
            )


def validate_k_range(default_k: int, max_k: int) -> None:
    if default_k > max_k:
        raise InvalidInput("invalid_k", "default_k debe ser <= max_k")


def normalize_profile_data(kind: str, data: dict) -> dict:
    """Kind-specific fields: defaults, validation, and dropping the ones that do not apply.

    stt: protocol (default "ollama") + language (default "es"). tts: voice (required) + language.
    Ollama's native API lives at the server root, so a trailing `/v1` is dropped for it.
    """
    from chat_pdf.domain.providers import DEFAULT_LANGUAGE, DEFAULT_STT_PROTOCOL, STT_PROTOCOLS

    out = dict(data)
    for field in ("voice", "language", "protocol"):
        if isinstance(out.get(field), str):
            out[field] = out[field].strip() or None
    if kind == "stt":
        out["protocol"] = out.get("protocol") or DEFAULT_STT_PROTOCOL
        if out["protocol"] not in STT_PROTOCOLS:
            raise InvalidInput("invalid_protocol", "El protocolo debe ser 'openai' u 'ollama'")
        out["language"] = out.get("language") or DEFAULT_LANGUAGE
        out["voice"] = None
        if out["protocol"] == "ollama" and out.get("base_url"):
            out["base_url"] = out["base_url"].rstrip("/").removesuffix("/v1")
    elif kind == "tts":
        if not out.get("voice"):
            raise InvalidInput("voice_required", "Un perfil de voz (tts) necesita una voz, ej. F2")
        out["language"] = out.get("language") or DEFAULT_LANGUAGE
        out["protocol"] = None
    else:
        out.update(voice=None, language=None, protocol=None)
    return out
