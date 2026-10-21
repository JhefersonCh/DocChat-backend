"""Source catalog for the planner (plan §3.4): what media the document holds, so the model knows
when `find_all` or a modality hint make sense."""

from chat_pdf.domain.locator import format_time
from chat_pdf.domain.models import Chunk, Document
from chat_pdf.domain.prompts import CATALOG_HEADER

KIND_NAMES = {"pdf": "PDF", "text": "texto", "image": "imagen", "audio": "audio", "video": "video"}


def has_media(chunks: list[Chunk]) -> bool:
    return any(c.modality != "text" for c in chunks)


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def media_counts(chunks: list[Chunk]) -> list[str]:
    """Phrases like "3 imágenes (2 descritas)"; tiles of a picture are not counted twice."""
    pictures = [c for c in chunks if c.modality == "image" and c.origin != "tile"]
    frames = [c for c in chunks if c.modality == "frame"]
    windows = [c for c in chunks if c.modality == "audio"]
    parts = []
    for items, one, many in (
        (pictures, "imagen", "imágenes"),
        (frames, "frame de video", "frames de video"),
        (windows, "fragmento de audio", "fragmentos de audio"),
    ):
        if items:
            described = sum(1 for c in items if c.text_status == "ready")
            parts.append(
                f"{_count(len(items), one, many)} ({_count(described, 'descrita', 'descritas')})"
            )
    return parts


def _line(alias: str, doc: Document, chunks: list[Chunk]) -> str:
    kind = KIND_NAMES.get(doc.kind, doc.kind)
    parts = [
        f"{alias} · {doc.filename}",
        kind + (f" {doc.page_count} págs" if doc.page_count else ""),
    ]
    if doc.duration_s:
        parts.append(format_time(doc.duration_s))
    parts += media_counts(chunks)
    return " · ".join(parts)


def source_catalog(doc: Document, chunks: list[Chunk], alias: str = "S1") -> str:
    """`S1 · informe.pdf · PDF 42 págs · 18 imágenes (16 descritas)`; "" for a text-only source."""
    if not has_media(chunks):
        return ""
    return f"{CATALOG_HEADER}\n{_line(alias, doc, chunks)}"


def collection_catalog(entries: list[tuple[str, Document, list[Chunk]]]) -> str:
    """The planner's catalog of the readable sources. One source: as `source_catalog` (nothing
    for a text-only one). Several: every source, text-only ones too, so it can pick by alias."""
    if len(entries) == 1:
        alias, doc, chunks = entries[0]
        return source_catalog(doc, chunks, alias)
    if not entries:
        return ""
    lines = [_line(alias, doc, chunks) for alias, doc, chunks in entries]
    return "\n".join([CATALOG_HEADER, *lines])
