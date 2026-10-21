"""Small-to-big for video / audio (granularity pass): retrieval hits fine segments (a 9 s audio
window, a 1 fps frame); the answer reads the parent's text for context. Pure."""

from dataclasses import replace

from chat_pdf.domain.models import Chunk

REGIONS_MAX = 8
REGIONS_MAX_CHARS = 700
CONTEXT_MODALITIES = ("audio", "frame")  # image regions get their parent text elsewhere


def with_parent_text(found: list[Chunk], pool: list[Chunk]) -> list[Chunk]:
    """The same chunks, each fine audio / video child carrying its parent's text as
    `parent_text` (answer time only). A parent is attached to the FIRST hit under it only, so a
    long passage is not repeated for every window that matched."""
    by_id = {c.id: c for c in pool if c.id is not None}
    hit_ids = {c.id for c in found if c.id is not None}
    seen: set = set()
    out: list[Chunk] = []
    for chunk in found:
        regions = regions_text(chunk, pool, hit_ids)
        if regions:
            chunk = replace(chunk, regions_text=regions)
        parent = by_id.get(chunk.parent_id) if chunk.parent_id else None
        if (
            parent is None
            or chunk.modality not in CONTEXT_MODALITIES
            or chunk.parent_text is not None
            or not parent.content.strip()
            or parent.content == chunk.content
            or parent.id in seen
        ):
            out.append(chunk)
            continue
        seen.add(parent.id)
        out.append(replace(chunk, parent_text=parent.content))
    return out


def regions_text(chunk: Chunk, pool: list[Chunk], hit_ids: set) -> str | None:
    """A whole picture hit -> its region children as "label (description); ...", bounded.
    Regions that are hits themselves are skipped (they are shown on their own)."""
    if chunk.modality != "image" or chunk.parent_id is not None or chunk.id is None:
        return None
    parts: list[str] = []
    for c in pool:
        if c.parent_id != chunk.id or not c.region_label or c.id in hit_ids:
            continue
        desc = c.content.strip()
        parts.append(
            f"{c.region_label} ({desc})" if desc and desc != c.region_label else c.region_label
        )
        if len(parts) == REGIONS_MAX:
            break
    return "; ".join(parts)[:REGIONS_MAX_CHARS] or None
