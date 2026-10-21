import re
from typing import Any

from chat_pdf.domain.find_all import MAX_OCCURRENCES, FindAllResult
from chat_pdf.domain.locator import WHOLE_PAGE_COVERAGE, format_locator, format_time, region_label
from chat_pdf.domain.models import Chat, Chunk, Source
from chat_pdf.domain.prompts import (
    ANSWER_USER_TEMPLATE,
    ATTACHED_IMAGE,
    FIND_ALL_CONFIDENCE,
    FIND_ALL_HEADER,
    FIND_ALL_MIXED,
    FIND_ALL_NONE,
    FIND_ALL_NONE_CHECKED,
    FIND_ALL_TOOL_DESCRIPTION,
    FIND_ALL_TOOL_NAME,
    FIND_ALL_TRUNCATED,
    FIND_ALL_UNVERIFIED,
    FIND_ALL_VERIFIED,
    LENGTH_INSTRUCTIONS,
    MEDIA_KIND_LABEL,
    MODALITY_PARAM_DESCRIPTION,
    PERSONA_INSTRUCTION,
    SEARCH_TOOL_DESCRIPTION,
    SEARCH_TOOL_NAME,
    SOURCE_LABEL,
    SOURCE_MEDIA,
    SOURCE_NO_DESCRIPTION,
    SOURCE_PAGE,
    SOURCE_PARAM_DESCRIPTION,
    SOURCE_PARENT,
    SOURCE_REGIONS,
    SOURCE_SECTION,
    SOURCES_NONE,
    SPOILER_ANSWER,
    STYLE_CUSTOM,
    STYLE_INSTRUCTIONS,
    SYSTEM_ANSWER,
    SYSTEM_MEDIA,
    SYSTEM_MEDIA_IMAGES,
    SYSTEM_MULTI_ANSWER,
    SYSTEM_WEAK_EVIDENCE,
)
from chat_pdf.domain.retrieval import MODALITY_HINTS
from chat_pdf.domain.sources import ALL

IMAGE_SOURCES_ATTACHED = 3  # actual pictures sent to a vision-capable chat model


def _with_source(properties: dict[str, Any], aliases: list[str] | None) -> None:
    if aliases and len(aliases) > 1:  # `source` only exists when there is something to choose
        properties["source"] = {
            "type": "string",
            "enum": [ALL, *aliases],
            "description": SOURCE_PARAM_DESCRIPTION,
        }


def search_tool(
    max_k: int, media: bool = False, aliases: list[str] | None = None
) -> dict[str, Any]:
    """`media`: the chat has images / audio, so the model may also pass a modality hint.
    `aliases`: the chat's readable sources (S1..Sn); with several, `source` filters the search."""
    properties: dict[str, Any] = {
        "query": {"type": "string"},
        "k": {"type": "integer", "minimum": 1, "maximum": max_k},
    }
    if media:
        properties["modality"] = {
            "type": "string",
            "enum": list(MODALITY_HINTS),
            "description": MODALITY_PARAM_DESCRIPTION,
        }
    _with_source(properties, aliases)
    return {
        "type": "function",
        "function": {
            "name": SEARCH_TOOL_NAME,
            "description": SEARCH_TOOL_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["query", "k"],
            },
        },
    }


def find_all_tool(aliases: list[str] | None = None) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "query": {"type": "string"},
        "modality": {"type": "string", "enum": ["imagen", "audio", "video"]},
    }
    _with_source(properties, aliases)
    return {
        "type": "function",
        "function": {
            "name": FIND_ALL_TOOL_NAME,
            "description": FIND_ALL_TOOL_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["query"],
            },
        },
    }


SOURCE_EXACT_FRAGMENT = "«{context}» (el fragmento que coincide con la búsqueda: «{fragment}»)"


def media_locator(chunk: Chunk) -> str:
    """Where a media segment sits, as a person would say it ("pág. 3, arriba", "01:20–01:40")."""
    if chunk.modality == "frame" and chunk.t_start is not None:
        return format_time(chunk.t_start)  # a frame is a moment, not a range
    where = format_locator(chunk.page_start, chunk.bbox, chunk.t_start, chunk.t_end)
    if not where and chunk.bbox is not None:  # a loose image: only its zone, never "the page"
        x0, y0, x1, y1 = chunk.bbox
        whole = x1 - x0 >= WHOLE_PAGE_COVERAGE and y1 - y0 >= WHOLE_PAGE_COVERAGE
        where = "" if whole else region_label(chunk.bbox)
    return where


def source_label(s: Source) -> str:
    """` (S2 · video.mp4)` when the chat has several sources, so the model knows whose it is."""
    return SOURCE_LABEL.format(alias=s.alias, name=s.filename) if s.labeled and s.alias else ""


def format_source(s: Source) -> str:
    chunk = s.chunk
    if chunk.modality == "text":
        page = SOURCE_PAGE.format(page=chunk.page_start) if chunk.page_start else ""
        section = SOURCE_SECTION.format(section=chunk.section) if chunk.section else ""
        return f"[{s.n}]{source_label(s)}{page}{section} {chunk.content}"
    where = media_locator(chunk)
    kind = MEDIA_KIND_LABEL[chunk.modality]
    if s.source_kind == "video":  # the speech of a video is cited as the video, at that moment
        kind = MEDIA_KIND_LABEL["frame"]
    label = SOURCE_MEDIA.format(kind=kind, locator=f" · {where}" if where else "")
    text = chunk.content.strip()
    if not text:
        text = SOURCE_NO_DESCRIPTION
    elif chunk.modality == "audio":
        if chunk.parent_text and chunk.parent_text != text:  # small-to-big: the wider passage
            text = SOURCE_EXACT_FRAGMENT.format(context=chunk.parent_text, fragment=text)
        else:
            text = f"«{text}»"  # a transcript is quoted speech
    if chunk.parent_text:  # small-to-big: the picture this region belongs to
        text += SOURCE_PARENT.format(text=chunk.parent_text)
    if chunk.regions_text:  # small-to-big: what its regions show (colors, objects)
        text += SOURCE_REGIONS.format(text=chunk.regions_text)
    return f"[{s.n}]{source_label(s)}{label} {text}"


def format_sources(sources: list[Source]) -> str:
    return "\n\n".join(format_source(s) for s in sources)


def format_find_all(query: str, result: FindAllResult) -> str:
    """Tool result of `find_all`: every occurrence with its locator, and how sure we are."""
    query = " ".join(query.split())
    if not result.sources:
        checked = FIND_ALL_NONE_CHECKED.format(count=result.discarded) if result.discarded else ""
        return FIND_ALL_NONE.format(query=query, checked=checked)
    unchecked = len(result.confidence)
    if unchecked == 0:
        verified = FIND_ALL_VERIFIED
    elif result.checked == 0:
        verified = FIND_ALL_UNVERIFIED
    else:
        verified = FIND_ALL_MIXED.format(checked=result.checked)
    lines = [FIND_ALL_HEADER.format(count=len(result.sources), query=query, verified=verified)]
    for s in result.sources:
        level = result.confidence.get(s.n)
        lines.append(format_source(s) + (FIND_ALL_CONFIDENCE.format(level=level) if level else ""))
    text = "\n\n".join(lines)
    if result.extra:
        text += FIND_ALL_TRUNCATED.format(extra=result.extra, limit=MAX_OCCURRENCES)
    return text


def style_instructions(chat: Chat) -> str:
    """The chat's style + length settings as system-prompt text ("" for the defaults)."""
    lines = [STYLE_INSTRUCTIONS.get(chat.style, ""), LENGTH_INSTRUCTIONS.get(chat.length, "")]
    custom = (chat.custom_instructions or "").strip()
    if chat.style == "custom" and custom:
        lines.append(STYLE_CUSTOM.format(text=custom))
    persona = (chat.persona or "").strip()
    if persona:
        lines.append(PERSONA_INSTRUCTION.format(persona=persona))
    return " ".join(line for line in lines if line)


def spoiler_instruction(max_page: int | None) -> str:
    """System-prompt text for "only what I have read" ("" when the mode is off)."""
    return "" if max_page is None else SPOILER_ANSWER.format(page=max_page)


def pictures_to_attach(sources: list[Source], limit: int = IMAGE_SOURCES_ATTACHED) -> list[Source]:
    """The best-ranked image / frame sources that have a picture on disk (sources are in rank
    order), at most `limit`."""
    pictured = [s for s in sources if s.chunk.modality in ("image", "frame") and s.chunk.asset_key]
    return pictured[:limit]


def build_answer_messages(
    history: list[dict[str, Any]],
    question: str,
    sources: list[Source],
    style: str = "",
    weak: bool = False,
    images: list[tuple[int, str]] | None = None,
) -> list[dict[str, Any]]:
    """`images`: (source number, data URL) pairs, attached as image parts of the USER message
    only (some providers reject images in system / assistant messages)."""
    user = ANSWER_USER_TEMPLATE.format(
        sources=format_sources(sources) or SOURCES_NONE, question=question
    )
    media = any(s.chunk.modality != "text" for s in sources)
    system = " ".join(
        part
        for part in (
            SYSTEM_ANSWER,
            media and SYSTEM_MEDIA + (SYSTEM_MEDIA_IMAGES if images else ""),
            any(s.labeled for s in sources) and SYSTEM_MULTI_ANSWER,
            style,
            weak and SYSTEM_WEAK_EVIDENCE,
        )
        if part
    )
    content: str | list[dict[str, Any]] = user
    if images:
        content = [{"type": "text", "text": user}]
        for n, url in images:
            content.append({"type": "text", "text": ATTACHED_IMAGE.format(n=n)})
            content.append({"type": "image_url", "image_url": {"url": url}})
    return [
        {"role": "system", "content": system},
        *history,
        {"role": "user", "content": content},
    ]


_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def strip_citations(text: str) -> str:
    """Remove `[n]` markers (and the space before them): numbers only mean something per answer."""
    return re.sub(rf"\s*{_CITATION.pattern}", "", text)


def parse_cited(text: str, sources: list[Source]) -> list[int]:
    """Numbers `[n]` used in the answer that map to a real source (out-of-range ignored)."""
    valid = {s.n for s in sources}
    used = {int(n) for group in _CITATION.findall(text) for n in re.split(r"\s*,\s*", group)}
    return sorted(used & valid)
