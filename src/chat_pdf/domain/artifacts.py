"""Study artifacts: kinds, options, chunk selection (with pages) and the generation prompt."""

import re
from typing import Any

from chat_pdf.domain.artifact_prompts import (
    ARTIFACT_SYSTEM,
    CHART_REQUEST,
    CONTEXT,
    INSTRUCTIONS,
)
from chat_pdf.domain.concept_map import clean_map_options
from chat_pdf.domain.errors import InvalidInput
from chat_pdf.domain.generation import (
    HEAD_CHUNKS,
    SECTIONS_CHARS,
    _joined_titles,
    _spread,
    section_titles,
)
from chat_pdf.domain.models import Chunk
from chat_pdf.domain.rubrics import DEFAULT_PRESET, PRESETS, criteria
from chat_pdf.domain.slides import clean_slide_options

# slides / concept_map have their own prompts (domain/slides.py, domain/concept_map.py)
KINDS = (*INSTRUCTIONS, "slides", "concept_map")
ARTIFACT_BUDGET = 16000  # chars of document text per generation
EXCERPT_CHARS = 900
MAX_COLUMNS = 12
MAX_COLUMN_CHARS = 60
MAX_REQUEST_CHARS = 300
_WORD = re.compile(r"\w{3,}")


def clean_options(kind: str, options: dict[str, Any] | None) -> dict[str, Any]:
    """Validated options of `kind` (`chart.request`, `extraction.columns`, `rubric.preset`)."""
    if kind not in KINDS:
        raise InvalidInput("invalid_artifact_kind", f"Tipo de artefacto desconocido: {kind}")
    options = options or {}
    if kind == "chart":
        request = options.get("request")
        text = " ".join(request.split())[:MAX_REQUEST_CHARS] if isinstance(request, str) else ""
        return {"request": text} if text else {}
    if kind == "extraction":
        raw = options.get("columns")
        columns = (
            [" ".join(c.split()) for c in raw if isinstance(c, str)]
            if isinstance(raw, list)
            else []
        )
        columns = list(dict.fromkeys(c[:MAX_COLUMN_CHARS] for c in columns if c))[:MAX_COLUMNS]
        if not columns:
            raise InvalidInput("columns_required", "Indicá al menos una columna a extraer")
        return {"columns": columns}
    if kind == "rubric":
        preset = options.get("preset")
        return {"preset": preset if preset in PRESETS else DEFAULT_PRESET}
    if kind == "slides":
        return clean_slide_options(options)
    if kind == "concept_map":
        return clean_map_options(options)
    return {}


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _score(chunk: Chunk, wanted: set[str], numeric: bool) -> float:
    text = chunk.content
    hits = len(wanted & _words(text)) * 50
    digits = sum(ch.isdigit() for ch in text) if numeric else 0
    return hits + digits


def select_chunks(
    chunks: list[Chunk], budget: int = ARTIFACT_BUDGET, query: str = "", numeric: bool = False
) -> list[Chunk]:
    """Chunks fitting `budget`, in document order. Spread over the document by default; when
    there is a `query` (or `numeric` data is wanted) the best-scoring chunks win instead."""
    slots = max(budget // (EXCERPT_CHARS + 2), 1)
    wanted = _words(query)
    if wanted or numeric:
        ranked = sorted(chunks, key=lambda c: -_score(c, wanted, numeric))
        if ranked and _score(ranked[0], wanted, numeric) > 0:
            return sorted(ranked[:slots], key=lambda c: c.ordinal)
    head = chunks[:HEAD_CHUNKS]
    return head + _spread(chunks[HEAD_CHUNKS:], max(slots - len(head), 0))


def _excerpt(chunk: Chunk) -> str:
    page = f"[pág. {chunk.page_start}] " if chunk.page_start else ""
    return page + chunk.content[:EXCERPT_CHARS]


def artifact_messages(
    kind: str, options: dict[str, Any], chunks: list[Chunk]
) -> list[dict[str, str]]:
    query = options.get("request") or " ".join(options.get("columns", []))
    picked = select_chunks(chunks, query=query, numeric=kind == "chart")[
        : max(ARTIFACT_BUDGET // (EXCERPT_CHARS + 2), 1)
    ]
    sections = _joined_titles(section_titles(chunks), SECTIONS_CHARS) or "(sin secciones)"
    request = CHART_REQUEST.format(request=options["request"]) if options.get("request") else ""
    instruction = INSTRUCTIONS[kind].format(
        request=request,
        columns=", ".join(options.get("columns", [])),
        criteria="\n".join(
            f"- {c['id']}: {c['label']}" for c in criteria(options.get("preset", DEFAULT_PRESET))
        ),
    )
    user = CONTEXT.format(sections=sections, excerpts="\n\n".join(map(_excerpt, picked)))
    return [
        {"role": "system", "content": ARTIFACT_SYSTEM},
        {"role": "user", "content": user + instruction},
    ]
