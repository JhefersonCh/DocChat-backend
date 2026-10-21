"""Validation of what the model returns per artifact kind: invalid items are dropped.

Every parser returns the JSON-ready `data` of the artifact, or None when nothing valid is left.
"""

from collections.abc import Callable
from typing import Any

from chat_pdf.domain.generation import extract_json
from chat_pdf.domain.models import Chunk
from chat_pdf.domain.text_match import quote_in

NO_NUMERIC_DATA = "No se encontraron datos numéricos en el documento."
MAX_TEXT = 1200
MAX_MAP_DEPTH = 4
MAX_MAP_NODES = 80
CHART_TYPES = ("bar", "line", "pie")


def _text(value: Any, limit: int = MAX_TEXT) -> str:
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if value == value and abs(value) != float("inf") else None


class PageRefs:
    """Turns the page the model claims into `{page, chunk_id}` of a real chunk (else nulls)."""

    def __init__(self, chunks: list[Chunk]):
        self._chunks = [c for c in chunks if c.page_start]

    def ref(self, raw: Any) -> dict[str, Any]:
        page = raw if isinstance(raw, int) and not isinstance(raw, bool) else None
        for c in self._chunks:
            if page is not None and c.page_start <= page <= (c.page_end or c.page_start):
                return {
                    "page": page,
                    "chunk_id": str(c.id) if c.id else None,
                    "document_id": str(c.document_id) if c.document_id else None,
                }
        return {"page": None, "chunk_id": None}

    def locate(self, quote: Any, page: Any = None) -> dict[str, Any] | None:
        """`{page, chunk_id}` of the chunk that really contains `quote` (the claimed page's
        chunks are tried first), or None: unverified quotes must not reach the user."""
        text = _text(quote, 400)
        if not text:
            return None
        claimed = page if isinstance(page, int) and not isinstance(page, bool) else None

        def covers(c: Chunk) -> bool:
            return bool(claimed) and c.page_start <= claimed <= (c.page_end or c.page_start)

        for c in sorted(self._chunks, key=lambda c: not covers(c)):
            if quote_in(text, c.content):
                return {"page": claimed if covers(c) else c.page_start,
                        "chunk_id": str(c.id) if c.id else None,
                        "document_id": str(c.document_id) if c.document_id else None}  # fmt: skip
        return None


def _items(data: Any, key: str) -> list[dict]:
    if isinstance(data, dict):
        data = data.get(key)
    return [i for i in data if isinstance(i, dict)] if isinstance(data, list) else []


def _pairs(items: list[dict], a: str, b: str, refs: PageRefs) -> list[dict]:
    out = []
    for item in items:
        first, second = _text(item.get(a), 300), _text(item.get(b))
        if first and second:
            out.append({a: first, b: second, **refs.ref(item.get("page"))})
    return out


def _faq(data: Any, refs: PageRefs) -> dict | None:
    items = _pairs(_items(data, "items"), "question", "answer", refs)
    return {"items": items} if items else None


def _study_guide(data: Any, refs: PageRefs) -> dict | None:
    if not isinstance(data, dict):
        return None
    out = {
        "concepts": _pairs(_items(data, "concepts"), "term", "definition", refs),
        "glossary": _pairs(_items(data, "glossary"), "term", "definition", refs),
        "sections": _pairs(_items(data, "sections"), "title", "summary", refs),
    }
    return out if any(out.values()) else None


def _quiz(data: Any, refs: PageRefs) -> dict | None:
    out = []
    for item in _items(data, "questions"):
        options = item.get("options")
        index = item.get("answer_index")
        question = _text(item.get("question"), 400)
        if not (question and isinstance(options, list) and len(options) == 4):
            continue
        options = [_text(o, 300) for o in options]
        if not all(options) or len(set(options)) != 4:
            continue
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 4:
            continue
        out.append({
            "question": question,
            "options": options,
            "answer_index": index,
            "explanation": _text(item.get("explanation")),
            **refs.ref(item.get("page")),
        })  # fmt: skip
    return {"questions": out} if out else None


def _flashcards(data: Any, refs: PageRefs) -> dict | None:
    cards = _pairs(_items(data, "cards"), "front", "back", refs)
    return {"cards": cards} if cards else None


def _mind_map(data: Any, refs: PageRefs) -> dict | None:
    budget = [MAX_MAP_NODES]

    def node(raw: Any, depth: int) -> dict | None:
        title = _text(raw.get("title"), 120) if isinstance(raw, dict) else ""
        if not title or budget[0] <= 0:
            return None
        budget[0] -= 1
        children = raw.get("children") if depth < MAX_MAP_DEPTH else None
        kids = [k for c in children or [] if (k := node(c, depth + 1))]
        return {"title": title, **refs.ref(raw.get("page")), "children": kids}

    if isinstance(data, dict) and "root" in data:
        data = data["root"]
    root = node(data, 1)
    return {"root": root} if root else None


def _chart(data: Any, refs: PageRefs) -> dict | None:
    chart = data.get("chart") if isinstance(data, dict) else None
    if chart is None:
        return {"chart": None, "message": NO_NUMERIC_DATA} if isinstance(data, dict) else None
    if not isinstance(chart, dict) or chart.get("type") not in CHART_TYPES:
        return None
    labels = [_text(label, 60) for label in chart.get("labels") or []]
    series = []
    for s in chart.get("series") or []:
        values = [_number(v) for v in s.get("values") or []] if isinstance(s, dict) else []
        if len(values) == len(labels) and all(v is not None for v in values):
            series.append({"name": _text(s.get("name"), 60), "values": values})
    if len(labels) < 2 or not all(labels) or not series:
        return {"chart": None, "message": NO_NUMERIC_DATA}
    if chart["type"] == "pie":
        series = series[:1]
    return {"chart": {
        "type": chart["type"],
        "title": _text(chart.get("title"), 120),
        "labels": labels,
        "series": series,
        **refs.ref(chart.get("page")),
    }}  # fmt: skip


def _extraction(data: Any, refs: PageRefs, columns: list[str]) -> dict | None:
    rows = []
    for row in _items(data, "rows"):
        cells = {}
        for col in columns:
            cell = row.get(col)
            cell = cell if isinstance(cell, dict) else {"value": cell}
            value = cell.get("value")
            value = _text(value, 500) if isinstance(value, str) else (
                None if value is None or isinstance(value, dict | list) else str(value)
            )  # fmt: skip
            cells[col] = {
                "value": value or None,
                "quote": _text(cell.get("quote"), 400) or None,
                **refs.ref(cell.get("page")),
            }
        if any(c["value"] for c in cells.values()):
            rows.append(cells)
    return {"columns": columns, "rows": rows} if rows else None


_PARSERS: dict[str, Callable[..., dict | None]] = {
    "faq": _faq,
    "study_guide": _study_guide,
    "quiz": _quiz,
    "flashcards": _flashcards,
    "mind_map": _mind_map,
    "chart": _chart,
}


def parse_artifact(
    kind: str, text: str | list[str], chunks: list[Chunk], options: dict[str, Any] | None = None
) -> dict | None:
    """`text` is one model answer, or one per batch for the multi-call kinds (merged)."""
    from chat_pdf.domain.artifact_parsers_ext import EXT_PARSERS, merge_batches

    texts = [text] if isinstance(text, str) else text
    refs = PageRefs(chunks)
    if kind in EXT_PARSERS:
        data = merge_batches(kind, [extract_json(t) for t in texts])
        return EXT_PARSERS[kind](data, refs, options or {})
    data = extract_json(texts[0]) if texts else None
    if kind == "extraction":
        return _extraction(data, refs, (options or {}).get("columns", []))
    return _PARSERS[kind](data, refs)
