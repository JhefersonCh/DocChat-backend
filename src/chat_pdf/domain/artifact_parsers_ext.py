"""Parsers of the verified-evidence kinds: audit, entity_graph, rubric, timeline.

Every item keeps a verbatim `quote` that was found in a real chunk (else it is dropped).
"""

import logging
from collections.abc import Callable
from typing import Any

from chat_pdf.domain.artifact_parsers import PageRefs, _items, _number, _text
from chat_pdf.domain.audit import audit_facts
from chat_pdf.domain.entity_graph import build_graph, entity_type
from chat_pdf.domain.numbers import find_numbers, parse_number
from chat_pdf.domain.rubrics import (
    DEFAULT_PRESET,
    PRESET_LABELS,
    STATUSES,
    criteria,
)
from chat_pdf.domain.text_match import norm
from chat_pdf.domain.timeline import normalize_date, valid_iso

NO_TEMPORAL_DATA = "No se encontró contenido temporal (fechas o eventos) en el documento."
NO_NUMBERS = "No se encontraron cifras suficientes para auditar."
FACT_KINDS = ("amount", "percent", "count", "date", "other")
MAX_EVENTS = 30
MAX_FACTS = 300
QUOTE_LIMIT = 400
log = logging.getLogger("uvicorn.error")


def merge_batches(kind: str, datas: list[Any]) -> Any:
    """Combine the answers of several LLM calls (audit/entity_graph): concatenate the lists."""
    keys = {"audit": ("facts",), "entity_graph": ("entities", "relations")}.get(kind)
    if keys is None:
        return next((d for d in datas if d is not None), None)
    valid = [d for d in datas if isinstance(d, dict)]
    if not valid:
        return None
    return {k: [i for d in valid for i in _items(d, k)] for k in keys}


# --- audit ------------------------------------------------------------------------------------


def _fact_value(raw: Any, quote: str) -> float | None:
    """The model's value, checked against the numbers the quote really contains. A value that
    only matches a quote number's mantissa ("44" for "44 millones") is rescaled; one that
    appears nowhere in the quote is dropped (it would be a hallucinated figure)."""
    value = _number(raw) if not isinstance(raw, str) else parse_number(raw)
    if value is None:
        return None
    nums = find_numbers(quote)
    for scaled, _mantissa, _ in nums:
        if abs(scaled - value) <= 1e-6 * max(abs(scaled), 1):
            return scaled
    for scaled, mantissa, _ in nums:
        if abs(mantissa - value) <= 1e-6 * max(abs(mantissa), 1):
            return scaled
    return None


def _audit(data: Any, refs: PageRefs, options: dict) -> dict | None:
    if not isinstance(data, dict):
        return None
    facts = []
    raw_facts = _items(data, "facts")[:MAX_FACTS]
    for raw in raw_facts:
        quote = _text(raw.get("quote"), QUOTE_LIMIT)
        ref = refs.locate(quote, raw.get("page"))
        label = _text(raw.get("label"), 120)
        value = _fact_value(raw.get("value"), quote) if ref else None
        if not (ref and label and value is not None):
            continue
        kind = raw.get("kind") if raw.get("kind") in FACT_KINDS else "other"
        facts.append({
            "label": label,
            "value": value,
            "unit": _text(raw.get("unit"), 20),
            "kind": kind,
            "group": _text(raw.get("group"), 120) or None,
            "is_total": raw.get("is_total") is True,
            "quote": quote,
            **ref,
        })  # fmt: skip
    log.info("audit: kept %d of %d facts", len(facts), len(raw_facts))
    if not facts:
        return {"findings": [], "facts_count": 0, "checked_groups": 0, "checked": {},
                "message": NO_NUMBERS}  # fmt: skip
    return audit_facts(facts)


# --- entity graph -----------------------------------------------------------------------------


def _entity_graph(data: Any, refs: PageRefs, options: dict) -> dict | None:
    entities = []
    for raw in _items(data, "entities"):
        name = _text(raw.get("name"), 80)
        if not name:
            continue
        aliases = [a for a in (_text(x, 80) for x in raw.get("aliases") or []) if a]
        entities.append({
            "name": name,
            "type": entity_type(raw.get("type")),
            "description": _text(raw.get("description"), 400),
            "aliases": aliases[:6],
            **refs.ref(raw.get("page")),
        })  # fmt: skip
    relations = []
    for raw in _items(data, "relations"):
        quote = _text(raw.get("quote"), QUOTE_LIMIT)
        ref = refs.locate(quote, raw.get("page"))
        source, target = _text(raw.get("source"), 80), _text(raw.get("target"), 80)
        label = _text(raw.get("label"), 60)
        if ref and source and target and label:
            relations.append({"source": source, "target": target, "label": label,
                              "quote": quote, **ref})  # fmt: skip
    log.info("entity_graph: %d entities, kept %d of %d relations", len(entities), len(relations),
             len(_items(data, "relations")))  # fmt: skip
    return build_graph(entities, relations)


# --- rubric -----------------------------------------------------------------------------------


def _rubric(data: Any, refs: PageRefs, options: dict) -> dict | None:
    preset = options.get("preset") if options.get("preset") in PRESET_LABELS else DEFAULT_PRESET
    answers = {}
    for raw in _items(data, "criteria"):
        key = norm(_text(raw.get("id"), 60))
        if key:
            answers[key] = raw
    if not answers:
        return None
    out = []
    for crit in criteria(preset):
        raw = answers.get(norm(crit["id"]), {})
        status = raw.get("status") if raw.get("status") in STATUSES else "no_encontrado"
        quote = _text(raw.get("quote"), QUOTE_LIMIT)
        ref = refs.locate(quote, raw.get("page")) if status != "no_encontrado" else None
        if status != "no_encontrado" and ref is None:
            status, quote = "no_encontrado", ""  # evidence not verifiable
        out.append({
            **crit,
            "status": status,
            "justification": _text(raw.get("justification"), 500)
            or ("No se encontró evidencia en el documento." if status == "no_encontrado" else ""),
            "quote": quote or None,
            "suggestion": _text(raw.get("suggestion"), 400) or None,
            **(ref or {"page": None, "chunk_id": None}),
        })  # fmt: skip
    summary = {s: sum(1 for c in out if c["status"] == s) for s in STATUSES}
    return {"preset": preset, "preset_label": PRESET_LABELS[preset], "criteria": out,
            "summary": summary}  # fmt: skip


# --- timeline ---------------------------------------------------------------------------------


def _timeline(data: Any, refs: PageRefs, options: dict) -> dict | None:
    if not isinstance(data, dict | list):
        return None
    events = []
    raw_events = _items(data, "events")
    for raw in raw_events:
        quote = _text(raw.get("quote"), QUOTE_LIMIT)
        date_text = _text(raw.get("date_text"), 60)
        title = _text(raw.get("title"), 160)
        ref = refs.locate(quote, raw.get("page"))
        if not (ref and date_text and title and norm(date_text) in norm(quote)):
            continue
        events.append({
            "date_text": date_text,
            "date_iso": normalize_date(date_text) or valid_iso(raw.get("date_iso")),
            "title": title,
            "description": _text(raw.get("description"), 400),
            "quote": quote,
            **ref,
        })  # fmt: skip
    events = events[:MAX_EVENTS]
    log.info("timeline: kept %d of %d events", len(events), len(raw_events))
    if not events:
        return {"events": [], "ordered_by": None, "message": NO_TEMPORAL_DATA}
    dated = all(e["date_iso"] for e in events)
    if dated:
        events.sort(key=lambda e: e["date_iso"])
    return {"events": events, "ordered_by": "date" if dated else "document"}


EXT_PARSERS: dict[str, Callable[[Any, PageRefs, dict], dict | None]] = {
    "audit": _audit,
    "entity_graph": _entity_graph,
    "rubric": _rubric,
    "timeline": _timeline,
}
