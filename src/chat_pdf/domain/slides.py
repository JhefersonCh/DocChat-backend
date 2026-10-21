"""Slide decks (Estudio): suitability pre-check, generation prompts and the verified-deck parser.

Every content slide keeps only refs whose `quote` really exists in a chunk (page or time located);
a slide left without a verified ref is dropped. Visuals must come from the offered segment list
(images / regions / frames) and chart specs from numbers present in the text. Pure functions."""

import math
from collections.abc import Iterable
from typing import Any
from uuid import UUID

from chat_pdf.domain.artifact_parsers import PageRefs, _chart, _items, _text
from chat_pdf.domain.generation import extract_json
from chat_pdf.domain.models import Chunk, Document
from chat_pdf.domain.numbers import find_numbers
from chat_pdf.domain.text_match import quote_in

STYLES = ("ejecutiva", "académica", "didáctica")
DEFAULT_STYLE = "ejecutiva"
LAYOUTS = ("title", "bullets", "two_column", "image", "chart", "quote", "closing")
BARE_LAYOUTS = ("title", "closing")  # need no refs
MIN_SLIDES, MAX_SLIDES = 3, 24
MAX_CALLS = 3
SLIDE_BUDGET = 16000  # chars of source text per call
EXCERPT = 900
MAX_VISUALS = 40
MAX_BULLETS = 8
MAX_REFS = 4
MAX_AUDIENCE = 120
ASSESS_BUDGET = 9000
SUMMARY_CHARS = 500


def clean_slide_options(options: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n = options.get("slides")
    if isinstance(n, int) and not isinstance(n, bool):
        out["slides"] = max(MIN_SLIDES, min(n, MAX_SLIDES))
    audience = options.get("audience")
    if isinstance(audience, str) and " ".join(audience.split()):
        out["audience"] = " ".join(audience.split())[:MAX_AUDIENCE]
    if options.get("style") in STYLES:
        out["style"] = options["style"]
    return out


def default_slide_count(chunks: list[Chunk]) -> int:
    chars = sum(len(c.content) for c in chunks if c.modality in ("text", "audio"))
    return max(6, min(round(chars / 2500), 14))


# --- sampling ---------------------------------------------------------------------------------


def _label(chunk: Chunk, aliases: dict[UUID, str]) -> str:
    alias = aliases.get(chunk.document_id) if chunk.document_id and len(aliases) > 1 else None
    if chunk.page_start:
        where = f"pág. {chunk.page_start}"
    elif chunk.t_start is not None:
        where = f"t {chunk.t_start:.0f}s"
    else:
        where = ""
    return f"[{' '.join(p for p in (alias, where) if p)}]" if alias or where else "[]"


def _excerpt(chunk: Chunk, aliases: dict[UUID, str]) -> str:
    return f"{_label(chunk, aliases)} {' '.join(chunk.content.split())[:EXCERPT]}"


def visual_segments(chunks: list[Chunk]) -> list[Chunk]:
    """Pictures the deck may show: images, regions and video frames with a derived file
    (tiles are skipped), captioned ones first, at most MAX_VISUALS spread over the sources."""
    found = [
        c
        for c in chunks
        if c.modality in ("image", "frame") and c.asset_key and c.id and c.origin != "tile"
    ]
    found.sort(key=lambda c: (c.text_status != "ready", c.ordinal))
    head = found[:MAX_VISUALS]
    return sorted(head, key=lambda c: c.ordinal)


def _sample(chunks: list[Chunk], budget: int) -> list[Chunk]:
    """Text/transcript chunks fitting `budget`, spread evenly over each source (and the doc)."""
    slots = max(budget // (EXCERPT + 2), 1)
    pool = [c for c in chunks if c.modality in ("text", "audio") and c.content.strip()]
    groups: dict[Any, list[Chunk]] = {}
    for c in pool:
        groups.setdefault(c.document_id, []).append(c)
    per = max(slots // max(len(groups), 1), 1)
    picked: list[Chunk] = []
    for group in groups.values():
        if len(group) <= per:
            picked += group
        else:
            step = len(group) / per
            picked += [group[int(i * step)] for i in range(per)]
    from chat_pdf.domain.artifact_batches import numeric_chunks  # (cycle: batches -> artifacts)

    extra = [c for c in numeric_chunks(pool, 2) if c not in picked]
    return sorted([*picked, *extra], key=lambda c: c.ordinal)


# --- suitability pre-check --------------------------------------------------------------------

ASSESS_SYSTEM = (
    "Evaluás si el material de una conversación sirve para armar una presentación de "
    "diapositivas. Una presentación necesita ideas claras, estructura (secciones o hilo "
    "narrativo) y contenido suficiente. NO sirve material muy escaso, un listado de cifras "
    "sueltas sin contexto, un formulario o texto casi ilegible. Respondé SOLO con JSON válido."
)
ASSESS_USER = (
    "Fuentes:\n{catalog}\n\nResumen:\n{overview}\n\nFragmentos de muestra:\n\n{excerpts}\n\n"
    'Devolvé {{"suitable": true | false, "score": número de 0 a 1, "reasons": ["1 a 4 motivos '
    'breves, en el idioma del material"], "recommended_slides": entero de 4 a 20, '
    '"audience": "público objetivo sugerido o null", "outline_hint": ["3 a 8 títulos del '
    'esquema sugerido"]}}.'
)


def assess_messages(
    entries: list[tuple[str, Document, list[Chunk]]], chunks: list[Chunk]
) -> list[dict[str, str]]:
    from chat_pdf.domain.catalog import _line

    aliases = {doc.id: alias for alias, doc, _ in entries}
    catalog = "\n".join(_line(alias, doc, cs) for alias, doc, cs in entries)
    summaries = [
        f"{alias} · {' '.join(doc.summary.split())[:SUMMARY_CHARS]}"
        for alias, doc, _ in entries
        if doc.summary
    ]
    excerpts = "\n\n".join(_excerpt(c, aliases) for c in _sample(chunks, ASSESS_BUDGET))
    user = ASSESS_USER.format(
        catalog=catalog,
        overview="\n".join(summaries) or "(sin resumen)",
        excerpts=excerpts or "(sin texto)",
    )
    return [{"role": "system", "content": ASSESS_SYSTEM}, {"role": "user", "content": user}]


def parse_assessment(text: str, fallback_slides: int) -> dict[str, Any] | None:
    data = extract_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("suitable"), bool):
        return None
    raw = data.get("score")
    score = float(raw) if isinstance(raw, int | float) and not isinstance(raw, bool) else None
    if score is None or not math.isfinite(score):
        score = 1.0 if data["suitable"] else 0.0
    score = round(max(0.0, min(score, 1.0)), 2)
    recommended = data.get("recommended_slides")
    if isinstance(recommended, bool) or not isinstance(recommended, int):
        recommended = fallback_slides
    audience = _text(data.get("audience"), MAX_AUDIENCE) or None
    return {
        "suitable": data["suitable"] and score >= 0.35,
        "score": score,
        "reasons": [t for r in _list(data.get("reasons")) if (t := _text(r, 240))][:4],
        "recommended_slides": max(MIN_SLIDES, min(recommended, MAX_SLIDES)),
        "audience": audience,
        "outline_hint": [t for r in _list(data.get("outline_hint")) if (t := _text(r, 100))][:8],
    }


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


# --- generation -------------------------------------------------------------------------------

SLIDES_SYSTEM = (
    "Armás presentaciones de diapositivas a partir de fragmentos de fuentes. Cada fragmento viene "
    "marcado con su ubicación ([S1 pág. 3] o [S2 t 12s]). Usá SOLO información de los "
    "fragmentos, en el idioma de las fuentes. Respondé SOLO con JSON válido, sin texto adicional."
)
SLIDES_USER = (
    "Fuentes:\n{catalog}\n\nSecciones: {sections}\n\nFragmentos:\n\n{excerpts}\n\n{visuals}"
    "Estilo: {style}. Público: {audience}.\n{part}\n"
    'Devolvé {{"title": "...", "subtitle": "...", "slides": [{{"layout": "title | bullets | '
    'two_column | image | chart | quote | closing", "title": "...", "bullets": ["..."], "notes": '
    '"notas del orador, 2 a 4 oraciones", "refs": [{{"quote": "cita textual EXACTA y breve '
    'copiada de un fragmento", "source": "S1" o null, "page": N o null, "t": segundos o null}}], '
    '"visual": null | {{"kind": "image", "segment_id": "V1"}} | {{"kind": "chart", "spec": '
    '{{"type": "bar | line | pie", "title": "...", "labels": ["..."], "series": [{{"name": '
    '"...", "values": [números]}}]}}}}}}]}}.\n'
    "Reglas: toda diapositiva de contenido lleva 1 a 3 refs con cita literal de un fragmento "
    "(sin ella se descarta). bullets: 2 a 5 frases cortas (máx. 18 palabras); two_column: "
    "bullets pares = columna 1, impares = columna 2 con cantidad pareja; quote: una sola "
    "cita destacada en bullets[0]; image: elegí un id de la lista de imágenes y solo si ilustra "
    "el contenido; chart: solo con números que aparecen escritos en los fragmentos, no "
    "inventes ni estimes. No repitas contenido entre diapositivas."
)
VISUALS_TEXT = "Imágenes disponibles (usá solo estos id en visual.segment_id):\n{items}\n\n"


def _visual_line(index: int, chunk: Chunk, aliases: dict[UUID, str]) -> str:
    caption = " ".join(chunk.content.split())[:140] or "(sin descripción)"
    return f"V{index} · {_label(chunk, aliases)} · {caption}"


def slide_batches(
    options: dict[str, Any],
    chunks: list[Chunk],
    aliases: dict[UUID, str],
    catalog: str,
) -> list[list[dict[str, str]]]:
    """1 to MAX_CALLS calls: the sampled text is split in contiguous groups (a map over the
    sections); the first writes the cover, the last the closing, each its share of slides."""
    from chat_pdf.domain.generation import SECTIONS_CHARS, _joined_titles, section_titles

    total = options.get("slides") or default_slide_count(chunks)
    picked = _sample(chunks, SLIDE_BUDGET * MAX_CALLS)
    per_call = max(SLIDE_BUDGET // (EXCERPT + 2), 1)
    calls = max(1, min(MAX_CALLS, math.ceil(len(picked) / per_call)))
    size = math.ceil(len(picked) / calls) or 1
    groups = [picked[i : i + size] for i in range(0, len(picked), size)] or [[]]
    visuals = visual_segments(chunks)
    shown = (
        VISUALS_TEXT.format(
            items="\n".join(_visual_line(i, c, aliases) for i, c in enumerate(visuals, 1))
        )
        if visuals
        else ""
    )
    sections = _joined_titles(section_titles(chunks), SECTIONS_CHARS) or "(sin secciones)"
    out = []
    for index, group in enumerate(groups):
        share = max(2, round(total * len(group) / max(len(picked), 1))) if calls > 1 else total
        parts = [f"Escribí {share} diapositivas en total para esta parte."]
        if index == 0:
            parts.append("Empezá con una diapositiva layout=title (portada).")
        else:
            parts.append('Sin portada: "title" y "subtitle" pueden ir vacíos.')
        if index == len(groups) - 1:
            parts.append("Terminá con una diapositiva layout=closing (cierre / conclusiones).")
        else:
            parts.append("Sin diapositiva de cierre: la presentación continúa en otra parte.")
        user = SLIDES_USER.format(
            catalog=catalog or "(sin catálogo)",
            sections=sections,
            excerpts="\n\n".join(_excerpt(c, aliases) for c in group),
            visuals=shown,
            style=options.get("style", DEFAULT_STYLE),
            audience=options.get("audience") or "general",
            part=" ".join(parts),
        )
        out.append(
            [{"role": "system", "content": SLIDES_SYSTEM}, {"role": "user", "content": user}]
        )
    return out


# --- parsing ----------------------------------------------------------------------------------


def _where(chunk: Chunk) -> dict[str, Any]:
    return {
        "page": chunk.page_start,
        "t_start": chunk.t_start,
        "t_end": chunk.t_end,
        "bbox": chunk.bbox,
        "kind": {"audio": "audio", "frame": "video", "image": "image"}.get(chunk.modality, "text"),
        "chunk_id": str(chunk.id) if chunk.id else None,
        "document_id": str(chunk.document_id) if chunk.document_id else None,
    }


def _locate(quote: Any, raw: dict, chunks: list[Chunk], aliases: dict[str, UUID]) -> dict | None:
    """The chunk that really holds `quote` (the claimed source / page / time first), or None."""
    text = _text(quote, 400)
    if not text:
        return None
    wanted = aliases.get(str(raw.get("source") or "").upper())
    page = raw.get("page") if isinstance(raw.get("page"), int) else None
    t = raw.get("t") if isinstance(raw.get("t"), int | float) else None

    def rank(c: Chunk) -> tuple[bool, bool]:
        near = (
            page is not None
            and bool(c.page_start)
            and c.page_start <= page <= (c.page_end or c.page_start)
        ) or (
            t is not None
            and c.t_start is not None
            and c.t_start - 1 <= t <= (c.t_end if c.t_end is not None else c.t_start) + 1
        )
        return (wanted is not None and c.document_id != wanted, not near)

    for c in sorted(chunks, key=rank):
        if c.content and quote_in(text, c.content):
            return {"quote": text, **_where(c)}
    return None


def _numbers_present(spec: dict, chunks: list[Chunk]) -> bool:
    """Every value of the chart appears (scaled or as written) among the numbers of the text."""
    known: list[float] = []
    for c in chunks:
        for scaled, mantissa, _ in find_numbers(c.content):
            known += [scaled, mantissa]

    def seen(v: float) -> bool:
        return any(abs(k - v) <= 1e-6 * max(abs(k), 1) for k in known)

    return all(seen(v) for s in spec["series"] for v in s["values"])


def _visual(
    raw: Any, visuals: dict[str, Chunk], chunks: list[Chunk], refs: PageRefs
) -> dict | None:
    if not isinstance(raw, dict):
        return None
    if raw.get("kind") == "image":
        chunk = visuals.get(str(raw.get("segment_id") or "").strip().upper())
        if chunk is None:
            return None
        return {
            "kind": "image",
            "segment_id": str(chunk.id),
            "caption": " ".join(chunk.content.split())[:200],
            **{k: v for k, v in _where(chunk).items() if k in ("page", "t_start", "document_id")},
        }
    if raw.get("kind") == "chart":
        parsed = _chart({"chart": raw.get("spec")}, refs)
        spec = parsed.get("chart") if parsed else None
        if not spec or not _numbers_present(spec, chunks):
            return None
        return {"kind": "chart", "spec": spec}
    return None


def _slide(raw: dict, chunks, aliases, visuals, refs) -> dict | None:
    layout = raw.get("layout") if raw.get("layout") in LAYOUTS else "bullets"
    bullets = [t for b in _list(raw.get("bullets")) if (t := _text(b, 240))][:MAX_BULLETS]
    title = _text(raw.get("title"), 140)
    if not title and not bullets:
        return None
    found = [_locate(r.get("quote"), r, chunks, aliases) for r in _items(raw, "refs")]
    slide_refs: list[dict] = []
    seen: set = set()
    for r in found:
        if r and (key := (r["chunk_id"], r["quote"])) not in seen:
            seen.add(key)
            slide_refs.append(r)
    visual = _visual(raw.get("visual"), visuals, chunks, refs)
    if layout in ("image", "chart") and not (visual and visual["kind"] == layout):
        layout = "bullets"
    if layout in ("bullets", "two_column", "quote") and not bullets:
        return None
    if layout == "two_column" and len(bullets) < 2:
        layout = "bullets"
    return {
        "layout": layout,
        "title": title,
        "bullets": bullets,
        "notes": _text(raw.get("notes"), 1000),
        "refs": slide_refs[:MAX_REFS],
        "visual": visual,
    }


def parse_deck(
    texts: Iterable[str],
    chunks: list[Chunk],
    aliases_by_doc: dict[UUID, str],
    options: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """The verified deck `{title, subtitle, style, audience, slides}` or None when fewer than two
    content slides survive. Invalid refs are dropped; a content slide without refs is dropped."""
    options = options or {}
    refs = PageRefs(chunks)
    by_alias = {a.upper(): d for d, a in aliases_by_doc.items()}
    visuals = {f"V{i}": c for i, c in enumerate(visual_segments(chunks), 1)}
    title = subtitle = ""
    slides: list[dict] = []
    for text in texts:
        data = extract_json(text)
        if not isinstance(data, dict):
            continue
        title = title or _text(data.get("title"), 140)
        subtitle = subtitle or _text(data.get("subtitle"), 200)
        for raw in _items(data, "slides"):
            slide = _slide(raw, chunks, by_alias, visuals, refs)
            if slide is None:
                continue
            if slide["layout"] not in BARE_LAYOUTS and not slide["refs"]:
                continue
            slides.append(slide)
    content = [s for s in slides if s["layout"] not in BARE_LAYOUTS]
    if len(content) < 2:
        return None
    covers = [s for s in slides if s["layout"] == "title"][:1]
    closing = [s for s in slides if s["layout"] == "closing"][-1:]
    body = content[: max(options.get("slides") or default_slide_count(chunks), MIN_SLIDES)]
    title = title or (covers[0]["title"] if covers else body[0]["title"])
    cover = covers[0] if covers else {"layout": "title", "title": title, "bullets": [],
                                      "notes": "", "refs": [], "visual": None}  # fmt: skip
    cover = {**cover, "title": cover["title"] or title, "bullets": [subtitle] if subtitle else []}
    return {
        "title": title,
        "subtitle": subtitle,
        "style": options.get("style", DEFAULT_STYLE),
        "audience": options.get("audience"),
        "slides": [cover, *body, *closing],
    }
