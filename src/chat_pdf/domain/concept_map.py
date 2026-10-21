"""Concept maps (Novak): concepts joined by LABELED links (propositions), hierarchical from the most
general concept, plus cross-links between branches and examples as leaves.

Every link keeps only refs whose `quote` really exists in a chunk (page / region / time located);
an unverifiable link is dropped, a node survives only with a verified ref of its own or through a
verified link. The hierarchy (levels, `jerarquia` vs `cruce`) is computed deterministically by a
BFS from the root, so a flaky "merge" answer from the model never breaks the map. Pure functions."""

import math
from collections import deque
from typing import Any
from uuid import UUID

from chat_pdf.domain.artifact_parsers import _items, _text
from chat_pdf.domain.generation import extract_json
from chat_pdf.domain.models import Chunk
from chat_pdf.domain.slides import EXCERPT, _excerpt, _locate, _sample
from chat_pdf.domain.text_match import norm

MIN_DEPTH, MAX_DEPTH, DEFAULT_DEPTH = 2, 4, 3
MIN_NODES, MAX_NODES, DEFAULT_NODES = 8, 60, 30
MAX_FOCUS = 120
MAX_CALLS = 2  # extraction calls; the merge call is the third
CALL_BUDGET = 14000
LABEL_CHARS, LINK_CHARS, SUMMARY_CHARS = 60, 40, 320
MAX_REFS = 4
MAX_ALIASES = 5
EXPAND_NEW = 6
MAX_TOTAL_NODES = 200  # overall cap of a map grown by expansions (the depth limit does not apply)
EXPAND_BUDGET = 9000
MERGE_LINES = 140
KINDS = ("concepto", "ejemplo")


def clean_map_options(options: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    focus = options.get("focus")
    if isinstance(focus, str) and " ".join(focus.split()):
        out["focus"] = " ".join(focus.split())[:MAX_FOCUS]
    for key, low, high in (("depth", MIN_DEPTH, MAX_DEPTH), ("max_nodes", MIN_NODES, MAX_NODES)):
        n = options.get(key)
        if isinstance(n, int) and not isinstance(n, bool):
            out[key] = max(low, min(n, high))
    return out


def key_of(label: str) -> str:
    """Identity of a concept: accent/case/punctuation-insensitive, trivial plurals folded."""
    words = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in norm(label).split()]
    return " ".join(words)


# --- prompts ----------------------------------------------------------------------------------

EXTRACT_SYSTEM = (
    "Construís mapas conceptuales (estilo Novak) a partir de fragmentos de fuentes. Cada fragmento "
    "viene marcado con su ubicación ([S1 pág. 3] o [S2 t 12s]). Usá SOLO información de los "
    "fragmentos, en el idioma de las fuentes. Respondé SOLO con JSON válido, sin texto adicional."
)
EXTRACT_USER = (
    "Fuentes:\n{catalog}\n\nFragmentos:\n\n{excerpts}\n\n{focus}"
    'Devolvé {{"concepts": [{{"label": "sustantivo o frase nominal de 1 a 5 palabras", '
    '"summary": "1 o 2 oraciones", "kind": "concepto" | "ejemplo", "aliases": ["sinónimos '
    'usados en el texto"], "quote": "cita textual EXACTA y breve copiada de un fragmento donde '
    'aparece el concepto", "source": "S1" o null, "page": N o null, "t": segundos o null}}], '
    '"propositions": [{{"from": "label de un concepto", "to": "label de otro concepto", '
    '"label": "frase de enlace de 1 a 4 palabras con verbo o relación (causa, es parte de, se '
    'mide con, permite, produce, requiere, ejemplo de...)", "quote": "cita textual EXACTA que '
    'respalda la proposición completa", "source": "S1" o null, "page": N o null, "t": segundos '
    "o null}}]}}.\n"
    "Reglas: {n} conceptos como máximo, de lo más general a lo específico; los conceptos son "
    "ideas (no oraciones), reutilizá EXACTAMENTE el mismo label en concepts y en propositions; "
    "cada proposición se lee 'from — label → to' y debe ser verdadera según el texto; incluí "
    "relaciones entre ramas distintas (no solo padre-hijo); los ejemplos concretos van como "
    "kind=ejemplo; toda proposición lleva quote literal (sin ella se descarta)."
)
FOCUS_TEXT = "Tema central del mapa: {focus}. Organizá los conceptos alrededor de ese tema.\n"

MERGE_SYSTEM = (
    "Ordenás mapas conceptuales: unificás sinónimos y elegís el concepto más general. "
    "Respondé SOLO con JSON válido, sin texto adicional."
)
MERGE_USER = (
    "Conceptos:\n{concepts}\n\nProposiciones:\n{props}\n\n{focus}"
    'Devolvé {{"root": "label EXACTO del concepto más general {root_hint}", "merge": '
    '[{{"canonical": "label EXACTO elegido", "aliases": ["labels EXACTOS que son sinónimos '
    'o duplicados del canónico"]}}]}}. Unificá solo conceptos que realmente significan lo '
    "mismo; si no hay duplicados devolvé merge vacío."
)

EXPAND_SYSTEM = EXTRACT_SYSTEM
EXPAND_USER = (
    "Fuentes:\n{catalog}\n\nFragmentos relacionados:\n\n{excerpts}\n\n"
    'Concepto a expandir: "{label}" ({summary}).\nConceptos que ya están en el mapa: {existing}.\n'
    "Proponé de 3 a {n} SUBCONCEPTOS nuevos (más específicos, partes, causas, consecuencias, "
    "ejemplos) que el texto relacione DIRECTAMENTE con «{label}» (cada uno debe poder unirse a "
    "ese concepto con una proposición verdadera), y sus proposiciones. Una proposición "
    "puede unir un subconcepto nuevo con el concepto a expandir o con otro concepto del mapa "
    "(usá su label EXACTO). No repitas conceptos existentes como nuevos.\n"
    'Devolvé {{"concepts": [{{"label": "...", "summary": "...", "kind": "concepto" | '
    '"ejemplo", "aliases": [], "quote": "cita textual EXACTA", "source": "S1" o null, "page": '
    'N o null, "t": segundos o null}}], "propositions": [{{"from": "label", "to": "label", '
    '"label": "frase de enlace de 1 a 4 palabras", "quote": "cita textual EXACTA que respalda '
    'la proposición", "source": "S1" o null, "page": N o null, "t": segundos o null}}]}}.'
)


def extract_batches(
    options: dict[str, Any], chunks: list[Chunk], aliases: dict[UUID, str], catalog: str
) -> list[list[dict[str, str]]]:
    """1 to MAX_CALLS calls: the sampled text split in contiguous groups (map over sections)."""
    picked = _sample(chunks, CALL_BUDGET * MAX_CALLS)
    per_call = max(CALL_BUDGET // (EXCERPT + 2), 1)
    calls = max(1, min(MAX_CALLS, math.ceil(len(picked) / per_call)))
    size = math.ceil(len(picked) / calls) or 1
    groups = [picked[i : i + size] for i in range(0, len(picked), size)] or [[]]
    total = options.get("max_nodes", DEFAULT_NODES)
    focus = FOCUS_TEXT.format(focus=options["focus"]) if options.get("focus") else ""
    return [
        [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {
                "role": "user",
                "content": EXTRACT_USER.format(
                    catalog=catalog or "(sin catálogo)",
                    excerpts="\n\n".join(_excerpt(c, aliases) for c in group),
                    focus=focus,
                    n=max(8, round(total * 1.3 / len(groups))),
                ),
            },
        ]
        for group in groups
    ]


# --- extraction parsing -----------------------------------------------------------------------


def _add_ref(refs: list[dict], ref: dict | None) -> None:
    if (
        ref
        and len(refs) < MAX_REFS
        and (ref["chunk_id"], ref["quote"]) not in {(r["chunk_id"], r["quote"]) for r in refs}
    ):
        refs.append(ref)


def _concept(label: str, summary: str = "", kind: str = "concepto") -> dict[str, Any]:
    return {
        "label": label,
        "summary": summary,
        "kind": kind,
        "aliases": [],
        "refs": [],
        "parent": None,
        "link": None,
        "topic": None,
    }


def parse_extraction(
    texts: list[str],
    chunks: list[Chunk],
    aliases_by_doc: dict[UUID, str],
    branches: dict[int, str] | None = None,
) -> tuple[dict[str, dict], list[dict]]:
    """`(concepts by key, propositions)`; propositions without a verified quote are dropped and
    concepts may carry no ref yet (supported later through a link)."""
    by_alias = {a.upper(): d for d, a in aliases_by_doc.items()}
    concepts: dict[str, dict] = {}
    props: list[dict] = []

    def get(label: str, summary: str = "", kind: str = "concepto") -> dict | None:
        text = _text(label, LABEL_CHARS)
        text = text[:1].upper() + text[1:]
        k = key_of(text)
        if not k:
            return None
        if k not in concepts:
            concepts[k] = _concept(text, _text(summary, SUMMARY_CHARS), kind)
        return concepts[k]

    for text in texts:
        data = extract_json(text)
        if not isinstance(data, dict):
            continue
        for raw in _items(data, "branches") if branches is not None else []:
            topic = raw.get("cluster")
            b = get(raw.get("label"), raw.get("summary"))
            if b is None or not isinstance(topic, int) or topic in branches:
                continue
            branches[topic] = key_of(b["label"])
            b["topic"] = topic
            _add_ref(b["refs"], _locate(raw.get("quote"), raw, chunks, by_alias))
        for raw in _items(data, "concepts"):
            kind = raw.get("kind") if raw.get("kind") in KINDS else "concepto"
            c = get(raw.get("label"), raw.get("summary"), kind)
            if c is None:
                continue
            if not c["summary"]:
                c["summary"] = _text(raw.get("summary"), SUMMARY_CHARS)
            if raw.get("parent") and not c["parent"]:
                c["parent"] = _text(raw.get("parent"), LABEL_CHARS)
            if raw.get("link") and not c["link"]:
                c["link"] = _text(raw.get("link"), LINK_CHARS)
            if isinstance(raw.get("cluster"), int) and c["topic"] is None:
                c["topic"] = raw["cluster"]
            for a in raw.get("aliases") if isinstance(raw.get("aliases"), list) else []:
                alias = _text(a, LABEL_CHARS)
                if alias and key_of(alias) != key_of(c["label"]) and len(c["aliases"]) < 5:
                    c["aliases"].append(alias)
            _add_ref(c["refs"], _locate(raw.get("quote"), raw, chunks, by_alias))
        for raw in _items(data, "propositions"):
            label = _text(raw.get("label"), LINK_CHARS)
            ref = _locate(raw.get("quote"), raw, chunks, by_alias)
            a, b = get(raw.get("from")), get(raw.get("to"))
            if not (label and ref and a and b) or a is b:
                continue
            props.append(
                {"from": key_of(a["label"]), "to": key_of(b["label"]), "label": label, "ref": ref}
            )
    return concepts, props


# --- merge ------------------------------------------------------------------------------------


def merge_messages(
    concepts: dict[str, dict], props: list[dict], options: dict[str, Any]
) -> list[dict[str, str]]:
    names = {k: c["label"] for k, c in concepts.items()}
    lines = []
    for c in list(concepts.values())[:MERGE_LINES]:
        extra = f" (también: {', '.join(c['aliases'])})" if c["aliases"] else ""
        lines.append(f"- {c['label']}{extra} [{c['kind']}]")
    plines = [f"- {names[p['from']]} —{p['label']}→ {names[p['to']]}" for p in props[:MERGE_LINES]]
    focus = options.get("focus")
    user = MERGE_USER.format(
        concepts="\n".join(lines) or "(ninguno)",
        props="\n".join(plines) or "(ninguna)",
        focus=FOCUS_TEXT.format(focus=focus) if focus else "",
        root_hint=f"(o el que mejor representa «{focus}»)" if focus else "(un tema, no un ejemplo)",
    )
    return [{"role": "system", "content": MERGE_SYSTEM}, {"role": "user", "content": user}]


def parse_merge(text: str) -> dict[str, Any]:
    """`{root, groups: [(canonical, [aliases])]}`; anything malformed is simply ignored."""
    data = extract_json(text)
    if not isinstance(data, dict):
        return {"root": None, "groups": [], "cross": []}
    groups = []
    for g in _items(data, "merge"):
        names = [a for a in (g.get("aliases") if isinstance(g.get("aliases"), list) else [])]
        canonical = _text(g.get("canonical"), LABEL_CHARS)
        aliases = [t for a in names if (t := _text(a, LABEL_CHARS))]
        if canonical and aliases:
            groups.append((canonical, aliases))
    root = data.get("root")
    summary = ""
    if isinstance(root, dict):
        summary = _text(root.get("summary"), SUMMARY_CHARS)
        root = root.get("label")
    return {
        "root": _text(root, LABEL_CHARS) or None,
        "root_summary": summary,
        "groups": groups,
        "cross": _items(data, "cross"),
    }


def _unify(concepts: dict[str, dict], props: list[dict], merge: dict[str, Any]):
    """Apply the synonym groups: `(concepts, props, resolve)` over the surviving keys."""
    resolve: dict[str, str] = {}
    for canonical, aliases in merge.get("groups", []):
        present = [
            k for k in dict.fromkeys(key_of(n) for n in (canonical, *aliases)) if k in concepts
        ]
        if len(present) < 2:
            continue
        target = key_of(canonical) if key_of(canonical) in concepts else present[0]
        for k in present:
            resolve[k] = resolve.get(target, target)
    merged: dict[str, dict] = {}
    for k, c in concepts.items():
        t = resolve.get(k, k)
        if t not in merged:
            merged[t] = {
                **concepts[t],
                "aliases": list(concepts[t]["aliases"]),
                "refs": list(concepts[t]["refs"]),
            }
        if t != k:
            m = merged[t]
            m["aliases"] = [*m["aliases"], c["label"], *c["aliases"]]
            m["summary"] = max(m["summary"], c["summary"], key=len)
            for r in c["refs"]:
                _add_ref(m["refs"], r)
    for m in merged.values():
        seen = {key_of(m["label"])}
        m["aliases"] = [a for a in m["aliases"] if not (key_of(a) in seen or seen.add(key_of(a)))][
            :MAX_ALIASES
        ]
    out: list[dict] = []
    seen_p: set = set()
    for p in props:
        s, t = resolve.get(p["from"], p["from"]), resolve.get(p["to"], p["to"])
        ident = (s, t, key_of(p["label"]))
        if s == t or ident in seen_p:
            continue
        seen_p.add(ident)
        out.append({**p, "from": s, "to": t})
    return merged, out, resolve


def _match_focus(concepts: dict[str, dict], focus: str) -> str | None:
    wanted = set(norm(focus).split())
    best, best_score = None, 0.0
    for k, c in concepts.items():
        for name in (c["label"], *c["aliases"]):
            words = set(norm(name).split())
            if not words:
                continue
            score = len(wanted & words) / max(len(wanted | words), 1)
            if score > best_score:
                best, best_score = k, score
    return best if best_score >= 0.4 else None


def build_map(
    concepts: dict[str, dict],
    props: list[dict],
    merge: dict[str, Any] | None,
    options: dict[str, Any],
) -> dict[str, Any] | None:
    """The verified concept map `{root_id, nodes, links}` or None when nothing coherent is left."""
    concepts, props, _ = _unify(concepts, props, merge or {})
    degree: dict[str, int] = {}
    for p in props:
        degree[p["from"]] = degree.get(p["from"], 0) + 1
        degree[p["to"]] = degree.get(p["to"], 0) + 1
    alive = {k for k, c in concepts.items() if c["refs"] or k in degree}
    props = [p for p in props if p["from"] in alive and p["to"] in alive]
    if len(alive) < 2 or not props:
        return None
    root = None
    if options.get("focus"):
        root = _match_focus({k: concepts[k] for k in alive}, options["focus"])
    if root is None and merge and merge.get("root"):
        root = key_of(merge["root"]) if key_of(merge["root"]) in alive else None
        root = root or _match_focus({k: concepts[k] for k in alive}, merge["root"])
    if root is None or concepts[root]["kind"] == "ejemplo" and degree.get(root, 0) < 2:
        pool = [k for k in alive if concepts[k]["kind"] != "ejemplo"] or list(alive)
        root = max(pool, key=lambda k: (degree.get(k, 0), bool(concepts[k]["summary"])))
    return _hierarchy(concepts, props, root, options)


def _hierarchy(concepts, props, root, options) -> dict[str, Any] | None:
    depth = options.get("depth", DEFAULT_DEPTH)
    adjacency: dict[str, list[int]] = {}
    for i, p in enumerate(props):
        adjacency.setdefault(p["from"], []).append(i)
        adjacency.setdefault(p["to"], []).append(i)
    level = {root: 0}
    tree_edges: set[int] = set()
    queue = deque([root])
    while queue:
        k = queue.popleft()
        if k != root and concepts[k]["kind"] == "ejemplo":
            continue  # examples are leaves
        if level[k] >= depth:
            continue
        for i in adjacency.get(k, []):
            other = props[i]["to"] if props[i]["from"] == k else props[i]["from"]
            if other not in level:
                level[other] = level[k] + 1
                tree_edges.add(i)
                queue.append(other)
    degree = {k: len(adjacency.get(k, [])) for k in level}
    keep = sorted(level, key=lambda k: (level[k], -degree[k]))[
        : options.get("max_nodes", DEFAULT_NODES)
    ]
    keep_set = set(keep)
    ids = {k: f"n{i + 1}" for i, k in enumerate(keep)}
    links = []
    for i, p in enumerate(props):
        if p["from"] in keep_set and p["to"] in keep_set:
            links.append(
                {
                    "id": f"e{len(links) + 1}",
                    "source": ids[p["from"]],
                    "target": ids[p["to"]],
                    "label": p["label"],
                    "kind": "jerarquia" if i in tree_edges else "cruce",
                    "refs": [p["ref"]],
                }
            )
    if not links:
        return None
    nodes = []
    for k in keep:
        c = concepts[k]
        refs = list(c["refs"])
        if not refs:
            mine = [r for lk in links if ids[k] in (lk["source"], lk["target"]) for r in lk["refs"]]
            refs = mine[:1]
        nodes.append(
            {
                "id": ids[k],
                "label": c["label"],
                "summary": c["summary"],
                "level": level[k],
                "kind": c["kind"],
                "aliases": c["aliases"],
                "refs": refs,
            }
        )
    return {"root_id": ids[root], "nodes": nodes, "links": links}


def parse_map(
    texts: list[str],
    merge_text: str | None,
    chunks: list[Chunk],
    aliases_by_doc: dict[UUID, str],
    options: dict[str, Any],
) -> dict[str, Any] | None:
    concepts, props = parse_extraction(texts, chunks, aliases_by_doc)
    merge = parse_merge(merge_text) if merge_text else None
    return build_map(concepts, props, merge, options)


# --- expand -----------------------------------------------------------------------------------


def expand_messages(
    data: dict[str, Any],
    node: dict[str, Any],
    chunks: list[Chunk],
    aliases: dict[UUID, str],
    catalog: str,
    picked: list[Chunk] | None = None,
) -> list[dict[str, str]]:
    """`picked`: the chunks the embedding retriever found for the node; without them (no
    embeddings available) the lexical keyword match is the fallback."""
    if picked is None:
        from chat_pdf.domain.artifacts import select_chunks

        pool = [c for c in chunks if c.modality in ("text", "audio") and c.content.strip()]
        picked = select_chunks(pool, EXPAND_BUDGET, f"{node['label']} {node['summary']}")
    existing = ", ".join(n["label"] for n in data["nodes"][:80])
    user = EXPAND_USER.format(
        catalog=catalog or "(sin catálogo)",
        excerpts="\n\n".join(_excerpt(c, aliases) for c in picked),
        label=node["label"],
        summary=node["summary"] or "sin resumen",
        existing=existing,
        n=EXPAND_NEW,
    )
    return [{"role": "system", "content": EXPAND_SYSTEM}, {"role": "user", "content": user}]


def merge_expansion(
    data: dict[str, Any],
    node_id: str,
    texts: list[str],
    chunks: list[Chunk],
    aliases_by_doc: dict[UUID, str],
) -> tuple[dict[str, Any], dict[str, list[str]]] | None:
    """The map grown with up to EXPAND_NEW verified subconcepts around `node_id`, and the ids that
    were added (`{nodes, links}`), or None when nothing new and verifiable came back."""
    concepts, props = parse_extraction(texts, chunks, aliases_by_doc)
    nodes = {n["id"]: n for n in data["nodes"]}
    known: dict[str, str] = {}
    for n in data["nodes"]:
        for name in (n["label"], *n.get("aliases", [])):
            known.setdefault(key_of(name), n["id"])
    new_keys = [k for k in concepts if k not in known]
    existing_links = {(lk["source"], lk["target"], key_of(lk["label"])) for lk in data["links"]}

    def ident(k: str) -> str:
        return known.get(k) or f"new:{k}"

    def spread(level: dict[str, int]) -> list[dict]:
        accepted: list[dict] = []
        for _ in range(3):  # propagate levels outward from the seeds
            for p in props:
                a, b = ident(p["from"]), ident(p["to"])
                for x, y in ((a, b), (b, a)):
                    if x in level and y.startswith("new:") and y not in level:
                        level[y] = level[x] + 1
                        accepted.append({"node": y, "via": p})
        return accepted

    # Subconcepts must hang from the expanded node (directly or through new ones); only when the
    # model ignored it, fall back to anything that links to the map.
    level: dict[str, int] = {node_id: nodes[node_id]["level"]}
    accepted = spread(level)
    if not accepted:
        level = {n["id"]: n["level"] for n in data["nodes"]}
        accepted = spread(level)
    first = {a["node"]: a for a in accepted}
    order = [k for k in new_keys if f"new:{k}" in first]
    order.sort(key=lambda k: (level[f"new:{k}"], k != key_of(nodes[node_id]["label"])))
    chosen = order[: min(EXPAND_NEW, MAX_TOTAL_NODES - len(data["nodes"]))]
    if not chosen:
        return None
    next_n = max(int(n["id"][1:]) for n in data["nodes"]) + 1
    new_ids: dict[str, str] = {}
    out_nodes = list(data["nodes"])
    for k in chosen:
        new_ids[f"new:{k}"] = f"n{next_n}"
        next_n += 1
    added_nodes: list[str] = []
    for k in chosen:
        c = concepts[k]
        nid = new_ids[f"new:{k}"]
        refs = list(c["refs"]) or [first[f"new:{k}"]["via"]["ref"]]
        out_nodes.append(
            {
                "id": nid,
                "label": c["label"],
                "summary": c["summary"],
                "level": level[f"new:{k}"],
                "kind": c["kind"],
                "aliases": c["aliases"],
                "refs": refs,
            }
        )
        added_nodes.append(nid)
    resolve = {**{k: v for k, v in known.items()}, **{k: new_ids[f"new:{k}"] for k in chosen}}
    discovered: set[str] = set()
    out_links = list(data["links"])
    added_links: list[str] = []
    n_links = len(out_links)
    for p in props:
        s, t = resolve.get(p["from"]), resolve.get(p["to"])
        if not (s and t) or s == t or (s, t, key_of(p["label"])) in existing_links:
            continue
        if s not in added_nodes and t not in added_nodes:
            continue
        existing_links.add((s, t, key_of(p["label"])))
        fresh = [x for x in (s, t) if x in added_nodes and x not in discovered]
        discovered.update(fresh)
        n_links += 1
        out_links.append(
            {
                "id": f"e{n_links}",
                "source": s,
                "target": t,
                "label": p["label"],
                "kind": "jerarquia" if fresh else "cruce",
                "refs": [p["ref"]],
            }
        )
        added_links.append(f"e{n_links}")
    if not added_links:
        return None
    keep = {n["id"] for n in out_nodes}
    linked = {x for lk in out_links for x in (lk["source"], lk["target"])}
    out_nodes = [n for n in out_nodes if n["id"] in linked or n["id"] in keep - set(added_nodes)]
    added_nodes = [i for i in added_nodes if i in linked]
    return {**data, "nodes": out_nodes, "links": out_links}, {
        "nodes": added_nodes,
        "links": added_links,
    }


def node_by_id(data: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    return next((n for n in data.get("nodes", []) if n.get("id") == node_id), None)
