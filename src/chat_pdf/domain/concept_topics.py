"""Concept maps built on semantic topics: the hierarchy (root -> one branch per topic -> concepts)
comes from the clusters of chunk embeddings; the model only NAMES branches and concepts, writes the
linking phrases and confirms the cross-links the embeddings proposed. Every link keeps a quote that
really exists in a chunk (verified), exactly like the positional flow. Pure functions."""

import math
from typing import Any
from uuid import UUID

from chat_pdf.domain.artifact_parsers import _text
from chat_pdf.domain.concept_cluster import Topic
from chat_pdf.domain.concept_map import (
    CALL_BUDGET,
    DEFAULT_DEPTH,
    DEFAULT_NODES,
    FOCUS_TEXT,
    LINK_CHARS,
    MAX_CALLS,
    MERGE_LINES,
    _match_focus,
    _unify,
    key_of,
)
from chat_pdf.domain.models import Chunk
from chat_pdf.domain.slides import EXCERPT, _excerpt, _locate

TOPIC_SYSTEM = (
    "Construís mapas conceptuales (estilo Novak) a partir de fragmentos de fuentes agrupados "
    "por temas. Cada fragmento viene marcado con su ubicación ([S1 pág. 3] o [S2 t 12s]). Usá "
    "SOLO información de los fragmentos, en el idioma de las fuentes. Respondé SOLO con JSON "
    "válido, sin texto adicional."
)
TOPIC_USER = (
    "Fuentes:\n{catalog}\n\nLos fragmentos se agruparon por significado (embeddings) en TEMAS; "
    "cada tema será una rama del mapa.\n\n{topics}\n\n{focus}"
    'Devolvé {{"branches": [{{"cluster": N, "label": "nombre del tema, 1 a 5 palabras", '
    '"summary": "1 oración", "quote": "cita textual EXACTA de un fragmento del tema", "source": '
    '"S1" o null, "page": N o null, "t": segundos o null}}], "concepts": [{{"label": "sustantivo '
    'o frase nominal de 1 a 5 palabras", "summary": "1 o 2 oraciones", "kind": "concepto" | '
    '"ejemplo", "aliases": [], "cluster": N, "parent": "label de otro concepto del mismo tema '
    'del que cuelga, o null si cuelga de la rama", "link": "frase de enlace de 1 a 4 palabras '
    "con la que cuelga de su padre o de la rama (incluye, se compone de, causa, se mide con, "
    'ejemplo de...)", "quote": "cita textual EXACTA y breve", "source": "S1" o null, "page": N '
    'o null, "t": segundos o null}}], "propositions": [{{"from": "label", "to": "label", "label": '
    '"frase de enlace de 1 a 4 palabras", "quote": "cita textual EXACTA que respalda la '
    'proposición", "source": "S1" o null, "page": N o null, "t": segundos o null}}]}}.\n'
    "Reglas: una rama por tema (usá el número del tema en cluster); {n} conceptos como máximo en "
    "total, repartidos entre los temas, de lo general a lo específico; reutilizá EXACTAMENTE el "
    "mismo label en concepts y propositions; en propositions poné SOLO relaciones entre conceptos "
    "de temas distintos o transversales (la jerarquía ya la dan parent y link); los ejemplos "
    "concretos van como kind=ejemplo; toda cita debe ser literal (si no, se descarta)."
)

MERGE_TOPIC_SYSTEM = (
    "Ordenás mapas conceptuales: nombrás el concepto raíz, unificás sinónimos y confirmás "
    "relaciones entre ramas. Respondé SOLO con JSON válido, sin texto adicional."
)
MERGE_TOPIC_USER = (
    "Ramas y conceptos:\n{tree}\n\nProposiciones:\n{props}\n\n{cross}{focus}"
    'Devolvé {{"root": {{"label": "concepto general que abarque todas las ramas, 1 a 5 '
    'palabras (un tema, no un ejemplo)", "summary": "1 oración"}}, "merge": [{{"canonical": '
    '"label EXACTO", "aliases": ["labels EXACTOS que son sinónimos del canónico"]}}], '
    '"cross": [{{"from": "label EXACTO", "to": "label EXACTO", "label": "frase de enlace de 1 a '
    '4 palabras", "quote": "cita textual EXACTA tomada de las citas listadas"}}]}}. '
    "Unificá solo conceptos que realmente significan lo mismo (merge vacío si no hay). En cross "
    "confirmá solo los pares propuestos que tengan una relación real (descartá el resto)."
)
CROSS_INTRO = (
    "Pares de conceptos de ramas distintas con significado cercano según embeddings (con la cita "
    "que respalda a cada uno):\n{pairs}\n\n"
)


def topic_batches(
    options: dict[str, Any], topics: list[Topic], aliases: dict[UUID, str], catalog: str
) -> list[list[dict[str, str]]]:
    """1 to MAX_CALLS calls: the topics (with their representative chunks) split in contiguous
    groups so each call stays within the character budget."""
    per_call = max(CALL_BUDGET // (EXCERPT + 2), 1)
    total = sum(len(t.picks) for t in topics)
    calls = max(1, min(MAX_CALLS, math.ceil(total / per_call), len(topics)))
    size = math.ceil(len(topics) / calls) or 1
    groups = [topics[i : i + size] for i in range(0, len(topics), size)] or [[]]
    n_total = options.get("max_nodes", DEFAULT_NODES)
    focus = FOCUS_TEXT.format(focus=options["focus"]) if options.get("focus") else ""
    out = []
    for group in groups:
        blocks = [
            f"TEMA {t.index} ({t.size} fragmentos):\n"
            + "\n\n".join(_excerpt(c, aliases) for c in t.picks)
            for t in group
        ]
        user = TOPIC_USER.format(
            catalog=catalog or "(sin catálogo)",
            topics="\n\n".join(blocks),
            focus=focus,
            n=max(6, round(n_total * 1.3 / len(groups))),
        )
        out.append([{"role": "system", "content": TOPIC_SYSTEM}, {"role": "user", "content": user}])
    return out


def topic_of_chunks(topics: list[Topic]) -> dict[str, int]:
    return {str(i): t.index for t in topics for i in t.members}


def assign_topics(
    concepts: dict[str, dict], topic_of: dict[str, int], valid: set[int] | None = None
) -> None:
    """Topic of a concept = that of the chunk its quote was found in (not the model's claim)"""
    for c in concepts.values():
        if valid is not None and c["topic"] in valid:
            continue  # the model placed it in a topic it was shown
        c["topic"] = None
        for r in c["refs"]:
            t = topic_of.get(r.get("chunk_id") or "")
            if t:
                c["topic"] = t
                break


def merge_messages_topics(
    concepts: dict[str, dict],
    props: list[dict],
    branches: dict[int, str],
    cross: list[tuple[str, str]],
    options: dict[str, Any],
) -> list[dict[str, str]]:
    names = {k: c["label"] for k, c in concepts.items()}
    by_topic: dict[Any, list[str]] = {}
    for k, c in concepts.items():
        if k not in branches.values():
            by_topic.setdefault(c["topic"], []).append(f"  - {c['label']} [{c['kind']}]")
    lines = []
    for t, k in branches.items():
        lines.append(f"{names[k]} (tema {t})")
        lines += by_topic.get(t, [])[:MERGE_LINES]
    for t, items in by_topic.items():
        if t not in branches:
            lines += items[:MERGE_LINES]
    plines = [f"- {names[p['from']]} —{p['label']}→ {names[p['to']]}" for p in props[:MERGE_LINES]]
    pairs = []
    for a, b in cross:
        quote = lambda k: concepts[k]["refs"][0]["quote"][:160] if concepts[k]["refs"] else ""  # noqa: E731
        pairs.append(f"- «{names[a]}» (cita: «{quote(a)}») ~ «{names[b]}» (cita: «{quote(b)}»)")
    focus = options.get("focus")
    user = MERGE_TOPIC_USER.format(
        tree="\n".join(lines) or "(ninguno)",
        props="\n".join(plines) or "(ninguna)",
        cross=CROSS_INTRO.format(pairs="\n".join(pairs)) if pairs else "",
        focus=FOCUS_TEXT.format(focus=focus) if focus else "",
    )
    return [
        {"role": "system", "content": MERGE_TOPIC_SYSTEM},
        {"role": "user", "content": user},
    ]


def cross_props(
    raw: list[dict],
    concepts: dict[str, dict],
    chunks: list[Chunk],
    aliases_by_doc: dict[UUID, str],
) -> list[dict]:
    """Cross-links the model confirmed: both ends must be known concepts and the quote must
    really exist in a chunk, otherwise they are dropped."""
    by_alias = {a.upper(): d for d, a in aliases_by_doc.items()}
    known: dict[str, str] = {}
    for k, c in concepts.items():
        for name in (c["label"], *c["aliases"]):
            known.setdefault(key_of(name), k)
    out = []
    for item in raw:
        a, b = known.get(key_of(item.get("from") or "")), known.get(key_of(item.get("to") or ""))
        label = _text(item.get("label"), LINK_CHARS)
        ref = _locate(item.get("quote"), item, chunks, by_alias)
        if a and b and a != b and label and ref:
            out.append({"from": a, "to": b, "label": label, "ref": ref})
    return out


def _support(k: str, concepts: dict[str, dict], props: list[dict], children: dict[str, list[str]]):
    """A verified ref that backs concept `k`: its own, a proposition's or a descendant's."""
    if concepts[k]["refs"]:
        return concepts[k]["refs"][0]
    for p in props:
        if k in (p["from"], p["to"]):
            return p["ref"]
    for child in children.get(k, []):
        found = _support(child, concepts, props, children)
        if found:
            return found
    return None


def build_topic_map(
    concepts: dict[str, dict],
    props: list[dict],
    branches: dict[int, str],
    merge: dict[str, Any] | None,
    options: dict[str, Any],
) -> dict[str, Any] | None:
    """The verified map `{root_id, nodes, links}`: root -> branches (one per topic) -> concepts
    (hung from their `parent` when it is a concept, else from the branch), depth-limited."""
    merge = merge or {}
    concepts, props, resolve = _unify(concepts, props, merge)
    branch_of = {t: resolve.get(k, k) for t, k in branches.items()}
    branch_of = {t: k for t, k in branch_of.items() if k in concepts}
    topics_present = {c["topic"] for c in concepts.values() if c["topic"] is not None}
    for t in sorted(topics_present - set(branch_of)):  # the model skipped a branch: first concept
        first = next(
            (k for k, c in concepts.items() if c["topic"] == t and c["kind"] != "ejemplo"), None
        )
        if first and first not in branch_of.values():
            branch_of[t] = first
    if not concepts:
        return None
    size = {t: sum(1 for c in concepts.values() if c["topic"] == t) for t in branch_of}

    root = None
    focus = options.get("focus")
    label = merge.get("root")
    if focus:
        root = _match_focus(concepts, focus)
        label = None if root else focus
    elif label and key_of(label) in concepts:
        root = key_of(label)
    if root is None and label:
        root = key_of(label)
        text = label[:1].upper() + label[1:]
        concepts[root] = {
            "label": text, "summary": merge.get("root_summary", ""), "kind": "concepto",
            "aliases": [], "refs": [], "parent": None, "link": None, "topic": None,
        }  # fmt: skip
    if root is None:
        if not branch_of:
            return None
        root = branch_of[max(branch_of, key=lambda t: size.get(t, 0))]

    parent: dict[str, str] = {}
    for b in branch_of.values():
        if b != root:
            parent[b] = root
    for k, c in concepts.items():
        if k == root or k in parent:
            continue
        pk = key_of(c["parent"]) if c.get("parent") else None
        pk = resolve.get(pk, pk) if pk else None
        if pk and pk in concepts and pk != k and concepts[pk]["kind"] != "ejemplo":
            parent[k] = pk
        else:
            parent[k] = branch_of.get(c["topic"]) or root
        if parent[k] == k:
            parent[k] = root
    for k in list(parent):  # break cycles by hanging the node from the root
        seen, cur = {k}, parent.get(k)
        while cur is not None and cur != root:
            if cur in seen:
                parent[k] = root
                break
            seen.add(cur)
            cur = parent.get(cur)

    children: dict[str, list[str]] = {}
    for k, p in parent.items():
        children.setdefault(p, []).append(k)
    level = {root: 0}
    order = [root]
    top: dict[str, str] = {}
    for k in order:
        for child in children.get(k, []):
            level[child] = level[k] + 1
            top[child] = child if k == root else top[k]
            order.append(child)
    depth = options.get("depth", DEFAULT_DEPTH)
    link_for: dict[str, tuple[str, dict]] = {}
    keep: list[str] = [root]
    rank: dict[tuple[int, str], int] = {}
    for k in sorted(order[1:], key=lambda k: level[k]):
        if level[k] > depth or parent[k] not in link_for and parent[k] != root:
            continue
        pair = next((p for p in props if {p["from"], p["to"]} == {k, parent[k]}), None)
        ref = pair["ref"] if pair else _support(k, concepts, props, children)
        if ref is None:
            continue
        phrase = pair["label"] if pair else (concepts[k].get("link") or "incluye")
        link_for[k] = (phrase[:LINK_CHARS], ref)
        keep.append(k)
    if len(keep) < 2:
        return None
    limit = options.get("max_nodes", DEFAULT_NODES)

    def fair(k: str):  # fill level by level, alternating between the branches
        key = (level[k], top.get(k, k))
        rank[key] = rank.get(key, -1) + 1
        return (level[k], rank[key], order.index(top.get(k, k)))

    kept = [root, *sorted(keep[1:], key=fair)[: limit - 1]]
    ids = {k: f"n{i + 1}" for i, k in enumerate(kept)}
    links = []
    used_pairs = set()
    for k in kept[1:]:
        phrase, ref = link_for[k]
        used_pairs.add(frozenset((k, parent[k])))
        links.append(
            {
                "id": f"e{len(links) + 1}", "source": ids[parent[k]], "target": ids[k],
                "label": phrase, "kind": "jerarquia", "refs": [ref],
            }
        )  # fmt: skip
    seen_cross = set()
    for p in props:
        pair = frozenset((p["from"], p["to"]))
        ident = (p["from"], p["to"], key_of(p["label"]))
        if (
            p["from"] in ids
            and p["to"] in ids
            and pair not in used_pairs
            and ident not in seen_cross
        ):
            seen_cross.add(ident)
            links.append(
                {
                    "id": f"e{len(links) + 1}", "source": ids[p["from"]], "target": ids[p["to"]],
                    "label": p["label"], "kind": "cruce", "refs": [p["ref"]],
                }
            )  # fmt: skip
    nodes = []
    for k in kept:
        c = concepts[k]
        refs = list(c["refs"]) or ([link_for[k][1]] if k in link_for else [])
        if not refs:
            refs = [
                r for lk in links if ids[k] in (lk["source"], lk["target"]) for r in lk["refs"]
            ][:1]
        if not refs:
            continue
        nodes.append(
            {
                "id": ids[k], "label": c["label"], "summary": c["summary"], "level": level[k],
                "kind": c["kind"], "aliases": c["aliases"], "refs": refs,
            }
        )  # fmt: skip
    return {"root_id": ids[root], "nodes": nodes, "links": links}
