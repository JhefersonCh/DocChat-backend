"""Entity graph cleanup: dedupe entities (names + aliases), resolve relations, cap size."""

from typing import Any

from chat_pdf.domain.text_match import norm

ENTITY_TYPES = ("persona", "organización", "lugar", "concepto", "objeto", "evento")
MAX_NODES = 40
MAX_EDGES = 80
_ARTICLES = {"el", "la", "los", "las", "un", "una", "de", "del", "don", "doña", "dr", "sr", "sra"}


def _key(name: str) -> str:
    tokens = [t for t in norm(name).split() if t not in _ARTICLES]
    return " ".join(tokens)


def entity_type(raw: Any) -> str:
    folded = norm(raw) if isinstance(raw, str) else ""
    return next((t for t in ENTITY_TYPES if norm(t) == folded), "concepto")


def merge_entities(raw: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Entities merged by normalized name/alias, plus `lookup`: normalized name -> index.

    A single-word name also merges into the only same-type entity whose name contains it
    ("Aureliano" -> "Aureliano Buendía"); ambiguous short names stay separate."""
    merged: list[dict] = []
    lookup: dict[str, int] = {}
    for item in raw:
        names = [item["name"], *item.get("aliases", [])]
        keys = [k for k in map(_key, names) if k]
        idx = next((lookup[k] for k in keys if k in lookup), None)
        if idx is None:
            idx = len(merged)
            merged.append({**item, "aliases": []})
        else:
            current = merged[idx]
            if len(item["description"]) > len(current["description"]):
                current["description"] = item["description"]
            if current["page"] is None:
                current["page"], current["chunk_id"] = item["page"], item.get("chunk_id")
        for name in names:
            if name != merged[idx]["name"] and name not in merged[idx]["aliases"]:
                merged[idx]["aliases"].append(name)
        for k in keys:
            lookup.setdefault(k, idx)
    return _absorb_short_names(merged, lookup)


def _absorb_short_names(merged: list[dict], lookup: dict[str, int]):
    parent = list(range(len(merged)))
    for i, ent in enumerate(merged):
        key = _key(ent["name"])
        if " " in key or not key:
            continue
        hosts = {
            j for j, other in enumerate(merged)
            if j != i and other["type"] == ent["type"] and key in _key(other["name"]).split()
        }  # fmt: skip
        if len(hosts) == 1:
            parent[i] = hosts.pop()

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    keep = [i for i in range(len(merged)) if root(i) == i]
    position = {old: new for new, old in enumerate(keep)}
    out = [merged[i] for i in keep]
    for i in range(len(merged)):
        if root(i) != i:
            host = out[position[root(i)]]
            host["aliases"].append(merged[i]["name"])
    remap = {k: position[root(i)] for k, i in lookup.items()}
    return out, remap


def build_graph(entities: list[dict], relations: list[dict]) -> dict | None:
    """Cleaned `{entities, relations}`: merged entities, relations resolved to entity ids
    (self-loops and duplicates dropped), at most MAX_NODES / MAX_EDGES, best-connected first."""
    merged, lookup = merge_entities(entities)
    seen: set[tuple[int, int, str]] = set()
    edges: list[tuple[int, int, dict]] = []
    for rel in relations:
        src, dst = lookup.get(_key(rel["source"])), lookup.get(_key(rel["target"]))
        sig = (src, dst, norm(rel["label"])) if src is not None and dst is not None else None
        if sig is None or src == dst or sig in seen:
            continue
        seen.add(sig)
        edges.append((src, dst, rel))
    degree = [0] * len(merged)
    for src, dst, _ in edges:
        degree[src] += 1
        degree[dst] += 1
    ranked = sorted(range(len(merged)), key=lambda i: (-degree[i], i))[:MAX_NODES]
    kept = sorted(ranked)
    ids = {old: f"e{n + 1}" for n, old in enumerate(kept)}
    out_entities = [
        {**{k: v for k, v in merged[i].items() if k != "aliases"}, "id": ids[i],
         "aliases": merged[i]["aliases"][:6]}
        for i in kept
    ]  # fmt: skip
    out_edges = [
        {**{k: v for k, v in rel.items() if k not in ("source", "target")},
         "source": ids[src], "target": ids[dst]}
        for src, dst, rel in edges
        if src in ids and dst in ids
    ][:MAX_EDGES]  # fmt: skip
    return {"entities": out_entities, "relations": out_edges} if out_entities else None
