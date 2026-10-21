import json
from uuid import uuid4

from chat_pdf.domain.artifacts import clean_options
from chat_pdf.domain.concept_map import merge_expansion, parse_map
from chat_pdf.domain.models import Chunk

DOC = uuid4()
TEXTS = [
    "La fotosíntesis produce oxígeno y glucosa a partir de luz, agua y dióxido de carbono.",
    "La clorofila es parte del cloroplasto y absorbe la luz solar. "
    "Un ejemplo es la hoja del roble.",
    "El oxígeno liberado se mide con un electrodo en el laboratorio.",
]
CHUNKS = [Chunk(i, t, i + 1, i + 1, 0, 1, id=uuid4(), document_id=DOC) for i, t in enumerate(TEXTS)]
ALIASES = {DOC: "S1"}


def c(label, quote, **kw):
    return {"label": label, "summary": f"Resumen de {label}", "quote": quote, **kw}


def p(a, b, label, quote, **kw):
    return {"from": a, "to": b, "label": label, "quote": quote, **kw}


def test_concept_map_verifies_links_dedupes_and_builds_hierarchy_and_expansion():
    extraction = json.dumps({
        "concepts": [
            c("Fotosíntesis", "La fotosíntesis produce oxígeno", source="S1", page=1),
            c("Oxígeno", "El oxígeno liberado se mide"),
            c("Clorofila", "La clorofila es parte del cloroplasto", page=2),
            c("Hoja de roble", "la hoja del roble", kind="ejemplo"),
            c("Concepto inventado", "esta cita no existe en el texto"),
        ],
        "propositions": [
            p("Fotosíntesis", "Oxígeno", "produce", "La fotosíntesis produce oxígeno"),
            p("Oxígeno", "Electrodo", "se mide con", "se mide con un electrodo", page=3),
            p("Clorofila", "Fotosíntesis", "permite", "absorbe la luz solar"),
            p("Hoja de roble", "Clorofila", "ejemplo de", "Un ejemplo es la hoja del roble"),
            p("Fotosíntesis", "Mentira", "causa", "cita falsa que no está en las fuentes"),
            p("Oxígeno", "Clorofila", "cruza con", "El oxígeno liberado se mide"),
        ],
    })  # fmt: skip
    merge = json.dumps({"root": "Fotosíntesis", "merge": [
        {"canonical": "Oxígeno", "aliases": ["O2", "Electrodo"]}]})  # fmt: skip
    out = parse_map([extraction], merge, CHUNKS, ALIASES, {"depth": 3})
    by = {n["label"]: n for n in out["nodes"]}
    assert "Concepto inventado" not in by and "Mentira" not in by  # unverifiable: dropped
    assert by["Fotosíntesis"]["level"] == 0 and out["root_id"] == by["Fotosíntesis"]["id"]
    assert by["Oxígeno"]["aliases"] == ["Electrodo"]  # synonym merged, not a second node
    assert by["Hoja de roble"]["kind"] == "ejemplo"
    assert all(n["refs"] and n["refs"][0]["document_id"] == str(DOC) for n in out["nodes"])
    assert all(lk["refs"][0]["quote"] and lk["label"] for lk in out["links"])
    kinds = {lk["kind"] for lk in out["links"]}
    assert kinds == {"jerarquia", "cruce"}

    grown = json.dumps({
        "concepts": [c("Cloroplasto", "La clorofila es parte del cloroplasto")],
        "propositions": [
            p("Cloroplasto", "Clorofila", "contiene", "La clorofila es parte del cloroplasto")
        ],
    })  # fmt: skip
    new, added = merge_expansion(out, by["Clorofila"]["id"], [grown], CHUNKS, ALIASES)
    assert len(added["nodes"]) == 1 and len(new["nodes"]) == len(out["nodes"]) + 1
    added_node = next(n for n in new["nodes"] if n["id"] == added["nodes"][0])
    assert added_node["level"] == by["Clorofila"]["level"] + 1
    assert merge_expansion(new, by["Clorofila"]["id"], [grown], CHUNKS, ALIASES) is None
    assert clean_options("concept_map", {"depth": 9, "max_nodes": 2, "focus": " x "}) == {
        "depth": 4, "max_nodes": 8, "focus": "x"}  # fmt: skip


def test_topics_cluster_vectors_and_build_branches_from_clusters():
    from chat_pdf.domain.concept_cluster import plan_topics
    from chat_pdf.domain.concept_map import parse_extraction
    from chat_pdf.domain.concept_topics import assign_topics, build_topic_map, topic_of_chunks

    axes = ([1.0, 0.0, 0.1], [0.0, 1.0, 0.1])
    items = [(c, axes[i % 2]) for i, c in enumerate(CHUNKS * 3)]
    items = [
        (Chunk(i, c.content, 1, 1, 0, 1, id=uuid4(), document_id=DOC), v)
        for i, (c, v) in enumerate(items)
    ]  # noqa: E501
    topics = plan_topics(items, 2, 4, 8)
    assert len(topics) == 2 and sum(t.size for t in topics) == 9
    chunks = [c for c, _ in items]
    extraction = json.dumps({
        "branches": [{"cluster": 1, "label": "Luz", "quote": "absorbe la luz solar"}],
        "concepts": [
            c("Clorofila", "La clorofila es parte del cloroplasto", cluster=1, link="absorbe"),
            c("Cloroplasto", "La clorofila es parte del cloroplasto", cluster=1, parent="clorofila"),  # noqa: E501
        ],
        "propositions": [],
    })  # fmt: skip
    branches: dict[int, str] = {}
    concepts, props = parse_extraction([extraction], chunks, ALIASES, branches)
    assert set(topic_of_chunks(topics).values()) == {1, 2}
    assign_topics({}, {})
    merge = {"root": "Fotosíntesis", "root_summary": "", "groups": []}
    out = build_topic_map(concepts, props, branches, merge, {"depth": 3})
    by = {n["label"]: n for n in out["nodes"]}
    assert by["Fotosíntesis"]["level"] == 0 and by["Luz"]["level"] == 1
    assert by["Clorofila"]["level"] == 2 and by["Cloroplasto"]["level"] == 3
    assert all(lk["kind"] == "jerarquia" and lk["refs"] for lk in out["links"])
