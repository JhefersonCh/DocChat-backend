import asyncio
import json
from uuid import uuid4

from test_artifacts import _setup

from chat_pdf.domain.artifact_batches import artifact_batches
from chat_pdf.domain.artifact_parsers import PageRefs, parse_artifact
from chat_pdf.domain.artifacts import clean_options
from chat_pdf.domain.entity_graph import MAX_EDGES, MAX_NODES, build_graph
from chat_pdf.domain.models import Chunk
from chat_pdf.domain.text_match import norm, quote_in
from chat_pdf.domain.timeline import normalize_date

TEXTS = [
    "El presupuesto inicial fue de $44.000.000 para la obra. Aureliano Buendía fundó Macondo.",
    "En marzo de 2025 se inauguró la planta. El equipo lo lidera María Gómez.",
    "Se invirtieron 44 millones en total.",
]
CHUNKS = [Chunk(i, t, i + 1, i + 1, 0, 1, id=uuid4()) for i, t in enumerate(TEXTS)]


def parse(kind, payload, options=None):
    return parse_artifact(kind, json.dumps(payload), CHUNKS, options)


def test_quote_matching_ignores_case_accents_whitespace_and_punctuation():
    assert quote_in("PRESUPUESTO   inicial fue de $44.000.000", TEXTS[0])
    assert quote_in("Maria Gomez", TEXTS[1]) and not quote_in("Maria Perez", TEXTS[1])
    assert not quote_in("", TEXTS[0]) and norm("Árbol,  Él") == "arbol el"


def test_locate_prefers_claimed_page_then_searches_everywhere():
    refs = PageRefs(CHUNKS)
    assert refs.locate("marzo de 2025", 2)["chunk_id"] == str(CHUNKS[1].id)
    assert refs.locate("marzo de 2025", 1)["page"] == 2  # wrong page -> real one
    assert refs.locate("cita inventada que no existe", 1) is None


def test_audit_parser_verifies_quotes_and_rescales_values():
    facts = [
        {"label": "Presupuesto inicial", "value": "$44.000.000", "kind": "amount",
         "quote": "presupuesto inicial fue de $44.000.000", "page": 1},
        {"label": "Escala", "value": 44, "quote": "44 millones en total", "page": 3},
        {"label": "Inventado", "value": 9, "quote": "no existe 9", "page": 1},
        {"label": "Valor ajeno", "value": 777, "quote": "$44.000.000", "page": 1},
    ]  # fmt: skip
    data = parse("audit", {"facts": facts})
    assert data["facts_count"] == 2 and data["findings"] == []
    assert parse_artifact("audit", [json.dumps({"facts": facts[1:2]})], CHUNKS, {})["checked"] == {
        "sum_groups": 0,
        "percent_groups": 0,
        "repeated_metrics": 0,
    }
    assert parse("audit", {"facts": []})["facts_count"] == 0
    assert parse("audit", "sin json") is None


def test_audit_batches_merge_into_one_check():
    a = json.dumps({"facts": [{"label": "Ingresos", "value": 44000000, "unit": "COP",
                               "quote": "$44.000.000", "page": 1}]})  # fmt: skip
    b = json.dumps({"facts": [{"label": "Ingresos", "value": 2025, "quote": "marzo de 2025",
                               "page": 2}]})  # fmt: skip
    data = parse_artifact("audit", [a, "basura", b], CHUNKS, {})
    assert data["facts_count"] == 2 and data["findings"][0]["type"] in (
        "conflicting_values",
        "scale_mismatch",
    )


def test_graph_dedupes_aliases_and_drops_unverified_edges():
    ents = [
        {"name": "Aureliano Buendía", "type": "persona", "description": "Fundador", "page": 1},
        {"name": "Aureliano", "type": "persona", "description": "El coronel", "page": 1},
        {"name": "Macondo", "type": "lugar", "page": 1, "aliases": ["el pueblo"]},
        {"name": "María Gómez", "type": "alguna", "page": 2},
    ]
    rels = [
        {"source": "Aureliano", "target": "el pueblo", "label": "fundó",
         "quote": "Aureliano Buendía fundó Macondo", "page": 1},
        {"source": "Aureliano Buendía", "target": "Macondo", "label": "fundó",
         "quote": "Aureliano Buendía fundó Macondo", "page": 1},
        {"source": "María Gómez", "target": "Macondo", "label": "vive en", "quote": "inventada"},
        {"source": "María Gómez", "target": "María Gómez", "label": "x",
         "quote": "María Gómez", "page": 2},
    ]  # fmt: skip
    graph = parse("entity_graph", {"entities": ents, "relations": rels})
    names = [e["name"] for e in graph["entities"]]
    assert names == ["Aureliano Buendía", "Macondo", "María Gómez"]
    assert graph["entities"][2]["type"] == "concepto"
    assert len(graph["relations"]) == 1  # duplicate merged, unverified + self-loop dropped
    assert graph["relations"][0] == {**graph["relations"][0], "source": "e1", "target": "e2"}


def test_graph_caps_nodes_and_edges():
    ents = [{"name": f"Entidad{i} X{i}", "type": "objeto", "description": "", "page": 1}
            for i in range(100)]  # fmt: skip
    names = [e["name"] for e in ents]
    rels = [
        {"source": names[i], "target": names[(i + 1) % 100], "label": f"r{j}", "quote": "q"}
        for i in range(100)
        for j in range(2)
    ]
    graph = build_graph(ents, rels)
    assert len(graph["entities"]) == MAX_NODES and len(graph["relations"]) <= MAX_EDGES
    ids = {e["id"] for e in graph["entities"]}
    assert all(r["source"] in ids and r["target"] in ids for r in graph["relations"])


def test_rubric_requires_verified_quote_unless_not_found():
    options = clean_options("rubric", {"preset": "plan_negocios"})
    assert options == {"preset": "plan_negocios"}
    assert clean_options("rubric", {"preset": "xx"}) == {"preset": "general"}
    items = [
        {"id": "resumen", "status": "cumple", "justification": "ok",
         "quote": "presupuesto inicial fue de", "page": 1, "suggestion": "s"},
        {"id": "mercado", "status": "cumple", "justification": "dice", "quote": "inventada"},
        {"id": "equipo", "status": "parcial", "quote": "lo lidera María Gómez", "page": 2},
    ]  # fmt: skip
    data = parse("rubric", {"criteria": items}, options)
    by_id = {c["id"]: c for c in data["criteria"]}
    assert len(data["criteria"]) == 10 and data["preset_label"] == "Plan de negocios"
    assert by_id["resumen"]["status"] == "cumple" and by_id["resumen"]["chunk_id"]
    assert by_id["mercado"]["status"] == "no_encontrado" and by_id["mercado"]["quote"] is None
    assert by_id["equipo"]["status"] == "parcial" and by_id["finanzas"]["status"] == "no_encontrado"
    assert data["summary"] == {"cumple": 1, "parcial": 1, "no_encontrado": 8}
    assert parse("rubric", {"criteria": []}, options) is None


def test_date_normalization():
    assert normalize_date("1909") == "1909"
    assert normalize_date("marzo de 2025") == "2025-03"
    assert normalize_date("15 de Marzo de 2025") == "2025-03-15"
    assert normalize_date("15/03/2025") == "2025-03-15"
    assert normalize_date("al día siguiente") is None


def test_timeline_drops_events_whose_date_is_not_in_the_quote():
    events = [
        {"date_text": "marzo de 2025", "title": "Inauguración", "page": 2,
         "quote": "En marzo de 2025 se inauguró la planta"},
        {"date_text": "1909", "title": "Falso", "quote": "En marzo de 2025 se inauguró", "page": 2},
        {"date_text": "2020", "title": "Sin cita", "quote": "inventada 2020", "page": 1},
    ]  # fmt: skip
    data = parse("timeline", {"events": events})
    assert [e["title"] for e in data["events"]] == ["Inauguración"]
    assert data["events"][0]["date_iso"] == "2025-03" and data["ordered_by"] == "date"
    empty = parse("timeline", {"events": []})
    assert empty["events"] == [] and "temporal" in empty["message"]


def test_batches_use_up_to_three_calls_for_long_documents():
    many = [Chunk(i, f"ventas {i} 1500 2021 {'x' * 800}", i + 1, i + 1, 0, 1) for i in range(120)]
    assert len(artifact_batches("audit", {}, many)) == 3
    assert len(artifact_batches("entity_graph", {}, many)) == 3
    assert len(artifact_batches("audit", {}, CHUNKS[:1])) == 1
    assert len(artifact_batches("timeline", {}, many)) == 1
    prompt = artifact_batches("rubric", {"preset": "ensayo"}, many)[0][1]["content"]
    assert "tesis" in prompt


def test_generation_runs_one_call_per_batch():
    answer = json.dumps({"facts": [{"label": "P", "value": 120, "quote": "texto 0 120",
                                    "page": 1}]})  # fmt: skip
    chat_id, artifacts, request, generate, chat = _setup(answer)
    created = asyncio.run(request.execute(chat_id, "audit", None))
    asyncio.run(generate.execute(created.id))
    stored = artifacts.items[created.id]
    assert stored.status == "ready" and stored.data["facts_count"] == 1
