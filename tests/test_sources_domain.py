"""Phase 7 (multi-source), pure parts: aliases, readable sources, filters, balance, titles."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from chat_pdf.domain.models import Chunk, Document, Source, SourceRef
from chat_pdf.domain.prompting import find_all_tool, format_source, search_tool
from chat_pdf.domain.retrieval import SourceSet
from chat_pdf.domain.sources import (
    CROSS_QUESTION,
    assign_aliases,
    balance_sources,
    chat_title,
    collection_overview,
    parse_source_filter,
    readable_sources,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _doc(i, name="x.pdf", status="ready", kind="pdf", summary=None, questions=()):
    return Document(
        uuid4(), uuid4(), name, "application/pdf", status, None, None, uuid4(),
        T0 + timedelta(minutes=i), summary=summary, suggested_questions=list(questions), kind=kind,
    )  # fmt: skip


def _chunk(doc, ordinal=0, text="t"):
    return Chunk(ordinal, text, 1, 1, 0, 1, id=uuid4(), document_id=doc.id)


def test_aliases_follow_the_order_sources_were_added():
    a, b, c = _doc(2, "b"), _doc(1, "a"), _doc(3, "c")

    assert assign_aliases([a, b, c]) == {b.id: "S1", a.id: "S2", c.id: "S3"}


def test_chat_title_is_the_name_of_one_source_or_the_count():
    assert chat_title([]) == ""
    assert chat_title(["informe.pdf"]) == "informe.pdf"
    assert chat_title(["a.pdf", "b.mp4", "c.mp3"]) == "3 fuentes"


def test_readable_sources_skip_muted_and_unready_and_per_ask_selection_wins():
    a, b, c = _doc(1), _doc(2, status="processing"), _doc(3)

    assert readable_sources([a, b, c]) == [a, c]
    assert readable_sources([a, b, c], disabled={a.id}) == [c]
    assert readable_sources([a, b, c], disabled={a.id}, requested={a.id}) == [a]
    assert readable_sources([a, b, c], requested={b.id}) == []  # unready never readable


def test_source_filter_parses_aliases_leniently():
    ids = {"S1": uuid4(), "S2": uuid4(), "S3": uuid4()}

    assert parse_source_filter(None, ids) is None
    assert parse_source_filter("all", ids) is None
    assert parse_source_filter("ALL", ids) is None
    assert parse_source_filter("s2", ids) == [ids["S2"]]
    assert parse_source_filter("S1, S3", ids) == [ids["S1"], ids["S3"]]
    assert parse_source_filter(["S3"], ids) == [ids["S3"]]
    assert parse_source_filter("S9", ids) == []  # only unknown aliases


def test_balance_keeps_a_dominant_source_from_starving_the_other():
    a, b = _doc(1), _doc(2)
    ranked = [_chunk(a, i) for i in range(6)]
    ranked.insert(3, _chunk(b, 0))  # the best chunk of B is only 4th overall

    top = balance_sources(ranked, 3)

    assert len(top) == 3 and ranked[3] in top  # B's best is guaranteed a slot
    assert top == [c for c in ranked if c in top]  # ranking order preserved


def test_balance_with_one_source_is_a_plain_top_k():
    a = _doc(1)
    ranked = [_chunk(a, i) for i in range(5)]

    assert balance_sources(ranked, 2) == ranked[:2]
    assert balance_sources([], 3) == []


def test_source_set_routes_chunks_to_their_source_and_labels_them_when_several():
    a, b = _doc(1, "informe.pdf"), _doc(2, "clip.mp4", kind="video")
    refs = {d.id: SourceRef(d.id, f"S{i}", d.filename, d.kind) for i, d in enumerate([a, b], 1)}
    chunks = SourceSet(refs, labeled=True).add([_chunk(a), _chunk(b)])

    assert [(s.alias, s.filename, s.source_kind) for s in chunks] == [
        ("S1", "informe.pdf", "pdf"),
        ("S2", "clip.mp4", "video"),
    ]
    assert chunks[1].cite(100).source_alias == "S2"
    assert format_source(chunks[1]).startswith("[2] (S2 · clip.mp4)")


def test_single_source_prompt_is_unlabeled():
    a = _doc(1, "informe.pdf")
    refs = {a.id: SourceRef(a.id, "S1", a.filename, a.kind)}
    [source] = SourceSet(refs, labeled=False).add([_chunk(a, text="hola")])

    assert source.alias == "S1"
    assert format_source(source) == "[1] (pág. 1) hola"


def test_legacy_add_still_takes_the_document_explicitly():
    a = _doc(1)
    [source] = SourceSet().add([Chunk(0, "x", 1, 1, 0, 1, id=uuid4())], a.id, "a.pdf", "pdf")

    assert isinstance(source, Source) and source.document_id == a.id and source.alias is None


def test_search_tool_offers_source_only_when_there_is_something_to_choose():
    assert "source" not in search_tool(10)["function"]["parameters"]["properties"]
    assert "source" not in search_tool(10, aliases=["S1"])["function"]["parameters"]["properties"]
    props = search_tool(10, aliases=["S1", "S2"])["function"]["parameters"]["properties"]
    assert props["source"]["enum"] == ["all", "S1", "S2"]
    assert find_all_tool(["S1", "S2"])["function"]["parameters"]["properties"]["source"]["enum"]


def test_collection_overview_aggregates_summaries_and_ends_with_a_cross_question():
    a = _doc(1, "a.pdf", summary="Habla de perros.", questions=["¿Qué perros?", "¿Cuántos?"])
    b = _doc(2, "b.mp3", kind="audio", summary="Una charla.", questions=["¿Quién habla?"])

    overview = collection_overview([b, a])

    assert collection_overview([a]) is None
    assert [s.alias for s in overview.sources] == ["S1", "S2"]
    assert overview.summary == "S1 · a.pdf: Habla de perros.\n\nS2 · b.mp3: Una charla."
    assert overview.suggested_questions == [
        "¿Qué perros?", "¿Quién habla?", "¿Cuántos?", CROSS_QUESTION
    ]  # fmt: skip
