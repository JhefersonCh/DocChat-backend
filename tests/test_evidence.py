from uuid import uuid4

from chat_pdf.domain.models import Chat, Chunk, Source
from chat_pdf.domain.prompting import build_answer_messages, style_instructions
from chat_pdf.domain.retrieval import is_weak_evidence


def _source(n, similarity):
    return Source(n, Chunk(n, f"t{n}", 1, 1, 0, 1, similarity=similarity), uuid4(), "x.pdf")


def test_weak_evidence_when_best_similarity_is_below_threshold():
    sources = [_source(1, 0.2), _source(2, 0.3)]

    assert is_weak_evidence(sources, 0.4)
    assert not is_weak_evidence([*sources, _source(3, 0.5)], 0.4)


def test_unknown_similarity_is_never_weak():
    assert not is_weak_evidence([_source(1, None)], 0.4)
    assert not is_weak_evidence([_source(1, 0.1), _source(2, None)], 0.4)


def test_citation_exposes_calibrated_relevance_score():
    # (cos - floor) / (1 - floor) with the text floor 0.50, clamped to [0, 1]
    assert _source(1, 0.87654).cite(50).score == 0.753
    assert _source(1, 1.0).cite(50).score == 1.0
    assert _source(1, -0.2).cite(50).score == 0.0
    assert _source(1, None).cite(50).score is None


def test_style_and_length_instructions_reach_the_system_prompt():
    chat = Chat(
        uuid4(), "", None, style="custom", custom_instructions="Usá viñetas", length="corta"
    )

    system = build_answer_messages([], "q", [_source(1, 0.9)], style_instructions(chat))[0]

    assert "Usá viñetas" in system["content"]
    assert "breve" in system["content"].lower()


def test_default_chat_adds_no_style_instructions():
    assert style_instructions(Chat(uuid4(), "", None)) == ""


def test_weak_evidence_adds_cautious_instruction():
    system = build_answer_messages([], "q", [_source(1, 0.1)], weak=True)[0]["content"]

    assert "evidencia" in system.lower() and "débil" in system.lower()
