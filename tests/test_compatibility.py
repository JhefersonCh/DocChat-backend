"""Source compatibility: per-source status from sampled chunk vectors (pure domain)."""

from uuid import uuid4

from chat_pdf.domain.compatibility import CompatItem, analyze, classify, similarity, topic_vectors

P1, P2 = uuid4(), uuid4()


def item(alias, *vectors, profile=P1, sha=None):
    return CompatItem(uuid4(), alias, f"{alias}.txt", profile, tuple(vectors), 1, None, sha)


def test_classify_bands_and_pair_score_is_symmetric():
    assert (classify(0.79), classify(0.71), classify(0.60)) == ("ok", "weak", "unrelated")
    a, b = ([1.0, 0.0], [0.0, 1.0]), ([1.0, 0.1], [0.0, 1.0], [0.5, 0.5])
    assert similarity(a, b) == similarity(b, a)
    assert topic_vectors([("text", [1.0]), ("image", [2.0])]) == ([1.0], [2.0])  # < 2 text chunks
    assert topic_vectors([("text", [1.0]), ("text", [3.0]), ("image", [2.0])]) == ([1.0], [3.0])


def test_unrelated_source_breaks_coherence():
    a, b = item("S1", [1.0, 0.6, 0.0]), item("S2", [1.0, 0.0, 0.3])
    c = item("S3", [0.0, 0.0, 1.0])
    result = analyze([a, b, c])
    by = {s.alias: s for s in result.sources}
    assert not result.coherent
    assert [by[x].status for x in ("S1", "S2", "S3")] == ["ok", "ok", "unrelated"]
    assert by["S1"].best_match.alias == "S2"


def test_clusters_of_unrelated_sources_and_copies_do_not_vouch_for_each_other():
    a, b = item("S1", [1.0, 0.0, 0.0]), item("S2", [0.0, 1.0, 0.0])
    twin1, twin2 = item("S3", [0.0, 0.0, 1.0], sha="x"), item("S4", [0.0, 0.1, 1.0], sha="x")
    result = analyze([a, b, twin1, twin2])
    by = {s.alias: s for s in result.sources}
    assert not result.coherent
    assert (by["S1"].status, by["S2"].status) == ("unrelated", "unrelated")
    assert by["S3"].status == by["S4"].status == "duplicate"
    assert by["S3"].best_match.alias == "S4" and "S4" in by["S3"].reason
    only_copies = analyze([item("S1", [1.0, 0.0], sha="x"), item("S2", [0.0, 1.0], sha="x")])
    assert only_copies.coherent  # copies of one source: nothing to be unrelated to


def test_other_profile_or_not_indexed_is_unknown_and_single_source_is_ok():
    a, b = item("S1", [1.0, 0.6]), item("S2", [1.0, 0.0])
    other, pending = item("S3", [1.0, 0.0, 0.0], profile=P2), item("S4")
    by = {s.alias: s for s in analyze([a, b, other, pending]).sources}
    assert (by["S1"].status, by["S3"].status, by["S4"].status) == ("ok", "unknown", "unknown")
    assert by["S3"].reason
    solo = analyze([a])
    assert solo.coherent and solo.sources[0].status == "ok"
