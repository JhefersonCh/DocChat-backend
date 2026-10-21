"""Claim check: split an answer into sentences, parse the model's labels, sanity-check them."""

import re
import unicodedata
from typing import Any

from chat_pdf.domain.generation import extract_json
from chat_pdf.domain.models import Citation
from chat_pdf.domain.prompting import strip_citations
from chat_pdf.domain.prompts import (
    VERIFY_MEDIA_NOTE,
    VERIFY_SOURCE,
    VERIFY_SYSTEM,
    VERIFY_USER_TEMPLATE,
)

STATUSES = ("respaldada", "parcial", "sin_respaldo")
MIN_OVERLAP = 0.15  # below this share of the claim's content words, "respaldada" is not credible
MIN_WORD_CHARS = 4  # shorter words (articles, prepositions, "que") carry no evidence
MAX_SOURCE_CHARS = 1500
LOW_OVERLAP_NOTE = "(Poca coincidencia de palabras con la fuente.)"
MAX_CLAIMS = 40
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[\wÁÉÍÓÚÜÑ¿¡«\"(])")
_MARKUP = re.compile(r"[*_`#>]+|^\s*(?:[-•]|\d+[.)])\s+", re.MULTILINE)
_WORD = re.compile(r"[a-z0-9]+")


def split_sentences(answer: str) -> list[str]:
    """Sentences of the answer without citation markers or markdown marks (one per list item)."""
    plain = _MARKUP.sub("", strip_citations(answer))
    sentences: list[str] = []
    for line in plain.splitlines():
        sentences += [" ".join(s.split()) for s in _SENTENCE_END.split(line)]
    return [s for s in sentences if s]


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _flat(text: str) -> str:
    return _fold(" ".join(text.split()))


def _stem(word: str) -> str:
    return word[:5]  # crude: "entrega"/"entregas"/"entregar" meet


def content_words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(_fold(text)) if len(w) >= MIN_WORD_CHARS}


def lexical_overlap(claim: str, source: str) -> float:
    """Share of the claim's content words that also appear in the source (0 when it has none)."""
    words = content_words(claim)
    return len(words & content_words(source)) / len(words) if words else 0.0


def verify_messages(sentences: list[str], sources: list[Citation]) -> list[dict[str, str]]:
    blocks = "\n\n".join(
        VERIFY_SOURCE.format(
            n=c.n,
            content=(
                VERIFY_MEDIA_NOTE.get(c.kind, "") + (c.content or c.snippet)[:MAX_SOURCE_CHARS]
            ),
        )
        for c in sources
    )
    listed = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    user = VERIFY_USER_TEMPLATE.format(sentences=listed, sources=blocks or "(sin fuentes)")
    return [{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": user}]


def _claim(raw: Any, sentences: int, sources: dict[int, Citation]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    text = raw.get("text")
    status = raw.get("status")
    if not isinstance(text, str) or not text.strip() or status not in STATUSES:
        return None
    index = raw.get("sentence_index")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < sentences:
        return None
    ns = raw.get("source_ns")
    given = ns if isinstance(ns, list) else []
    source_ns = sorted({n for n in given if isinstance(n, int) and n in sources})
    quote, reason = raw.get("quote"), raw.get("reason")
    claim: dict[str, Any] = {
        "text": " ".join(text.split()),
        "sentence_index": index,
        "status": status,
        "source_ns": source_ns,
        "reason": " ".join(reason.split()) if isinstance(reason, str) else "",
    }
    if isinstance(quote, str) and quote.strip():
        claim["quote"] = " ".join(quote.split())
    return claim


def parse_claims(text: str, sentences: int, sources: list[Citation]) -> list[dict[str, Any]] | None:
    """Valid claims of the model's JSON (invalid items dropped); None if it is not a claim list."""
    data = extract_json(text)
    if isinstance(data, dict):
        data = data.get("claims")
    if not isinstance(data, list):
        return None
    by_n = {c.n: c for c in sources}
    claims = [_claim(item, sentences, by_n) for item in data[:MAX_CLAIMS]]
    return [c for c in claims if c]


def check_claims(claims: list[dict[str, Any]], sources: list[Citation]) -> list[dict[str, Any]]:
    """Small models are lenient: a claim with almost no words in common with its sources cannot
    be "respaldada" (downgraded to "parcial"). Also: a quote must come from a cited source."""
    by_n = {c.n: (c.content or c.snippet) for c in sources}
    checked = []
    for claim in claims:
        ns = claim["source_ns"] or list(by_n)
        evidence = " ".join(by_n[n] for n in ns if n in by_n)
        claim = dict(claim)
        quote = claim.get("quote")
        if quote and _flat(quote) not in _flat(evidence):
            del claim["quote"]  # not verbatim: do not show an invented quote
        overlap = lexical_overlap(claim["text"], evidence)
        if claim["status"] == "respaldada" and overlap < MIN_OVERLAP:
            claim["status"] = "parcial"
            claim["reason"] = f"{claim['reason']} {LOW_OVERLAP_NOTE}".strip()
        checked.append(claim)
    return checked
