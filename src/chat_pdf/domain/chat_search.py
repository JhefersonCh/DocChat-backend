"""Keyword search over chats: query parsing, accent/case folding, snippets and ranking.

Not natural language: every term is a case- and accent-insensitive substring ("presu" finds
"Presupuesto"), several terms are ANDed and a "quoted phrase" must appear exactly as written.
"""

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime

MAX_TERMS = 8
SNIPPET_CHARS = 160
_TOKEN = re.compile(r'"([^"]+)"|(\S+)')
_CITATION = re.compile(r"\s?\[\s*S?\d+(?:\s*[,;]\s*S?\d+)*\s*\]")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_FENCE = re.compile(r"```[a-zA-Z0-9_-]*")
_MD_LINE = re.compile(r"(?m)^\s*(?:#{1,6}\s+|>\s?|[-*+]\s+|\d+[.)]\s+)")
_MD_MARKS = re.compile(r"[*_`~]{1,3}")
_SPACES = re.compile(r"\s+")


def _fold_char(ch: str) -> str:
    stripped = "".join(c for c in unicodedata.normalize("NFD", ch) if not unicodedata.combining(c))
    folded = stripped.lower()
    if len(folded) == 1:
        return folded
    return ch.lower() if len(ch.lower()) == 1 else ch


def fold(text: str) -> str:
    """Lower-case without accents, always the same length as `text` (offsets stay valid)."""
    return "".join(_fold_char(ch) for ch in text)


def parse_query(q: str) -> list[str]:
    """Folded terms: a quoted phrase is one term, the rest are split on whitespace."""
    terms: list[str] = []
    for match in _TOKEN.finditer(q):
        raw = match.group(1) or match.group(2)
        term = _SPACES.sub(" ", fold(raw)).strip().strip('"')
        if term and term not in terms:
            terms.append(term)
    return terms[:MAX_TERMS]


def like_pattern(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def clean_text(text: str) -> str:
    """Plain, single-line text for a snippet: no [n] citations, no markdown syntax."""
    text = _FENCE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _CITATION.sub("", text)
    text = _MD_LINE.sub("", text)
    text = _MD_MARKS.sub("", text)
    return _SPACES.sub(" ", text).strip()


def find_highlights(text: str, terms: list[str]) -> list[tuple[int, int]]:
    """Merged [start, end) ranges of every occurrence of every term in `text`."""
    folded = fold(text)
    spans: list[tuple[int, int]] = []
    for term in terms:
        start = folded.find(term)
        while start != -1:
            spans.append((start, start + len(term)))
            start = folded.find(term, start + 1)
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


@dataclass(frozen=True)
class Snippet:
    text: str
    highlights: list[tuple[int, int]]


def _utf16(text: str, i: int) -> int:
    """Code-point offset -> UTF-16 unit offset (what JavaScript string indexes use)."""
    return len(text[:i].encode("utf-16-le")) // 2


def make_snippet(content: str, terms: list[str], size: int = SNIPPET_CHARS) -> Snippet:
    """~`size` chars around the first match, with the highlight offsets relative to the snippet."""
    text = clean_text(content)
    marks = find_highlights(text, terms)
    if len(text) <= size:
        return Snippet(text, [(_utf16(text, s), _utf16(text, e)) for s, e in marks])
    first = marks[0][0] if marks else 0
    start = max(0, first - size // 3)
    if start:  # do not begin mid-word
        space = text.find(" ", start - 12, start + 12)
        start = space + 1 if space != -1 else start
    end = min(len(text), start + size)
    if end < len(text):
        space = text.rfind(" ", end - 12, end + 1)
        end = space if space > start else end
    segment = text[start:end]
    body = segment.strip()
    lo = start + len(segment) - len(segment.lstrip())
    hi = lo + len(body)
    lead = "… " if lo > 0 else ""
    shift = len(lead) - lo
    out = lead + body + (" …" if hi < len(text) else "")
    shifted = [(max(s, lo) + shift, min(e, hi) + shift) for s, e in marks if e > lo and s < hi]
    return Snippet(out, [(_utf16(out, s), _utf16(out, e)) for s, e in shifted if e > s])


def _word_bonus(folded: str, term: str) -> float:
    if re.search(rf"(?<![0-9a-z]){re.escape(term)}(?![0-9a-z])", folded):
        return 2.0  # whole word
    if re.search(rf"(?<![0-9a-z]){re.escape(term)}", folded):
        return 1.0  # start of a word
    return 0.0  # inside a word


def score(
    content: str, terms: list[str], created_at: datetime, now: datetime | None = None
) -> float:
    """Match quality (whole word > word prefix > inside; phrases count more) plus recency."""
    folded = fold(clean_text(content))
    quality = 0.0
    for term in terms:
        quality += 1.0 + _word_bonus(folded, term) + (1.0 if " " in term else 0.0)
    now = now or datetime.now(UTC)
    age_days = max(0.0, (now - created_at).total_seconds() / 86400)
    return quality + 3.0 * math.exp(-age_days / 45)
