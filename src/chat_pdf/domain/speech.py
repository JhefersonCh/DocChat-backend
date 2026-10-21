"""Text -> speakable text. Mirrored by `frontend/src/domain/speech.ts` (same test vectors)."""

import re
import unicodedata

MAX_SPEAK_CHARS = 1200

_FENCE = re.compile(r"```.*?(```|$)", re.DOTALL)
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_CITATION = re.compile(r"\s*\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_LIST_MARK = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+", re.MULTILINE)
_QUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_RULE = re.compile(r"^\s*(?:[-*_]\s*){3,}$", re.MULTILINE)
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$", re.MULTILINE)
_EMPHASIS = re.compile(
    r"(?:(\*\*|\*|~~)(?=\S)(.+?)(?<=\S)\1|(?<!\w)(__|_)(?=\S)(.+?)(?<=\S)\3(?!\w))", re.DOTALL
)
_ABBREVIATIONS = (
    (re.compile(r"\bpágs\.", re.IGNORECASE), "páginas"),
    (re.compile(r"\bpág\.", re.IGNORECASE), "página"),
    (re.compile(r"\bSra\."), "Señora"),
    (re.compile(r"\bSr\."), "Señor"),
    (re.compile(r"\bDra\."), "Doctora"),
    (re.compile(r"\bDr\."), "Doctor"),
)


# Soft hyphens, zero-width marks, emoji and symbols: TTS engines reject them outright.
_UNSPEAKABLE = re.compile(r"[\u00ad\u200b-\u200f\u2028-\u202e\u2060-\u206f\ufeff\ufe0f]")


def _drop_unspeakable(text: str) -> str:
    text = _UNSPEAKABLE.sub("", unicodedata.normalize("NFC", text))
    return "".join(
        c for c in text if unicodedata.category(c) not in ("So", "Cs", "Co", "Cn") and c != "\x00"
    )


def strip_for_speech(text: str) -> str:
    """Remove markdown, `[n]` citation markers and URLs; expand a few abbreviations."""
    t = _FENCE.sub(" ", _drop_unspeakable(text))
    t = _IMAGE.sub(r"\1", t)
    t = _LINK.sub(r"\1", t)
    t = _CITATION.sub("", t)
    t = re.sub(r"https?://\S+", "", t)
    t = _TABLE_SEP.sub("", t)
    t = _RULE.sub("", t)
    t = _HEADING.sub("", t)
    t = _QUOTE.sub("", t)
    t = _LIST_MARK.sub("", t)
    for _ in range(2):  # nested emphasis: **_x_**
        t = _EMPHASIS.sub(lambda m: m.group(2) or m.group(4), t)
    t = re.sub(r"[`*~#]", "", t).replace("|", " ")  # leftovers of emphasis split across sentences
    for pattern, replacement in _ABBREVIATIONS:
        t = pattern.sub(replacement, t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()
