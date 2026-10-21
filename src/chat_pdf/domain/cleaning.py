"""Extraction cleanup: layout artefacts that hurt retrieval (pure, no IO)."""

import re
from collections import Counter
from dataclasses import replace

from chat_pdf.domain.models import Page

BOILERPLATE_SHARE = 0.5  # a line on at least this share of the pages is a running header/footer
BOILERPLATE_MIN_PAGES = 3  # below this, repetition says nothing

# "S E C C I Ó N 0 9": 3+ spaced capitals/digits (lowercase would eat words like "y", "a", "o")
_LETTER_SPACED = re.compile(r"(?<!\S)[A-ZÀ-ÖØ-Þ0-9](?: [A-ZÀ-ÖØ-Þ0-9]){2,}(?!\S)")
_HYPHENATED = re.compile(r"(?<=[^\W\d_])-[ \t]*\n\s*(?=[^\W\d_])")
_LETTER_DIGIT = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")


def _line_key(line: str, page: Page) -> str:
    """Comparable form of a line: the page's own number masked ("Página 3" ~ "Página 4")."""
    key = " ".join(line.split()).casefold()
    return key if page.number is None else re.sub(rf"(?<!\d){page.number}(?!\d)", "#", key)


def _is_page_number(line: str, page: Page) -> bool:
    return page.number is not None and line.strip() == str(page.number)


def strip_boilerplate(pages: list[Page]) -> list[Page]:
    """Drop running headers/footers and standalone page numbers; other lines stay verbatim.

    A header/footer is a line with letters repeated (its page number masked) on at least
    BOILERPLATE_SHARE of the pages. A page number is a line equal to the page's number at the
    page's top or bottom (elsewhere a bare number is content, e.g. a list item).
    """
    seen = Counter(
        key
        for page in pages
        for key in {_line_key(ln, page) for ln in page.text.splitlines() if _has_letters(ln)}
    )
    threshold = max(BOILERPLATE_MIN_PAGES, BOILERPLATE_SHARE * len(pages))
    repeated = {key for key, n in seen.items() if n >= threshold}

    cleaned = []
    for page in pages:
        lines = [ln for ln in page.text.splitlines() if _line_key(ln, page) not in repeated]
        filled = [i for i, line in enumerate(lines) if line.strip()]
        edges = {filled[0], filled[-1]} if filled else set()
        lines = [
            line for i, line in enumerate(lines) if not (i in edges and _is_page_number(line, page))
        ]
        text = "\n".join(lines)
        cleaned.append(page if text == page.text else replace(page, text=text))
    return cleaned


def _has_letters(line: str) -> bool:
    return any(ch.isalpha() for ch in line)


def collapse_letter_spacing(text: str) -> str:
    """'S E C C I Ó N 0 9' -> 'SECCIÓN 09'. Word gaps inside the run are lost ('RAÍCESDEL')."""

    def join(match: re.Match[str]) -> str:
        return _LETTER_DIGIT.sub(" ", match.group().replace(" ", ""))

    return _LETTER_SPACED.sub(join, text)


def is_letter_spaced(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and _LETTER_SPACED.fullmatch(stripped) is not None


def normalize(text: str) -> str:
    """Text to embed and full-text index: letter-spacing and hyphenation undone, one space."""
    text = _HYPHENATED.sub("", text)
    return " ".join(collapse_letter_spacing(text).split())
