"""Loose text matching: lets a verbatim `quote` be checked against chunk text.

Whitespace, case, accents and punctuation are ignored (PDF extraction mangles all of them).
"""

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def norm(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).casefold()
    stripped = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return _NON_ALNUM.sub(" ", stripped).strip()


def quote_in(quote: str, text: str) -> bool:
    """True when the (normalized, non-empty) `quote` appears inside `text`."""
    needle = norm(quote)
    return bool(needle) and f" {needle} " in f" {norm(text)} "
