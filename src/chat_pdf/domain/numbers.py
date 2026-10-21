"""Spanish-aware number parsing: "$44.000.000", "44 millones", "12,5 %", "1.2M", "(1.200)"."""

import re

_SCALES = {
    "mil millones": 1e9,
    "millones": 1e6,
    "millon": 1e6,
    "millón": 1e6,
    "mm": 1e6,
    "miles": 1e3,
    "mil": 1e3,
    "k": 1e3,
}
_NUMBER = re.compile(
    r"(?<![\w.,])(\()?(-)?\s*(\d+(?:[.,]\d+)*)"
    r"(?:\s*(?i:(mil millones|millones|millón|millon|miles|mil|mm))\b|([MKk])\b)?"
    r"(\s*%)?"
)


def _plain(raw: str, has_scale: bool) -> float | None:
    """Digits with `.`/`,` separators -> float. A lone `,` is decimal (Spanish); a lone `.`
    followed by exactly 3 digits is thousands unless a scale word follows ("1.500" vs "1.5M")."""
    dots, commas = raw.count("."), raw.count(",")
    if dots and commas:
        decimal = "." if raw.rfind(".") > raw.rfind(",") else ","
        thousands = "," if decimal == "." else "."
        raw = raw.replace(thousands, "").replace(decimal, ".")
    elif dots > 1:
        raw = raw.replace(".", "")
    elif commas > 1:
        raw = raw.replace(",", "")
    elif dots == 1:
        head, tail = raw.split(".")
        if len(tail) == 3 and head != "0" and not has_scale:
            raw = head + tail
    elif commas == 1:
        raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def find_numbers(text: str) -> list[tuple[float, float, str]]:
    """Every number in `text` as (value, mantissa, unit): value is scaled ("44 millones" ->
    44e6), mantissa is the number as written (44), unit is "%" or "". Parenthesised or
    "-" numbers are negative."""
    found = []
    for m in _NUMBER.finditer(text):
        paren, minus, raw, word, letter, pct = m.groups()
        word = (word or letter or "").lower()
        scale = 1e6 if letter == "M" else _SCALES.get(word, 1.0)
        mantissa = _plain(raw, bool(word))
        if mantissa is None:
            continue
        sign = -1 if (minus or (paren and text[m.end() : m.end() + 1] == ")")) else 1
        found.append((sign * mantissa * scale, sign * mantissa, "%" if pct else ""))
    return found


def parse_number(text: str) -> float | None:
    """The single number in a short string ("$44.000.000" -> 44000000.0), else None."""
    nums = find_numbers(text)
    return nums[0][0] if len(nums) == 1 else None
