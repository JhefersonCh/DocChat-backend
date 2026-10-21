"""Timeline helpers: normalize Spanish date expressions to ISO (year / month / day)."""

import re

from chat_pdf.domain.text_match import norm

MONTHS = {
    m: i + 1
    for i, m in enumerate(
        (
            "enero febrero marzo abril mayo junio julio agosto septiembre octubre "
            "noviembre diciembre"
        ).split()
    )
}
MONTHS["setiembre"] = 9
_YEAR = r"(\d{3,4})"
_ISO = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")
_PATTERNS = (
    (re.compile(rf"^(\d{{1,2}}) de (\w+) (?:de |del )?{_YEAR}$"), "dmy"),
    (re.compile(rf"^(\w+) (?:de |del )?{_YEAR}$"), "my"),
    (re.compile(r"^(\d{1,2}) (\d{1,2}) (\d{4})$"), "num"),
    (re.compile(rf"^(?:en |año )?{_YEAR}$"), "y"),
)


def _ymd(year: int, month: int | None = None, day: int | None = None) -> str | None:
    if not 1 <= year <= 2999 or (month is not None and not 1 <= month <= 12):
        return None
    if day is not None and not 1 <= day <= 31:
        return None
    out = f"{year:04d}"
    if month:
        out += f"-{month:02d}"
        if day:
            out += f"-{day:02d}"
    return out


def normalize_date(text: str) -> str | None:
    """ "1909" -> "1909", "marzo de 2025" -> "2025-03", "15 de marzo de 2025" -> "2025-03-15",
    "15/03/2025" -> "2025-03-15"; None when it is not a calendar date."""
    raw = text.strip()
    if _ISO.match(raw):
        return raw
    folded = norm(raw)
    for pattern, kind in _PATTERNS:
        m = pattern.match(folded)
        if not m:
            continue
        if kind == "dmy":
            month = MONTHS.get(m.group(2))
            return _ymd(int(m.group(3)), month, int(m.group(1))) if month else None
        if kind == "my":
            month = MONTHS.get(m.group(1))
            return _ymd(int(m.group(2)), month) if month else None
        if kind == "num":
            return _ymd(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        return _ymd(int(m.group(1)))
    return None


def valid_iso(value: object) -> str | None:
    return value if isinstance(value, str) and _ISO.match(value.strip()) else None
