"""Where a segment sits: page region (bbox) and time range. Pure value rules, no I/O."""

from chat_pdf.domain.errors import InvalidInput

# Thirds of the page: (row, column) -> how a person would say it.
_REGIONS = {
    (0, 0): "arriba a la izquierda",
    (0, 1): "arriba",
    (0, 2): "arriba a la derecha",
    (1, 0): "a la izquierda",
    (1, 1): "centro",
    (1, 2): "a la derecha",
    (2, 0): "abajo a la izquierda",
    (2, 1): "abajo",
    (2, 2): "abajo a la derecha",
}
WHOLE_PAGE_COVERAGE = 0.9  # a bbox this wide and tall is "the whole thing"


def validate_bbox(bbox: list[float]) -> list[float]:
    """`[x0, y0, x1, y1]` normalized to 0-1 with x0 < x1 and y0 < y1."""
    if len(bbox) != 4:
        raise InvalidInput("invalid_bbox", "bbox necesita 4 valores [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise InvalidInput("invalid_bbox", "bbox debe estar normalizado (0-1) y no ser vacío")
    return [x0, y0, x1, y1]


def validate_range(t_start: float, t_end: float) -> tuple[float, float]:
    if t_start < 0 or t_end <= t_start:
        raise InvalidInput("invalid_time_range", "El rango de tiempo debe ser 0 <= inicio < fin")
    return t_start, t_end


def region_label(bbox: list[float]) -> str:
    """Spanish description of the third of the page that holds the bbox's center."""
    x0, y0, x1, y1 = validate_bbox(bbox)
    if x1 - x0 >= WHOLE_PAGE_COVERAGE and y1 - y0 >= WHOLE_PAGE_COVERAGE:
        return "toda la página"
    column = min(2, int((x0 + x1) / 2 * 3))
    row = min(2, int((y0 + y1) / 2 * 3))
    return _REGIONS[(row, column)]


def format_time(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def format_range(t_start: float, t_end: float) -> str:
    return f"{format_time(t_start)}–{format_time(t_end)}"


def format_locator(
    page: int | None = None,
    bbox: list[float] | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
) -> str:
    """Human locator: "pág. 12, arriba a la izquierda" or "03:20–03:40" ("" when unknown)."""
    parts = []
    if page is not None:
        parts.append(f"pág. {page}")
        if bbox is not None:
            parts.append(region_label(bbox))
    if t_start is not None:
        parts.append(format_range(t_start, t_end) if t_end is not None else format_time(t_start))
    return ", ".join(parts)
