"""TextSplitter: structure-aware recursive chunking with page ranges and char offsets (pure)."""

import re
from bisect import bisect_right
from dataclasses import dataclass

from chat_pdf.domain.cleaning import is_letter_spaced, normalize
from chat_pdf.domain.config import RagConfig
from chat_pdf.domain.models import Chunk, Page

PAGE_SEPARATOR = "\n\n"
KICKER_MAX_CHARS = 40  # "S E C C I Ó N 0 9" right above a heading belongs to it
HEADING_MAX_CHARS = 80

# Split preference inside a section: paragraphs -> sentences -> lines -> words.
_SEPARATORS = [
    re.compile(r"\n[ \t]*\n\s*"),
    re.compile(r"(?<=[.!?…])\s+"),
    re.compile(r"\n\s*"),
    re.compile(r"\s+"),
]
_MARKDOWN_HEADING = re.compile(r"#{1,6}\s+(?P<title>\S.*)")
_NUMBERED_HEADING = re.compile(r"\d{1,2}\.(?:\d{1,2}\.?)*\s+[¿¡\"“]?[A-ZÁÉÍÓÚÑ]\S*(?:\s+\S+){0,9}")
# Chapter markers: a bare roman numeral ("II", "IV.") or "CAPÍTULO 3" / "Capítulo IV: Métodos".
_CHAPTER = re.compile(
    r"[IVXLC]{1,8}\.?|cap[ií]tulo\s+(?:\d{1,3}|[IVXLC]{1,8})\.?(?:\s*[:.\-–—]\s*\S.*)?",
    re.IGNORECASE,
)
_ROMAN = re.compile(r"[IVXLC]{1,8}\.?")
CHAPTER_TITLE_MAX_WORDS = 10


@dataclass(frozen=True)
class _Line:
    start: int
    end: int
    text: str
    page: int  # index into the page list


@dataclass(frozen=True)
class _Section:
    start: int
    title: str | None


def chunk_pages(pages: list[Page], cfg: RagConfig) -> list[Chunk]:
    """Split pages into chunks that never cross a heading, cut mid-word or exceed `chunk_size`.

    Offsets index the pages joined by PAGE_SEPARATOR; `content` is that exact span (verbatim) and
    `index_text` its normalized form. Each chunk carries the title of the heading above it.
    """
    text, page_starts = _join(pages)

    def page_at(offset: int) -> int | None:
        return pages[bisect_right(page_starts, offset) - 1].number

    sections = _sections(pages, text, page_starts)
    bounds = [s.start for s in sections[1:]] + [len(text)]
    chunks: list[Chunk] = []
    for section, end in zip(sections, bounds, strict=True):
        atoms = _atoms(text, section.start, end, cfg.chunk_size, 0)
        for start, stop in _merge(atoms, cfg.chunk_size, cfg.chunk_overlap):
            raw = text[start:stop]
            start, stop = start + len(raw) - len(raw.lstrip()), start + len(raw.rstrip())
            if stop <= start:
                continue
            content = text[start:stop]
            chunks.append(
                Chunk(
                    ordinal=len(chunks),
                    content=content,
                    page_start=page_at(start),
                    page_end=page_at(stop - 1),
                    char_start=start,
                    char_end=stop,
                    section=section.title,
                    index_text=normalize(content),
                )
            )
    return chunks


def _join(pages: list[Page]) -> tuple[str, list[int]]:
    starts, parts, offset = [], [], 0
    for page in pages:
        starts.append(offset)
        parts.append(page.text)
        offset += len(page.text) + len(PAGE_SEPARATOR)
    return PAGE_SEPARATOR.join(parts), starts


def _lines(pages: list[Page], text: str, page_starts: list[int]) -> list[_Line]:
    lines = []
    for index, page_start in enumerate(page_starts):
        pos = page_start
        for raw in pages[index].text.split("\n"):
            if raw.strip():
                lines.append(_Line(pos, pos + len(raw), raw.strip(), index))
            pos += len(raw) + 1
    return lines


def _heading_title(line: _Line, pages: list[Page], use_layout: bool) -> str | None:
    if use_layout:
        headings = pages[line.page].headings or ()
        key = " ".join(line.text.split())
        return key if key in {" ".join(h.split()) for h in headings} else None
    if markdown := _MARKDOWN_HEADING.fullmatch(line.text):
        return markdown.group("title").strip()
    if (
        len(line.text) <= HEADING_MAX_CHARS
        and line.text[-1] not in ".,;:"
        and _NUMBERED_HEADING.fullmatch(line.text)
    ):
        return line.text
    return None


def _is_chapter(text: str) -> bool:
    return len(text) <= HEADING_MAX_CHARS and _CHAPTER.fullmatch(text) is not None


def _chapter_title(lines: list[_Line], i: int) -> tuple[str, set[int]] | None:
    """Title of the chapter whose marker is lines[i], plus the line indexes it consumed.

    A bare numeral takes the short unpunctuated line after it as its name ("II" + "Métodos" ->
    "II. Métodos"); "CAPÍTULO 3" already is a title.
    """
    marker = lines[i].text
    if not _ROMAN.fullmatch(marker):
        return normalize(marker), set()
    nxt = lines[i + 1].text if i + 1 < len(lines) else ""
    if (
        nxt
        and len(nxt) <= HEADING_MAX_CHARS
        and len(nxt.split()) <= CHAPTER_TITLE_MAX_WORDS
        and nxt[-1] not in ".,;:"
        and nxt[0].isupper()
        and not _is_chapter(nxt)
    ):
        return normalize(f"{marker.rstrip('.')}. {nxt}"), {i + 1}
    return normalize(marker), set()


def _sections(pages: list[Page], text: str, page_starts: list[int]) -> list[_Section]:
    """Section starts. Layout headings win when the loader found any; else text rules."""
    use_layout = any(page.headings for page in pages)
    lines = _lines(pages, text, page_starts)
    sections = [_Section(0, None)]
    previous: _Line | None = None
    consumed: set[int] = set()  # chapter title lines already folded into their marker
    for i, line in enumerate(lines):
        if i in consumed:
            previous = line
            continue
        if _is_chapter(line.text) and (chapter := _chapter_title(lines, i)):
            title, used = chapter
            consumed.update(used)
            if line.start == 0 or text[: line.start].strip() == "":
                sections[0] = _Section(0, title)
            else:
                sections.append(_Section(line.start, title))
            previous = line
            continue
        title = _heading_title(line, pages, use_layout)
        if title is None:
            previous = line
            continue
        if (
            i
            and (i - 1) not in consumed
            and sections[-1].title
            and _heading_title(lines[i - 1], pages, use_layout)
        ):
            # consecutive heading lines are one title (a heading wrapped over two lines)
            merged = sections[-1]
            sections[-1] = _Section(merged.start, f"{merged.title} {title}")
            continue
        start = line.start
        if previous and len(previous.text) <= KICKER_MAX_CHARS and is_letter_spaced(previous.text):
            start = previous.start
        title = normalize(title)
        if start == 0 or text[:start].strip() == "":
            sections[0] = _Section(0, title)
        else:
            sections.append(_Section(start, title))
        previous = line
    return sections


def _atoms(text: str, start: int, end: int, size: int, level: int) -> list[tuple[int, int]]:
    """Cut [start, end) at the most structural separator that yields pieces <= size."""
    if end - start <= size:
        return [(start, end)]
    if level == len(_SEPARATORS):  # a single "word" longer than size: hard cut
        return [(i, min(i + size, end)) for i in range(start, end, size)]
    cuts = [m.end() for m in _SEPARATORS[level].finditer(text, start, end) if m.end() < end]
    atoms: list[tuple[int, int]] = []
    for a, b in zip([start, *cuts], [*cuts, end], strict=True):
        if b > a:
            atoms += _atoms(text, a, b, size, level + 1) if b - a > size else [(a, b)]
    return atoms


def _merge(atoms: list[tuple[int, int]], size: int, overlap: int) -> list[tuple[int, int]]:
    """Greedily pack consecutive atoms up to `size`; each chunk repeats up to `overlap` chars of
    the previous one's tail (whole atoms only, so it starts on a boundary too)."""
    spans: list[tuple[int, int]] = []
    current: list[tuple[int, int]] = []
    for atom in atoms:
        if current and atom[1] - current[0][0] > size:
            spans.append((current[0][0], current[-1][1]))
            tail = [a for a in current if current[-1][1] - a[0] <= overlap]
            current = tail if tail and atom[1] - tail[0][0] <= size else []
        current.append(atom)
    if current:
        spans.append((current[0][0], current[-1][1]))
    return spans
