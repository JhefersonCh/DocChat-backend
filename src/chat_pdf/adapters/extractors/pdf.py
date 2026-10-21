import logging
import shutil
from collections import Counter
from pathlib import Path

import pymupdf

from chat_pdf.domain.models import Page

HEADING_SIZE_RATIO = 1.4  # a line this much larger than the body font is a heading
HEADING_MIN_LETTERS = 3  # big numbers ("5.700", "50 t") are figures, not headings
HEADING_MAX_CHARS = 120
OCR_MIN_CHARS = 20  # a page with less text than this is treated as scanned
OCR_LANGUAGES = "spa+eng"
OCR_DPI = 200

log = logging.getLogger(__name__)


def ocr_available() -> bool:
    return shutil.which("tesseract") is not None


def _ocr(page: pymupdf.Page) -> str:
    """Text of a scanned page via Tesseract (needs the binary + language data on the machine)."""
    try:
        textpage = page.get_textpage_ocr(language=OCR_LANGUAGES, dpi=OCR_DPI, full=True)
        return page.get_text(textpage=textpage).replace("\x00", "")
    except Exception:  # missing tessdata/language, corrupt image: the page stays empty
        log.warning("OCR failed for page %s", page.number + 1, exc_info=True)
        return ""


def _lines(page: pymupdf.Page) -> list[tuple[str, float]]:
    """(text, dominant font size) of each text line, in reading order."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if spans:
                text = "".join(s["text"] for s in line["spans"]).strip()
                lines.append((text, max(spans, key=lambda s: len(s["text"].strip()))["size"]))
    return lines


def _headings(pages: list[list[tuple[str, float]]]) -> list[tuple[str, ...]]:
    sizes: Counter[float] = Counter()
    for lines in pages:
        for text, size in lines:
            sizes[round(size, 1)] += len(text)
    if not sizes:
        return [() for _ in pages]
    body = sizes.most_common(1)[0][0]  # the size most text is set in
    return [
        tuple(
            text
            for text, size in lines
            if size >= HEADING_SIZE_RATIO * body
            and len(text) <= HEADING_MAX_CHARS
            and sum(ch.isalpha() for ch in text) >= HEADING_MIN_LETTERS
        )
        for lines in pages
    ]


class PdfLoader:
    """Implements the Loader port for PDF."""

    def load(self, path: Path) -> list[Page]:
        """One Page per PDF page (1-based). Pages without text layer come back empty.

        Headings are a layout hint (font size vs the body size); the splitter uses them to start
        sections. The text itself is PyMuPDF's plain extraction, unchanged. Pages with (almost) no
        text layer are OCRed when Tesseract is installed; otherwise they stay empty.
        """
        with pymupdf.open(path) as doc:
            texts = [page.get_text().replace("\x00", "") for page in doc]  # Postgres rejects NUL
            headings = _headings([_lines(page) for page in doc])
            scanned = [i for i, t in enumerate(texts) if len(t.strip()) < OCR_MIN_CHARS]
            if scanned and ocr_available():
                for i in scanned:
                    texts[i] = _ocr(doc[i]) or texts[i]
        return [
            Page(number=i, text=text, headings=h)
            for i, (text, h) in enumerate(zip(texts, headings, strict=True), start=1)
        ]
