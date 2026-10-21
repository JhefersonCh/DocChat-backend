import codecs
from pathlib import Path

from chat_pdf.domain.models import Page


class TextLoader:
    """Implements the Loader port for TXT / Markdown (a single page without number)."""

    def load(self, path: Path) -> list[Page]:
        raw = path.read_bytes()
        if raw.startswith(codecs.BOM_UTF8):
            encoding = "utf-8-sig"
        elif raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            encoding = "utf-16"  # e.g. "Save as Unicode" from Windows Notepad
        else:
            encoding = "utf-8"
        text = raw.decode(encoding, errors="replace").replace("\x00", "")  # Postgres rejects NUL
        return [Page(number=None, text=text)]
