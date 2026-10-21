import pymupdf
import pytest

from chat_pdf.adapters.extractors import pdf as pdf_module
from chat_pdf.adapters.extractors.pdf import PdfLoader
from chat_pdf.adapters.extractors.text import TextLoader


@pytest.mark.parametrize(
    "raw",
    [
        "Hola ñandú\n".encode("utf-16"),  # BOM: would decode to NUL-riddled text as utf-8
        "Hola ñandú\n".encode("utf-8-sig"),
        "Hola\x00 ñandú\n".encode(),  # stray NUL: Postgres text rejects it
    ],
    ids=["utf-16", "utf-8-sig", "nul"],
)
def test_text_loader_decodes_bom_and_strips_nul(tmp_path, raw):
    path = tmp_path / "a.txt"
    path.write_bytes(raw)

    [page] = TextLoader().load(path)

    assert page.text == "Hola ñandú\n"


def test_pdf_loader_marks_lines_in_a_much_larger_font_as_headings(tmp_path):
    path = tmp_path / "a.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "Área de mercado", fontsize=20)
        for i in range(8):
            page.insert_text((72, 110 + 14 * i), f"Texto de cuerpo número {i}.", fontsize=10)
        page.insert_text((72, 240), "12", fontsize=20)  # big but no words: not a heading
        doc.save(path)

    [page] = PdfLoader().load(path)

    assert page.headings == ("Área de mercado",)
    assert "Texto de cuerpo número 7." in page.text


def _scanned_pdf(path):
    """A PDF whose only content is a picture of text: no text layer."""
    with pymupdf.open() as source:
        source.new_page().insert_text((72, 120), "Contrato de arrendamiento", fontsize=28)
        pixmap = source[0].get_pixmap(dpi=200)
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_image(page.rect, pixmap=pixmap)
        doc.save(path)


def test_pdf_loader_leaves_scanned_pages_empty_without_tesseract(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_module, "ocr_available", lambda: False)
    _scanned_pdf(tmp_path / "scan.pdf")

    [page] = PdfLoader().load(tmp_path / "scan.pdf")

    assert page.text.strip() == ""


@pytest.mark.skipif(not pdf_module.ocr_available(), reason="tesseract not installed")
def test_pdf_loader_ocrs_scanned_pages(tmp_path):
    _scanned_pdf(tmp_path / "scan.pdf")

    [page] = PdfLoader().load(tmp_path / "scan.pdf")

    assert "arrendamiento" in page.text.lower()
