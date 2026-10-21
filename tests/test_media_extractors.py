"""Image-region extraction with tiny generated PDFs / images (plan §2.2, §2.3)."""

import io

import pymupdf
import pytest
from PIL import Image

from chat_pdf.adapters.media.image_file import ImageFileExtractor
from chat_pdf.adapters.media.pdf_regions import PdfMediaExtractor
from chat_pdf.domain.errors import InvalidInput
from chat_pdf.domain.figures import MediaRegion

BODY = "Texto de cuerpo del documento de prueba con bastante contenido para tener capa de texto."


def picture(color: tuple[int, int, int], size=(320, 240)) -> bytes:
    """A deterministic non-flat picture (diagonal gradient) as PNG bytes."""
    image = Image.new("RGB", size)
    r, g, b = color
    image.putdata(
        [((r + x) % 256, (g + y) % 256, b) for y in range(size[1]) for x in range(size[0])]
    )
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def _page_with_text(doc: pymupdf.Document) -> pymupdf.Page:
    page = doc.new_page()  # A4 595 x 842
    page.insert_text((72, 72), BODY, fontsize=9)
    return page


def _pdf(tmp_path, build) -> str:
    path = tmp_path / "t.pdf"
    with pymupdf.open() as doc:
        build(doc)
        doc.save(path)
    return path


def test_embedded_pictures_keep_a_normalized_bbox(tmp_path):
    def build(doc):
        _page_with_text(doc).insert_image(
            pymupdf.Rect(100, 200, 420, 440), stream=picture((10, 20, 30))
        )

    [region] = PdfMediaExtractor().plan(_pdf(tmp_path, build), {1: 90}).regions

    assert (region.origin, region.page) == ("embedded_image", 1)
    assert region.bbox == pytest.approx([100 / 595, 200 / 842, 420 / 595, 440 / 842], abs=1e-3)
    assert region.sha


def test_tiny_images_are_skipped(tmp_path):
    def build(doc):
        page = _page_with_text(doc)
        page.insert_image(pymupdf.Rect(100, 200, 120, 220), stream=picture((1, 2, 3), (16, 16)))
        page.insert_image(pymupdf.Rect(100, 300, 400, 600), stream=picture((9, 9, 9), (16, 300)))

    assert PdfMediaExtractor().plan(_pdf(tmp_path, build), {1: 90}).regions == []


def test_a_logo_on_every_page_keeps_one_region_per_occurrence(tmp_path):
    logo, photo = picture((200, 0, 0), (200, 100)), picture((0, 90, 200))

    def build(doc):
        for n in range(3):
            page = _page_with_text(doc)
            page.insert_image(pymupdf.Rect(20, 20, 220, 120), stream=logo)
            if n == 1:
                page.insert_image(pymupdf.Rect(100, 300, 400, 560), stream=photo)

    regions = PdfMediaExtractor().plan(_pdf(tmp_path, build), {}).regions

    assert [(r.origin, r.page) for r in regions] == [("embedded_image", n) for n in (1, 2, 2, 3)]


def test_vector_chart_becomes_a_figure_and_small_rules_do_not(tmp_path):
    def build(doc):
        page = _page_with_text(doc)
        for i, height in enumerate((80, 160, 120, 200)):  # bars of a drawn chart
            page.draw_rect(
                pymupdf.Rect(100 + i * 70, 500 - height, 150 + i * 70, 500), fill=(0.2, 0.4, 0.8)
            )
        page.draw_rect(pymupdf.Rect(90, 290, 400, 505))  # the axes frame joins the bars
        page.draw_line((72, 90), (200, 90))  # an underline: far below 8 % of the page

    [region] = PdfMediaExtractor().plan(_pdf(tmp_path, build), {1: 90}).regions

    assert (region.origin, region.sha) == ("figure", None)
    assert region.bbox[0] < 100 / 595 < region.bbox[2]  # padded around the bars


def _scan(doc, text="Contrato"):
    with pymupdf.open() as source:
        source.new_page().insert_text((72, 120), text, fontsize=28)
        pix = source[0].get_pixmap(dpi=100)
    page = doc.new_page()
    page.insert_image(page.rect, pixmap=pix)


def test_scanned_page_is_one_page_segment_when_ocr_found_text(tmp_path):
    regions = PdfMediaExtractor().plan(_pdf(tmp_path, _scan), {1: 1500}).regions

    assert [(r.origin, r.bbox) for r in regions] == [("page", [0.0, 0.0, 1.0, 1.0])]


def test_scanned_picture_page_without_text_also_gets_tiles(tmp_path):
    regions = PdfMediaExtractor().plan(_pdf(tmp_path, _scan), {1: 12}).regions

    assert [r.origin for r in regions] == ["page"] + ["tile"] * 9


def test_blank_page_has_no_segments(tmp_path):
    assert PdfMediaExtractor().plan(_pdf(tmp_path, lambda d: d.new_page()), {}).regions == []


def test_render_gives_bounded_webp_asset_thumb_and_png_embedding(tmp_path):
    def build(doc):
        _page_with_text(doc).insert_image(
            pymupdf.Rect(50, 100, 550, 700), stream=picture((5, 5, 5))
        )

    path = _pdf(tmp_path, build)
    extractor = PdfMediaExtractor()
    [region] = extractor.plan(path, {1: 90}).regions

    rendered = extractor.render(path, region)

    asset, thumb, embed = (
        Image.open(io.BytesIO(b)) for b in (rendered.asset, rendered.thumb, rendered.embed)
    )
    assert (asset.format, thumb.format, embed.format) == ("WEBP", "WEBP", "PNG")
    assert max(asset.size) <= 1024 and max(thumb.size) <= 256 and max(embed.size) <= 384
    assert max(asset.size) == 1024 and len(rendered.sha) == 64


def test_identical_regions_share_a_content_hash(tmp_path):
    same = picture((7, 7, 7))

    def build(doc):
        _page_with_text(doc).insert_image(pymupdf.Rect(50, 100, 350, 340), stream=same)
        _page_with_text(doc).insert_image(pymupdf.Rect(50, 100, 350, 340), stream=same)

    path = _pdf(tmp_path, build)
    extractor = PdfMediaExtractor()
    first, second = extractor.plan(path, {1: 90, 2: 90}).regions

    assert extractor.render(path, first).sha == extractor.render(path, second).sha


def _image_file(tmp_path, size, name="a.png", **save):
    path = tmp_path / name
    Image.new("RGB", size, (200, 30, 30)).save(path, **save)
    return path


def test_small_standalone_image_is_a_single_whole_image_segment(tmp_path):
    plan = ImageFileExtractor().plan(_image_file(tmp_path, (800, 600)), {})

    assert [(r.origin, r.bbox, r.page) for r in plan.regions] == [
        ("whole_image", [0, 0, 1, 1], None)
    ]
    assert (plan.width, plan.height) == (800, 600)


def test_large_standalone_image_adds_tiles_and_a_tile_renders_its_own_region(tmp_path):
    path = _image_file(tmp_path, (2000, 1000), "big.jpg")
    extractor = ImageFileExtractor()

    plan = extractor.plan(path, {})
    tile = extractor.render(path, plan.regions[1])

    assert [r.origin for r in plan.regions] == ["whole_image"] + ["tile"] * 9
    assert Image.open(io.BytesIO(tile.asset)).size == (733, 367)


def test_exif_orientation_is_applied(tmp_path):
    path = tmp_path / "rot.jpg"
    image = Image.new("RGB", (400, 200), (10, 200, 10))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    image.save(path, exif=exif)

    plan = ImageFileExtractor().plan(path, {})

    assert (plan.width, plan.height) == (200, 400)


def test_transparent_png_is_composited_on_white(tmp_path):
    path = tmp_path / "t.png"
    Image.new("RGBA", (100, 100), (0, 0, 0, 0)).save(path)
    extractor = ImageFileExtractor()

    rendered = extractor.render(path, MediaRegion("whole_image", None, [0, 0, 1, 1]))

    assert Image.open(io.BytesIO(rendered.embed)).getpixel((5, 5)) == (255, 255, 255)


def test_undecodable_image_is_invalid_input(tmp_path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"not an image")

    with pytest.raises(InvalidInput):
        ImageFileExtractor().plan(path, {})
