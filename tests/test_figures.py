import pytest

from chat_pdf.domain.figures import (
    is_figure_cluster,
    is_meaningful_image,
    is_scanned,
    normalize_bbox,
    scanned_page_regions,
    standalone_image_regions,
    tile_boxes,
)


def test_bbox_is_normalized_clamped_and_ordered():
    assert normalize_bbox(100, 200, 300, 400, 600, 800) == [0.1667, 0.25, 0.5, 0.5]
    assert normalize_bbox(-10, -10, 700, 900, 600, 800) == [0.0, 0.0, 1.0, 1.0]
    assert normalize_bbox(300, 400, 100, 200, 600, 800) == [0.1667, 0.25, 0.5, 0.5]


@pytest.mark.parametrize(
    ("w", "h", "box", "keep"),
    [
        (500, 400, [0.1, 0.1, 0.6, 0.5], True),
        (20, 400, [0.1, 0.1, 0.6, 0.5], False),  # a side under 24 px
        (500, 400, [0.1, 0.1, 0.12, 0.12], False),  # under 0.2 % of the page
    ],
)
def test_tiny_images_are_skipped(w, h, box, keep):
    assert is_meaningful_image(w, h, box) is keep


def test_vector_cluster_size_decides_figure():
    assert is_figure_cluster([0.1, 0.1, 0.6, 0.6])
    assert not is_figure_cluster([0.1, 0.1, 0.2, 0.2])  # a rule or a small icon
    assert not is_figure_cluster([0.0, 0.0, 1.0, 1.0])  # page background


def test_scanned_page_has_no_text_layer_and_is_mostly_picture():
    assert is_scanned(0, 0.95)
    assert not is_scanned(500, 0.95)  # has text
    assert not is_scanned(0, 0.1)  # blank page


def test_scanned_page_with_ocr_text_is_a_single_page_segment():
    [only] = scanned_page_regions(4, ocr_chars=2000)

    assert (only.origin, only.page, only.bbox) == ("page", 4, [0.0, 0.0, 1.0, 1.0])


def test_pictorial_scan_adds_nine_overlapping_tiles():
    regions = scanned_page_regions(1, ocr_chars=30)

    assert [r.origin for r in regions] == ["page"] + ["tile"] * 9
    left, right = regions[1].bbox, regions[2].bbox
    assert left[2] > right[0]  # neighbours overlap


def test_tiles_cover_the_whole_unit_square():
    boxes = tile_boxes()
    assert min(b[0] for b in boxes) == 0.0 and max(b[2] for b in boxes) == 1.0
    assert min(b[1] for b in boxes) == 0.0 and max(b[3] for b in boxes) == 1.0


def test_only_large_standalone_images_get_tiles():
    assert [r.origin for r in standalone_image_regions(800, 600)] == ["whole_image"]
    assert len(standalone_image_regions(2000, 1000)) == 10
