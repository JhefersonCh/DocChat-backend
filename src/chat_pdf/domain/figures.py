"""Which regions of a page / image become segments (plan §2.2, §2.3). Pure geometry + rules."""

from dataclasses import dataclass

MIN_SIDE_PX = 24  # an embedded image smaller than this (either side) is an icon / bullet
MIN_AREA_RATIO = 0.002  # ... or covering less than 0.2 % of the page
FIGURE_MIN_AREA_RATIO = 0.08  # a vector drawing cluster this big is a chart / diagram
FIGURE_MAX_AREA_RATIO = 0.90  # ... but one covering the whole page is a background
SCANNED_MAX_NATIVE_CHARS = 20  # a page with less text layer than this may be a scan
SCANNED_MIN_COVERAGE = 0.5  # ... if pictures cover at least this much of it
PICTORIAL_MAX_OCR_CHARS = 400  # a scan with less OCR text than this is mostly picture: tile it
TILE_OVERLAP = 0.10
TILE_MIN_SIDE_PX = 1200  # a standalone image at least this large also gets a 3x3 grid
GRID = 3  # grid tiles per side (native embedding only: they are a fallback to locate things)
MAX_ASSET_SIDE = 1024
THUMB_SIDE = 256

BBox = list[float]  # [x0, y0, x1, y1] normalized 0-1
FULL: BBox = [0.0, 0.0, 1.0, 1.0]


@dataclass(frozen=True)
class MediaRegion:
    """A region to turn into an image segment. `page` is None for a standalone image."""

    origin: str  # embedded_image | figure | page | tile | whole_image
    page: int | None
    bbox: BBox
    sha: str | None = None  # content hash of an embedded image (repeats keep one region each)


def normalize_bbox(x0: float, y0: float, x1: float, y1: float, width: float, height: float) -> BBox:
    """Page/image coordinates -> [0, 1] fractions, clamped, corners ordered. Independent of DPI."""

    def clamp(v: float) -> float:
        return round(min(1.0, max(0.0, v)), 4)

    ax, bx = sorted((clamp(x0 / width), clamp(x1 / width)))
    ay, by = sorted((clamp(y0 / height), clamp(y1 / height)))
    return [ax, ay, bx, by]


def bbox_area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def overlap_ratio(box: BBox, other: BBox) -> float:
    """Fraction of `box` that lies inside `other`."""
    w = min(box[2], other[2]) - max(box[0], other[0])
    h = min(box[3], other[3]) - max(box[1], other[1])
    area = bbox_area(box)
    return 0.0 if w <= 0 or h <= 0 or area == 0 else w * h / area


def is_meaningful_image(width_px: int, height_px: int, bbox: BBox) -> bool:
    """False for icons / bullets: a side under 64 px or under 1.5 % of the page."""
    return min(width_px, height_px) >= MIN_SIDE_PX and bbox_area(bbox) >= MIN_AREA_RATIO


def is_figure_cluster(bbox: BBox) -> bool:
    return FIGURE_MIN_AREA_RATIO <= bbox_area(bbox) <= FIGURE_MAX_AREA_RATIO


def is_scanned(native_chars: int, picture_coverage: float) -> bool:
    return native_chars < SCANNED_MAX_NATIVE_CHARS and picture_coverage >= SCANNED_MIN_COVERAGE


def tile_boxes(overlap: float = TILE_OVERLAP) -> list[BBox]:
    """3x3 grid over the unit square; neighbours overlap by `overlap` of a cell's side."""
    step = 1.0 / GRID
    pad = step * overlap
    return [
        [round(max(0.0, c * step - pad), 4), round(max(0.0, r * step - pad), 4),
         round(min(1.0, (c + 1) * step + pad), 4), round(min(1.0, (r + 1) * step + pad), 4)]
        for r in range(GRID)
        for c in range(GRID)
    ]  # fmt: skip


def scanned_page_regions(page: int, ocr_chars: int) -> list[MediaRegion]:
    """A scanned page is one `page` segment; if OCR found (almost) no text it is a picture, so
    it also gets a 3x3 grid to locate regions. Text-heavy scans rely on the OCR text."""
    regions = [MediaRegion("page", page, list(FULL))]
    if ocr_chars < PICTORIAL_MAX_OCR_CHARS:
        regions += [MediaRegion("tile", page, box) for box in tile_boxes()]
    return regions


def standalone_image_regions(width_px: int, height_px: int) -> list[MediaRegion]:
    regions = [MediaRegion("whole_image", None, list(FULL))]
    if max(width_px, height_px) >= TILE_MIN_SIDE_PX:
        regions += [MediaRegion("tile", None, box) for box in tile_boxes()]
    return regions


def context_text(section: str | None, page_text: str | None, limit: int = 300) -> str:
    """Short text embedded together with a picture: "title: <section> | text: <nearby text>"."""
    near = " ".join((page_text or "").split())[:limit]
    parts = [f"title: {section}" if section else "", f"text: {near}" if near else ""]
    return " | ".join(p for p in parts if p)


def compose_bbox(parent: BBox, child: BBox) -> BBox:
    """A bbox relative to a parent region -> coordinates of the page the parent sits on."""
    w, h = parent[2] - parent[0], parent[3] - parent[1]
    return [
        round(parent[0] + child[0] * w, 4), round(parent[1] + child[1] * h, 4),
        round(parent[0] + child[2] * w, 4), round(parent[1] + child[3] * h, 4),
    ]  # fmt: skip
