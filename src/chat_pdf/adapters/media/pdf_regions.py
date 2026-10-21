"""PDF -> image segments (plan §2.2): embedded pictures with their bbox, vector charts, and
scanned / picture-only pages. A picture repeated on many pages keeps one region per occurrence."""

import logging
from collections.abc import Mapping
from pathlib import Path

import pymupdf
from PIL import Image

from chat_pdf.adapters.media.encode import render_all
from chat_pdf.application.ports import MediaPlan, RenderedMedia
from chat_pdf.domain.errors import InvalidInput
from chat_pdf.domain.figures import (
    MAX_ASSET_SIDE,
    BBox,
    MediaRegion,
    bbox_area,
    is_figure_cluster,
    is_meaningful_image,
    is_scanned,
    normalize_bbox,
    overlap_ratio,
    scanned_page_regions,
)

log = logging.getLogger(__name__)

FIGURE_PADDING = 0.03  # page fraction added around a vector cluster so axis labels fit
FIGURE_OVERLAP_WITH_IMAGE = 0.5  # a cluster mostly covered by an embedded picture is that picture
MAX_ZOOM = 3.0  # small crops are rendered up to 3x (72 dpi points -> 216 dpi)


def _norm(rect: pymupdf.Rect, page: pymupdf.Rect) -> BBox:
    return normalize_bbox(
        rect.x0 - page.x0, rect.y0 - page.y0, rect.x1 - page.x0, rect.y1 - page.y0,
        page.width, page.height,
    )  # fmt: skip


def _pad(box: BBox, amount: float) -> BBox:
    return [max(0.0, box[0] - amount), max(0.0, box[1] - amount),
            min(1.0, box[2] + amount), min(1.0, box[3] + amount)]  # fmt: skip


def _page_regions(page: pymupdf.Page, ocr_chars: int) -> list[MediaRegion]:
    number, rect = page.number + 1, page.rect
    infos = page.get_image_info(hashes=True, xrefs=True)
    boxes = [_norm(pymupdf.Rect(i["bbox"]), rect) for i in infos]
    coverage = min(1.0, sum(bbox_area(b) for b in boxes))
    native_chars = len(page.get_text().strip())
    if is_scanned(native_chars, coverage):
        return scanned_page_regions(number, ocr_chars)
    regions = [
        MediaRegion("embedded_image", number, box, (info.get("digest") or b"").hex() or None)
        for info, box in zip(infos, boxes, strict=True)
        if is_meaningful_image(info["width"], info["height"], box)
    ]  # fmt: skip
    pictures = [r.bbox for r in regions]
    for cluster in page.cluster_drawings():
        box = _norm(cluster, rect)
        if is_figure_cluster(box) and not any(
            overlap_ratio(box, p) >= FIGURE_OVERLAP_WITH_IMAGE for p in pictures
        ):
            regions.append(MediaRegion("figure", number, _pad(box, FIGURE_PADDING)))
    return regions


class PdfMediaExtractor:
    """Implements the MediaExtractor port for PDF."""

    def plan(self, path: Path, page_text_chars: Mapping[int, int]) -> MediaPlan:
        with pymupdf.open(path) as doc:
            regions: list[MediaRegion] = []
            for page in doc:
                try:
                    regions += _page_regions(page, page_text_chars.get(page.number + 1, 0))
                except Exception:  # a damaged page must not lose the rest of the document
                    log.warning("media of page %s skipped", page.number + 1, exc_info=True)
            return MediaPlan(regions)

    def render(self, path: Path, region: MediaRegion) -> RenderedMedia:
        if region.page is None:
            raise ValueError("a PDF region needs a page")
        with pymupdf.open(path) as doc:
            page = doc[region.page - 1]
            r = page.rect
            x0, y0, x1, y1 = region.bbox
            clip = pymupdf.Rect(
                r.x0 + x0 * r.width, r.y0 + y0 * r.height, r.x0 + x1 * r.width, r.y0 + y1 * r.height
            )
            zoom = min(MAX_ASSET_SIDE / max(clip.width, clip.height, 1.0), MAX_ZOOM)
            try:
                pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
            except (RuntimeError, ValueError) as exc:
                raise InvalidInput("invalid_image", "No se pudo renderizar la región") from exc
            return render_all(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
