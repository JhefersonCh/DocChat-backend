"""A standalone image upload (plan §2.3): the whole image, plus 2x2 tiles when it is large."""

from collections.abc import Mapping
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from chat_pdf.adapters.media.encode import render_all, to_rgb
from chat_pdf.application.ports import MediaPlan, RenderedMedia
from chat_pdf.domain.errors import InvalidInput
from chat_pdf.domain.figures import MediaRegion, standalone_image_regions


def _open(path: Path) -> Image.Image:
    """Decoded RGB image with the EXIF orientation applied (first frame of a GIF)."""
    try:
        with Image.open(path) as source:
            return to_rgb(ImageOps.exif_transpose(source))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidInput("invalid_image", "No se pudo decodificar la imagen") from exc


class ImageFileExtractor:
    """Implements the MediaExtractor port for jpg / png / webp / gif files."""

    def plan(self, path: Path, page_text_chars: Mapping[int, int]) -> MediaPlan:
        width, height = _open(path).size
        return MediaPlan(standalone_image_regions(width, height), width, height)

    def render(self, path: Path, region: MediaRegion) -> RenderedMedia:
        image = _open(path)
        x0, y0, x1, y1 = region.bbox
        w, h = image.size
        box = (round(x0 * w), round(y0 * h), max(round(x1 * w), 1), max(round(y1 * h), 1))
        return render_all(image.crop(box))
