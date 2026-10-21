"""Pillow helpers: one rendered region -> the asset / thumb / embedding images (plan §1.4, §2.1)."""

import hashlib
import io

from PIL import Image

from chat_pdf.application.ports import RenderedMedia
from chat_pdf.domain.errors import InvalidInput
from chat_pdf.domain.figures import MAX_ASSET_SIDE, THUMB_SIDE

MIN_CROP_SIDE = 24  # only a speck smaller than this (in the parent's asset) says nothing
MIN_RENDER_SIDE = 192  # smaller crops / logos are upscaled to this before embedding
EMBED_SIDE = 384  # native image embeddings: 2.7 s vs 11-27 s per image on CPU, same score
WEBP_ASSET_QUALITY = 80
WEBP_THUMB_QUALITY = 72


def to_rgb(image: Image.Image) -> Image.Image:
    """RGB, with transparency composited on white (webp/png/gif logos would turn black)."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def fit(image: Image.Image, side: int) -> Image.Image:
    """Longest side <= `side`, aspect kept, never upscaled."""
    if max(image.size) <= side:
        return image
    copy = image.copy()
    copy.thumbnail((side, side), Image.Resampling.LANCZOS)
    return copy


def upscale(image: Image.Image, side: int = MIN_RENDER_SIDE) -> Image.Image:
    """Smallest side raised to `side` (LANCZOS): a 40 px logo is usable by the embedder / vision."""
    low = min(image.size)
    if low >= side:
        return image
    ratio = side / low
    size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _encode(image: Image.Image, fmt: str, **options: int) -> bytes:
    out = io.BytesIO()
    image.save(out, fmt, **options)
    return out.getvalue()


def render_all(image: Image.Image) -> RenderedMedia:
    image = upscale(to_rgb(image))
    embed = _encode(fit(image, EMBED_SIDE), "PNG")
    return RenderedMedia(
        asset=_encode(fit(image, MAX_ASSET_SIDE), "WEBP", quality=WEBP_ASSET_QUALITY),
        thumb=_encode(fit(image, THUMB_SIDE), "WEBP", quality=WEBP_THUMB_QUALITY),
        embed=embed,
        sha=hashlib.sha256(embed).hexdigest(),
    )


def crop_region(image: bytes, bbox: list[float]) -> RenderedMedia:
    """Crop a normalized bbox out of an encoded image (a parent's asset) -> its three sizes."""
    with Image.open(io.BytesIO(image)) as source:
        rgb = to_rgb(source)
    w, h = rgb.size
    x0, y0, x1, y1 = bbox
    box = (round(x0 * w), round(y0 * h), max(round(x1 * w), round(x0 * w) + 1),
           max(round(y1 * h), round(y0 * h) + 1))  # fmt: skip
    if min(box[2] - box[0], box[3] - box[1]) < MIN_CROP_SIDE:
        raise InvalidInput("region_too_small", "La región es demasiado chica")
    return render_all(rgb.crop(box))
