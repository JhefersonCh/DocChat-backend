from chat_pdf.adapters.extractors.pdf import PdfLoader
from chat_pdf.adapters.extractors.text import TextLoader
from chat_pdf.adapters.media.image_file import ImageFileExtractor
from chat_pdf.adapters.media.pdf_regions import PdfMediaExtractor
from chat_pdf.application.ports import Loader, MediaExtractor

# Keyed by canonical MIME type (see adapters/http/chats.py for upload validation).
LOADERS: dict[str, Loader] = {
    "application/pdf": PdfLoader(),
    "text/plain": TextLoader(),
    "text/markdown": TextLoader(),
}

# Image regions of a file (embedded pictures, charts, scans), by canonical MIME type.
_IMAGES = ImageFileExtractor()
MEDIA_EXTRACTORS: dict[str, MediaExtractor] = {
    "application/pdf": PdfMediaExtractor(),
    "image/jpeg": _IMAGES,
    "image/png": _IMAGES,
    "image/webp": _IMAGES,
    "image/gif": _IMAGES,
}
