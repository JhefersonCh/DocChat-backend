FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

# OCR for scanned PDFs (PyMuPDF drives the tesseract binary; `eng` ships with the base package).
# ffmpeg: voice mode converts browser recordings (webm/ogg/mp4) to 16 kHz mono WAV.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-spa ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && test -f /usr/share/tesseract-ocr/5/tessdata/spa.traineddata
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY migrations ./migrations

RUN useradd --create-home app && mkdir -p /data/uploads && chown app /data/uploads
USER app

ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "chat_pdf.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
