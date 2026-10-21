# Supertonic 3 TTS (ONNX, CPU). OpenAI-compatible POST /v1/audio/speech on :7788.
# The repo is archived upstream, so the version is pinned. The ~386 MB model is downloaded on
# first start into /models (named volume `ttsmodels`), never at build time. The cache dir is a
# subdirectory because the loader downloads into a sibling temp dir.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 SUPERTONIC_CACHE_DIR=/models/supertonic3 HF_HOME=/models/hf
# TTS_STEPS (diffusion steps, default 8 upstream): 5 is ~1.5x faster with little audible loss;
# the OpenAI-compatible route offers no per-request knob, so the default is patched to read it.
ENV TTS_STEPS=5
RUN pip install "supertonic[serve]==1.3.1" \
    && f=$(python -c "import supertonic.config as c; print(c.__file__)") \
    && sed -i 's/^DEFAULT_TOTAL_STEPS = 8$/DEFAULT_TOTAL_STEPS = int(os.getenv("TTS_STEPS", "8"))/' "$f" \
    && grep -q 'TTS_STEPS' "$f" \
    && useradd --create-home tts \
    && mkdir -p /models && chown tts /models
USER tts
VOLUME /models
EXPOSE 7788
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=12 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:7788/v1/health', timeout=2)"
CMD ["supertonic", "serve", "--host", "0.0.0.0", "--port", "7788"]
