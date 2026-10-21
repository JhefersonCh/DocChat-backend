"""Stage 1 of image ingest (plan D5): extract the regions, store crops / thumbs and embed each
one natively. Local only: it never needs the vision profile, so the source becomes searchable
before any description exists."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from uuid import UUID, uuid4

from chat_pdf.application.ports import (
    FileStorage,
    MediaExtractor,
    MediaPlan,
    ProviderFactory,
)
from chat_pdf.domain.errors import InvalidInput, ProviderError
from chat_pdf.domain.figures import context_text
from chat_pdf.domain.models import Chunk, Page
from chat_pdf.domain.providers import ProviderProfile

log = logging.getLogger(__name__)

MAX_CONSECUTIVE_EMBED_FAILURES = 3  # the embeddings server is down: stop instead of skipping all
Progress = Callable[[int], Awaitable[None]]  # done regions; raises IngestCancelled


def supports_images(profile: ProviderProfile) -> bool:
    """The embedding profile's probe found the server can embed images (mmproj loaded)."""
    return "image" in profile.modalities


def page_contexts(chunks: list[Chunk]) -> dict[int | None, str]:
    """`title: <section> | text: <nearby text>` of each page, from its text chunks: what gets
    embedded together with a picture of that page (contextual embedding)."""
    first: dict[int, Chunk] = {}
    for c in chunks:
        if c.modality == "text" and c.page_start is not None:
            first.setdefault(c.page_start, c)
    return {n: context_text(c.section, c.content) for n, c in first.items()}


class MediaIngest:
    def __init__(
        self,
        extractors: Mapping[str, MediaExtractor],
        factory: ProviderFactory,
        storage: FileStorage,
    ):
        self._extractors = extractors
        self._factory = factory
        self._storage = storage

    def handles(self, mime: str) -> bool:
        return mime in self._extractors

    async def plan(self, mime: str, path: Path, pages: list[Page]) -> MediaPlan:
        """Regions to turn into segments. `pages` carry the OCR text that decides tiling."""
        chars = {p.number: len(p.text.strip()) for p in pages if p.number is not None}
        return await asyncio.to_thread(self._extractors[mime].plan, path, chars)

    async def segments(
        self,
        document_id: UUID,
        mime: str,
        path: Path,
        profile: ProviderProfile,
        plan: MediaPlan,
        first_ordinal: int,
        progress: Progress,
        contexts: Mapping[int | None, str] | None = None,
    ) -> tuple[list[Chunk], list[list[float]]]:
        """One image segment per region (assets saved, native vector computed together with the
        page's context text). Grid tiles are children of the whole picture / page (native vector
        only; the vision model describes the parent and finds the content-aware regions). A
        region that cannot be rendered or embedded is skipped; a dead embeddings server aborts."""
        extractor = self._extractors[mime]
        embedder = self._factory.multimodal_embedder(profile)
        await self._storage.clear_assets(document_id)  # a re-index regenerates every crop
        cache: dict[tuple[str, str], list[float]] = {}  # same picture + same context: once
        parent: Chunk | None = None  # last whole picture / page: the parent of its grid tiles
        chunks: list[Chunk] = []
        vectors: list[list[float]] = []
        failures = 0
        for done, region in enumerate(plan.regions):
            await progress(done)
            try:
                rendered = await asyncio.to_thread(extractor.render, path, region)
            except InvalidInput:
                log.warning("region %s of %s cannot be rendered; skipped", region, document_id)
                continue
            context = (contexts or {}).get(region.page, "")
            key = (rendered.sha, context)
            vector = cache.get(key)
            if vector is None:
                try:
                    vector = await embedder.embed_image(rendered.embed, context or None)
                except (ProviderError, InvalidInput):
                    failures += 1
                    log.warning("image embedding failed for %s", document_id, exc_info=True)
                    if failures >= MAX_CONSECUTIVE_EMBED_FAILURES:
                        raise
                    continue
                cache[key] = vector
            failures = 0
            ordinal = first_ordinal + len(chunks)
            name = f"{ordinal}.webp"
            await self._storage.save_asset(document_id, "assets", name, rendered.asset)
            await self._storage.save_asset(document_id, "thumbs", name, rendered.thumb)
            is_tile = region.origin == "tile"
            chunk = Chunk(
                id=uuid4(),
                parent_id=parent.id if is_tile and parent else None,
                ordinal=ordinal,
                content="",  # the description fills it in stage 2
                page_start=region.page,
                page_end=region.page,
                char_start=None,
                char_end=None,
                modality="image",
                origin=region.origin,
                bbox=region.bbox,
                asset_key=name,
                media_sha=rendered.sha,
                text_status="skipped" if is_tile else "pending",
            )
            if not is_tile:
                parent = chunk
            chunks.append(chunk)
            vectors.append(vector)
        await progress(len(plan.regions))
        return chunks, vectors
