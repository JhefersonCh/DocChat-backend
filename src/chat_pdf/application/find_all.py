"""Exhaustive search of the media of one document (plan §3.3, D7).

Recall from the native vectors, the descriptions and keyword hits; precision from the vision
model, one yes/no per candidate, cached per (segment, normalized query) so that asking again
costs nothing. Without a vision profile the candidates are returned with a confidence label."""

import asyncio
import logging
from collections.abc import Awaitable
from dataclasses import dataclass, field
from uuid import UUID

from chat_pdf.application.ports import ChunkStore, FileStorage, Retriever, VisionModel
from chat_pdf.domain.calibration import CalibrationTable
from chat_pdf.domain.errors import ProviderError
from chat_pdf.domain.find_all import (
    MAX_OCCURRENCES,
    VISUAL,
    MediaMatch,
    collapse_regions,
    confidence,
    is_candidate,
    is_clear,
    normalize_query,
    parse_verdict,
    reading_order,
    strength,
    verify_prompt,
)
from chat_pdf.domain.models import Chunk

log = logging.getLogger(__name__)

MAX_VERIFICATIONS = (
    20  # vision calls per find_all (cached verdicts are free); the rest is unchecked
)
VERIFY_CONCURRENCY = 4
STOP_AFTER_CONSECUTIVE_FAILURES = 3  # a dead provider: stop paying for timeouts


@dataclass(frozen=True)
class Findings:
    chunks: list[Chunk]  # occurrences in reading order (at most MAX_OCCURRENCES)
    confidence: dict[UUID, str] = field(default_factory=dict)  # chunk id -> alta | media
    checked: int = 0  # confirmed by the vision model (now or from the cache)
    discarded: int = 0  # rejected by the vision model
    extra: int = 0  # occurrences beyond MAX_OCCURRENCES
    asked: int = 0  # vision calls made in this search (cache misses)


def modalities_for(hint: str) -> tuple[str, ...]:
    """Which segment modalities an exhaustive search covers for the model's hint."""
    return {"audio": ("audio",), "video": ("frame",)}.get(hint, ("image", "frame"))


class ExhaustiveFinder:
    def __init__(
        self,
        retriever: Retriever,
        chunks: ChunkStore,
        storage: FileStorage | None,
        vision: VisionModel | None,
        table: CalibrationTable,
    ):
        self._retriever = retriever
        self._chunks = chunks
        self._storage = storage
        self._vision = vision
        self._table = table

    async def find(
        self,
        document_ids: UUID | list[UUID],
        query: str,
        vector: list[float],
        hint: str = "imagen",
        max_page: int | None = None,
    ) -> Findings:
        """Every occurrence over one or several sources (`document_ids`)."""
        ids = [document_ids] if isinstance(document_ids, UUID) else list(document_ids)
        matches = await self._retriever.media_matches(
            ids, query, vector, modalities_for(hint), max_page
        )
        candidates = sorted(
            (m for m in matches if is_candidate(m, self._table)),
            key=lambda m: -strength(m, self._table),
        )
        key = normalize_query(query)
        verdicts, asked = await self._verdicts(ids[0], candidates, query, key)

        accepted: list[Chunk] = []  # confirmed (vision, cache or clear band)
        unchecked: dict[UUID, Chunk] = {}
        labels: dict[UUID, str] = {}
        checked = discarded = 0
        for match in candidates:
            chunk = match.chunk
            verdict = verdicts.get(chunk.id)
            if verdict is False:
                discarded += 1
            elif verdict is True:
                accepted.append(chunk)
                checked += 1
            elif is_clear(match, self._table):  # strong native evidence + keyword: unasked
                accepted.append(chunk)
                labels[chunk.id] = "alta"
            else:
                unchecked[chunk.id] = chunk
                labels[chunk.id] = confidence(match, self._table)
        occurrences = reading_order([*collapse_regions(accepted), *unchecked.values()])
        shown = occurrences[:MAX_OCCURRENCES]
        return Findings(
            shown,
            {c.id: labels[c.id] for c in shown if c.id in labels},
            checked,
            discarded,
            len(occurrences) - len(shown),
            asked,
        )

    async def _verdicts(
        self, document_id: UUID, candidates: list[MediaMatch], query: str, key: str
    ) -> tuple[dict[UUID, bool], int]:
        """Cached verdicts plus fresh ones for the candidates the vision model can look at."""
        if self._vision is None or self._storage is None:
            return {}, 0
        visual = [m for m in candidates if m.chunk.modality in VISUAL and m.chunk.asset_key]
        verdicts = await self._chunks.verdicts([m.chunk.id for m in visual], key)
        pending = [
            m for m in visual if m.chunk.id not in verdicts and not is_clear(m, self._table)
        ][:MAX_VERIFICATIONS]
        fresh = await self._ask(document_id, pending, query, key)
        return {**verdicts, **fresh}, len(pending)

    async def _ask(
        self, document_id: UUID, pending: list[MediaMatch], query: str, key: str
    ) -> dict[UUID, bool]:
        gate = asyncio.Semaphore(VERIFY_CONCURRENCY)
        failures = 0
        prompt = verify_prompt(query)

        async def one(match: MediaMatch) -> tuple[UUID, bool | None]:
            nonlocal failures
            chunk = match.chunk
            async with gate:
                if failures >= STOP_AFTER_CONSECUTIVE_FAILURES:
                    return chunk.id, None
                try:
                    verdict = await self._look(document_id, chunk, prompt)
                except (ProviderError, OSError) as exc:
                    failures += 1
                    log.warning("vision check of segment %s failed: %s", chunk.id, exc)
                    return chunk.id, None
                failures = 0
                return chunk.id, verdict

        answers = await _gather([one(m) for m in pending])
        fresh = {i: v for i, v in answers if v is not None}
        for chunk_id, verdict in fresh.items():
            await self._chunks.save_verdict(chunk_id, key, verdict)
        return fresh

    async def _look(self, document_id: UUID, chunk: Chunk, prompt: str) -> bool | None:
        # the asset lives in its own source's folder (`document_id` is only the fallback)
        path = self._storage.asset_path(chunk.document_id or document_id, "assets", chunk.asset_key)  # type: ignore[union-attr,arg-type]
        image = await asyncio.to_thread(path.read_bytes)
        answer = await self._vision.describe(image, prompt)  # type: ignore[union-attr]
        return parse_verdict(answer)


async def _gather(coroutines: list[Awaitable]) -> list:
    return list(await asyncio.gather(*coroutines))
