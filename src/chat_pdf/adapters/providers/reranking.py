from chat_pdf.domain.models import Chunk


class IdentityReranker:
    """Implements the Reranker port as a no-op: keeps the fused order.

    The seam for a cross-encoder (e.g. llama.cpp `/v1/rerank`): a real adapter replaces this in
    `bootstrap.py` without touching the use cases. A local CPU reranker measured ~3.8 s per
    candidate, too slow to run interactively, so none is wired for now.
    """

    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        return chunks
