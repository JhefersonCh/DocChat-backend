"""Retrieval eval (not part of the unit suite): ingest one file with the real pipeline and score it.

Runs the production path end to end (Loader -> splitter -> embeddings service -> PgChunkStore ->
DocumentSearch) against a throwaway database, then reports hit@1/3/5, MRR@10 and chunk stats for a
labeled question set. A hit is a retrieved chunk containing any `expect` phrase of the question.

    uv run python scripts/eval_retrieval.py PATH.pdf [--chunk-size 1000] [--no-text] [--show 3]

Needs the compose `db` (localhost:5433) and `embeddings` (localhost:8080) services up.
"""

import argparse
import asyncio
import hashlib
import json
import os
import time
import uuid
from dataclasses import fields, replace
from pathlib import Path

import psycopg
from psycopg_pool import AsyncConnectionPool

from chat_pdf.adapters.extractors import LOADERS
from chat_pdf.adapters.persistence.chats import PgChatRepository
from chat_pdf.adapters.persistence.chunks import PgChunkStore
from chat_pdf.adapters.persistence.documents import PgDocumentRepository
from chat_pdf.adapters.persistence.migrations import run_migrations
from chat_pdf.adapters.persistence.providers import PgProviderRepository
from chat_pdf.adapters.persistence.uow import PgUnitOfWork
from chat_pdf.adapters.providers.openai_compatible import HttpProviderFactory
from chat_pdf.adapters.providers.reranking import IdentityReranker
from chat_pdf.application.ask import DocumentSearch
from chat_pdf.application.documents import IngestDocument
from chat_pdf.application.ports import Candidates
from chat_pdf.application.providers import LOCAL_EMBEDDING
from chat_pdf.crypto import Crypto
from chat_pdf.domain.config import RagConfig
from chat_pdf.domain.rules import UPLOAD_TYPES

ADMIN_URL = os.environ.get(
    "EVAL_ADMIN_DATABASE_URL", "postgresql://chatpdf:chatpdf@localhost:5433/postgres"
)
DEFAULT_QUESTIONS = Path(__file__).with_name("eval_questions.json")
TOP = 10  # depth of the ranking used for MRR


class _OneFile:
    """FileStorage stand-in: every document id resolves to the evaluated file."""

    def __init__(self, path: Path):
        self._path = path

    def path(self, document_id: uuid.UUID) -> Path:
        return self._path


class _VectorOnly(PgChunkStore):
    """Ablation: drop the full-text ranking (what retrieval was before hybrid search)."""

    async def candidates(
        self, document_ids, query, vector, n, max_page=None, modalities=None
    ) -> Candidates:
        found = await super().candidates(document_ids, query, vector, n, max_page, modalities)
        return Candidates(found.by_vector, [])


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _first_hit(contents: list[str], expect: list[str]) -> int | None:
    phrases = [_norm(p) for p in expect]
    for rank, content in enumerate(contents, start=1):
        if any(p in _norm(content) for p in phrases):
            return rank
    return None


async def _evaluate(db_url: str, path: Path, questions: list[dict], rag: RagConfig, args) -> None:
    async with AsyncConnectionPool(db_url, open=False) as pool:
        async with pool.connection() as conn:
            await run_migrations(conn)
        providers = PgProviderRepository(pool, Crypto("eval"))
        overrides = {"base_url": args.embeddings_url}
        if args.query_prefix is not None:
            overrides["query_prefix"] = args.query_prefix
        profile = await providers.create({**LOCAL_EMBEDDING, **overrides})
        chat = await PgChatRepository(pool).create()
        docs = PgDocumentRepository(pool)
        document_id = uuid.uuid4()
        mime = UPLOAD_TYPES[path.suffix.lower()][0]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        await docs.create(document_id, chat.id, path.name, mime, digest, profile.id)

        factory = HttpProviderFactory(rag.embed_batch)
        ingest = IngestDocument(
            lambda: PgUnitOfWork(pool), providers, factory, LOADERS, _OneFile(path), rag
        )
        started = time.monotonic()
        await ingest.execute(document_id)
        ingest_s = time.monotonic() - started
        doc = await docs.get(document_id)
        if doc is None or doc.status != "ready":
            raise SystemExit(f"ingest failed: {doc.error if doc else 'missing document'}")

        async with pool.connection() as conn:
            cur = await conn.execute(
                "select count(*), avg(length(content)), max(length(content)) from chunks"
            )
            count, avg_len, max_len = await cur.fetchone()

        store = (_VectorOnly if args.no_text else PgChunkStore)(pool)
        hits, context = [], []
        for item in questions:
            search = DocumentSearch(
                store, IdentityReranker(), factory.embedder(profile), profile, doc, rag
            )
            sources = await search.run(item["q"], TOP)
            contents = [s.chunk.content for s in sources]
            rank = _first_hit(contents, item["expect"])
            hits.append(rank)
            context.append(sum(len(c) for c in contents[:5]))
            print(f"{rank or '-':>3}  p{item['pages']}  {item['q']}")
            for s in sources[: args.show]:
                head = " ".join(s.chunk.content.split())[:90]
                print(f"       p{s.chunk.page_start}-{s.chunk.page_end} [{s.chunk.section}] {head}")

    print(
        f"\nchunks={count} avg_chars={avg_len:.0f} max_chars={max_len} "
        f"chunk_size={rag.chunk_size} overlap={rag.chunk_overlap} ingest={ingest_s:.0f}s "
        f"context@5={sum(context) / len(context):.0f} chars "
        f"{'vector-only' if args.no_text else f'hybrid text_weight={rag.text_weight}'}"
    )
    kinds = {"all": hits}
    for kind in sorted({item.get("kind", "question") for item in questions}):
        kinds[kind] = [
            r for r, i in zip(hits, questions, strict=True) if i.get("kind", "question") == kind
        ]
    for kind, ranks in kinds.items():
        print(f"{kind:>9}: {_metrics(ranks)}")


def _metrics(ranks: list[int | None]) -> str:
    n = len(ranks)

    def hit_at(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / n

    mrr = sum(1 / r for r in ranks if r is not None) / n
    hits = " ".join(f"hit@{k}={hit_at(k):.2f}" for k in (1, 3, 5))
    return f"{hits} MRR@{TOP}={mrr:.3f} (n={n})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("path", type=Path)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--embeddings-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--show", type=int, default=0, help="print the top N chunks per question")
    parser.add_argument("--no-text", action="store_true", help="vector ranking only (ablation)")
    parser.add_argument("--query-prefix", help="override the seeded embedding query prefix")
    knobs = {f.name: f.type for f in fields(RagConfig) if f.type in (int, float)}
    for name, kind in knobs.items():  # any numeric RagConfig knob, e.g. --chunk-size 1000
        parser.add_argument(f"--{name.replace('_', '-')}", type=kind)
    args = parser.parse_args()
    overrides = {k: getattr(args, k) for k in knobs if getattr(args, k) is not None}
    rag = replace(RagConfig(), **overrides)
    questions = json.loads(args.questions.read_text())["questions"]

    name = f"chatpdf_eval_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
        admin.execute(f'create database "{name}"')
        try:
            db_url = ADMIN_URL.rsplit("/", 1)[0] + f"/{name}"
            asyncio.run(_evaluate(db_url, args.path.resolve(), questions, rag, args))
        finally:
            admin.execute(f'drop database "{name}" with (force)')


if __name__ == "__main__":
    main()
