import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import AsyncConnectionPool

from chat_pdf.adapters.http import artifacts as artifacts_http
from chat_pdf.adapters.http import health as health_http
from chat_pdf.adapters.http import inspector as inspector_http
from chat_pdf.adapters.http import library as library_http
from chat_pdf.adapters.http import media as media_http
from chat_pdf.adapters.http import notes as notes_http
from chat_pdf.adapters.http import providers as providers_http
from chat_pdf.adapters.http import search as search_http
from chat_pdf.adapters.http import tutor as tutor_http
from chat_pdf.adapters.http import voice as voice_http
from chat_pdf.adapters.http.access_log import PollingAccessFilter
from chat_pdf.adapters.http.chats import router as chats_router
from chat_pdf.adapters.http.errors import register_error_handlers
from chat_pdf.adapters.persistence.migrations import run_migrations
from chat_pdf.adapters.providers.openai_compatible import HttpProviderFactory
from chat_pdf.application.ports import ProviderFactory
from chat_pdf.bootstrap import build_use_cases
from chat_pdf.config import Settings
from chat_pdf.domain.config import RagConfig


def _quiet_polling_logs() -> None:
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, PollingAccessFilter) for f in access.filters):
        access.addFilter(PollingAccessFilter())


def create_app(
    settings: Settings | None = None, provider_factory: ProviderFactory | None = None
) -> FastAPI:
    """`provider_factory` lets tests replace the real LLM/embeddings HTTP clients."""
    settings = settings or Settings()  # type: ignore[call-arg]
    rag = RagConfig()
    _quiet_polling_logs()
    factory = provider_factory or HttpProviderFactory(
        rag.embed_batch,
        httpx.Timeout(settings.provider_timeout_s, connect=settings.provider_connect_timeout_s),
        settings.health_probe_timeout_s,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool = AsyncConnectionPool(settings.database_url, open=False)
        await pool.open(wait=True)
        async with pool.connection() as conn:
            await run_migrations(conn)
        app.state.use_cases = build_use_cases(settings, rag, pool, factory)
        await app.state.use_cases.seed_defaults.execute()
        await app.state.use_cases.fail_interrupted_ingests.execute()
        await app.state.use_cases.fail_interrupted_artifacts.execute()
        yield
        await pool.close()

    app = FastAPI(title="chat-pdf", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_error_handlers(app)
    app.include_router(health_http.router)
    app.include_router(providers_http.router)
    app.include_router(chats_router)
    app.include_router(media_http.router)
    app.include_router(artifacts_http.router)
    app.include_router(notes_http.router)
    app.include_router(inspector_http.router)
    app.include_router(tutor_http.router)
    app.include_router(voice_http.router)
    app.include_router(library_http.router)
    app.include_router(search_http.router)
    return app
