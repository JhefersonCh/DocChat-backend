import logging
from uuid import UUID

from chat_pdf.application.chats import require_chat
from chat_pdf.application.ports import (
    ChatRepository,
    DocumentRepository,
    ProviderFactory,
    ProviderRepository,
    UnitOfWorkFactory,
)
from chat_pdf.domain.errors import Conflict, NotFound, ProviderError
from chat_pdf.domain.generation import (
    overview_messages,
    parse_overview,
    parse_questions,
    suggestion_messages,
)
from chat_pdf.domain.providers import ProviderProfile

log = logging.getLogger(__name__)

INVALID_RESPONSE = "invalid_provider_response"


async def _require_chat_profile(providers: ProviderRepository) -> ProviderProfile:
    profile = await providers.active("chat")
    if profile is None:
        raise Conflict("no_chat_profile", "No hay un proveedor de chat activo")
    return profile


class SuggestFollowUps:
    """Three follow-up questions for an assistant message; generated once, then stored."""

    def __init__(
        self, chats: ChatRepository, providers: ProviderRepository, factory: ProviderFactory
    ):
        self._chats = chats
        self._providers = providers
        self._factory = factory

    async def execute(self, chat_id: UUID, message_id: UUID) -> list[str]:
        await require_chat(self._chats, chat_id)
        messages = await self._chats.messages(chat_id)
        index = next(
            (i for i, m in enumerate(messages) if m.id == message_id and m.role == "assistant"),
            None,
        )
        if index is None:
            raise NotFound("message_not_found", "Mensaje no encontrado")
        message = messages[index]
        if message.suggestions:
            return message.suggestions
        profile = await _require_chat_profile(self._providers)
        asked = next((m.content for m in reversed(messages[:index]) if m.role == "user"), "")
        sections = list(dict.fromkeys(c.section for c in message.citations or [] if c.section))
        result = await self._factory.chat(profile).complete(
            suggestion_messages(asked, message.content, sections)
        )
        questions = parse_questions(result.content or "")
        if not questions:
            raise ProviderError("El modelo no devolvió preguntas válidas", INVALID_RESPONSE)
        await self._chats.set_suggestions(message_id, questions)
        return questions


class RequestSummary:
    """Validates and marks the overview `pending`; the caller then runs GenerateOverview."""

    def __init__(self, docs: DocumentRepository, providers: ProviderRepository):
        self._docs = docs
        self._providers = providers

    async def execute(self, document_id: UUID) -> None:
        doc = await self._docs.get(document_id)
        if doc is None:
            raise NotFound("document_not_found", "Documento no encontrado")
        busy = Conflict("document_busy", "El documento no está listo o ya se está resumiendo")
        if doc.status != "ready" or doc.summary_status == "pending":
            raise busy
        await _require_chat_profile(self._providers)
        if not await self._docs.mark_summary_pending(document_id):
            raise busy


class GenerateOverview:
    """Background job: summary + starter questions. Never raises, never touches `status`."""

    def __init__(
        self, uow: UnitOfWorkFactory, providers: ProviderRepository, factory: ProviderFactory
    ):
        self._uow = uow
        self._providers = providers
        self._factory = factory

    async def execute(self, document_id: UUID) -> None:
        try:
            await self._generate(document_id)
        except Exception:
            log.exception("overview failed for %s", document_id)
            try:
                async with self._uow() as uow:
                    await uow.docs.set_summary(document_id, "failed")
            except Exception:
                log.exception("could not mark overview of %s as failed", document_id)

    async def _generate(self, document_id: UUID) -> None:
        profile = await self._providers.active("chat")
        async with self._uow() as uow:
            if profile is None:
                await uow.docs.set_summary(document_id, "skipped")
                return
            chunks = [c for c in await uow.chunks.for_document(document_id) if c.content.strip()]
        if not chunks:  # (media without description / transcript has nothing to say)
            raise RuntimeError("el documento no tiene fragmentos")
        result = await self._factory.chat(profile).complete(overview_messages(chunks))
        overview = parse_overview(result.content or "")
        if overview is None:
            raise ProviderError("El modelo no devolvió un resumen válido", INVALID_RESPONSE)
        async with self._uow() as uow:
            await uow.docs.set_summary(document_id, "ready", overview.summary, overview.questions)
