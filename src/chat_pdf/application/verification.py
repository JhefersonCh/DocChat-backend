from uuid import UUID

from chat_pdf.application.chats import require_chat
from chat_pdf.application.generation import INVALID_RESPONSE, _require_chat_profile
from chat_pdf.application.ports import ChatRepository, ProviderFactory, ProviderRepository
from chat_pdf.domain.errors import NotFound, ProviderError
from chat_pdf.domain.verification import (
    check_claims,
    parse_claims,
    split_sentences,
    verify_messages,
)


class VerifyAnswer:
    """Claim check of an assistant message against its citations; computed once, then stored."""

    def __init__(
        self, chats: ChatRepository, providers: ProviderRepository, factory: ProviderFactory
    ):
        self._chats = chats
        self._providers = providers
        self._factory = factory

    async def execute(self, chat_id: UUID, message_id: UUID) -> dict:
        await require_chat(self._chats, chat_id)
        message = next(
            (
                m
                for m in await self._chats.messages(chat_id)
                if m.id == message_id and m.role == "assistant"
            ),
            None,
        )
        if message is None:
            raise NotFound("message_not_found", "Mensaje no encontrado")
        if message.verification:
            return message.verification
        sources = message.citations or []
        sentences = split_sentences(message.content)
        claims: list[dict] = []
        if sentences:
            profile = await _require_chat_profile(self._providers)
            result = await self._factory.chat(profile).complete(verify_messages(sentences, sources))
            parsed = parse_claims(result.content or "", len(sentences), sources)
            if parsed is None:
                raise ProviderError("El modelo no devolvió una verificación", INVALID_RESPONSE)
            claims = check_claims(parsed, sources)
        verification = {"sentences": sentences, "claims": claims}
        await self._chats.set_verification(message_id, verification)
        return verification
