"""What a chat's study features (Estudio, tutor) read: the chunks of its readable sources."""

from dataclasses import replace
from uuid import UUID

from chat_pdf.application.ports import ChatRepository, ChunkStore, DocumentRepository
from chat_pdf.domain.models import Chunk, Document
from chat_pdf.domain.sources import readable_sources

SOURCE_ORDINAL_STRIDE = 1_000_000  # keeps reading order when chunks of N sources are mixed


async def readable_documents(
    chats: ChatRepository | None, docs: DocumentRepository, chat_id: UUID
) -> list[Document]:
    """Ready, active sources of the chat in S1..Sn order."""
    chat = await chats.get(chat_id) if chats is not None else None
    return readable_sources(
        await docs.list_for_chat(chat_id), chat.disabled_sources if chat else ()
    )


async def readable_chunks(
    chats: ChatRepository | None, docs: DocumentRepository, chunks: ChunkStore, chat_id: UUID
) -> list[Chunk]:
    """All the chunks of those sources, each tagged with its `document_id`; ordinals are
    offset per source so ordering by ordinal reads S1 first, then S2…"""
    found: list[Chunk] = []
    for index, doc in enumerate(await readable_documents(chats, docs, chat_id)):
        for chunk in await chunks.for_document(doc.id):
            found.append(
                replace(
                    chunk,
                    ordinal=index * SOURCE_ORDINAL_STRIDE + chunk.ordinal,
                    document_id=chunk.document_id or doc.id,
                )
            )
    return found
