"""In-memory repositories, chunk store and unit of work (re-exported by `fakes`)."""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

from chat_pdf.application.ports import Candidates
from chat_pdf.domain.find_all import MediaMatch, normalize_query
from chat_pdf.domain.models import Chat, Chunk, Document, Message


@dataclass
class FakeDb:
    chats: dict[UUID, Chat] = field(default_factory=dict)
    docs: dict[UUID, Document] = field(default_factory=dict)
    chunks: dict[UUID, list[Chunk]] = field(default_factory=dict)
    exchanges: list[dict] = field(default_factory=list)
    messages: dict[UUID, list[Message]] = field(default_factory=dict)  # by chat id

    def add_chat(self) -> Chat:
        chat = Chat(uuid4(), "", datetime.now(UTC))
        self.chats[chat.id] = chat
        return chat

    def add_document(self, chat_id: UUID, profile_id: UUID, status: str = "ready") -> Document:
        doc = Document(
            uuid4(),
            chat_id,
            "x.txt",
            "text/plain",
            status,
            None,
            None,
            profile_id,
            datetime.now(UTC),
        )  # the chat id doubles as collection id
        self.docs[doc.id] = doc
        return doc


class FakeChatRepository:
    def __init__(self, db: FakeDb):
        self._db = db

    async def create(self):
        return self._db.add_chat()

    async def get(self, chat_id):
        return self._db.chats.get(chat_id)

    async def delete(self, chat_id):
        del self._db.chats[chat_id]
        return [d.id for d in self._db.docs.values() if d.collection_id == chat_id]

    async def lock(self, chat_id):
        return None

    async def set_title(self, chat_id, title):
        self._db.chats[chat_id] = replace(self._db.chats[chat_id], title=title)

    async def messages(self, chat_id) -> list[Message]:
        return self._db.messages.get(chat_id, [])

    async def history(self, chat_id, limit) -> list[Message]:
        return []

    async def set_suggestions(self, message_id, questions):
        for chat_id, messages in self._db.messages.items():
            self._db.messages[chat_id] = [
                replace(m, suggestions=questions) if m.id == message_id else m for m in messages
            ]

    async def set_verification(self, message_id, verification):
        for chat_id, messages in self._db.messages.items():
            self._db.messages[chat_id] = [
                replace(m, verification=verification) if m.id == message_id else m for m in messages
            ]

    async def add_exchange(
        self, chat_id, question, answer, citations, searches, weak_evidence=False, persona=None
    ):
        self._db.exchanges.append(
            {
                "persona": persona,
                "answer": answer,
                "citations": citations,
                "searches": searches,
                "weak_evidence": weak_evidence,
            }
        )
        return uuid4()

    async def update_settings(self, chat_id, changes):
        self._db.chats[chat_id] = replace(self._db.chats[chat_id], **changes)
        return self._db.chats[chat_id]

    async def list(self):
        return list(self._db.chats.values())


class FakeDocumentRepository:
    def __init__(self, db: FakeDb):
        self._db = db

    async def create(
        self, document_id, chat_id, filename, mime, sha256, embedding_profile_id, kind="pdf"
    ):
        self._db.docs[document_id] = Document(
            document_id, chat_id, filename, mime, "pending", None, None,
            embedding_profile_id, datetime.now(UTC), kind=kind,
        )  # fmt: skip

    async def get(self, document_id):
        return self._db.docs.get(document_id)

    async def list_for_chat(self, chat_id):
        docs = [d for d in self._db.docs.values() if d.collection_id == chat_id]
        return sorted(docs, key=lambda d: (d.created_at, str(d.id)))

    async def delete(self, document_id):
        self._db.docs.pop(document_id, None)
        self._db.chunks.pop(document_id, None)

    async def set_progress(self, document_id, done, total, stage=None):
        doc = self._db.docs.get(document_id)
        if doc is None:
            return False
        self._db.docs[document_id] = replace(
            doc, progress_done=done, progress_total=total, stage=stage or doc.stage
        )
        return doc.cancel_requested

    async def request_cancel(self, document_id):
        doc = self._db.docs[document_id]
        if doc.status != "processing" and doc.enrich_status != "running":
            return False
        self._db.docs[document_id] = replace(doc, cancel_requested=True)
        return True

    async def set_enrich(self, document_id, status, error=None, stage="describing"):
        doc = self._db.docs[document_id]
        self._db.docs[document_id] = replace(
            doc, enrich_status=status, enrich_error=error,
            stage=stage if status == "running" else None,
            progress_done=None, progress_total=None, cancel_requested=False,
        )  # fmt: skip

    async def mark_enrich_pending(self, document_id):
        doc = self._db.docs[document_id]
        if doc.status != "ready" or doc.enrich_status not in ("partial", "failed", "skipped"):
            return False
        await self.set_enrich(document_id, "pending")
        return True

    async def mark_processing(self, document_id):
        if self._db.docs[document_id].status not in ("ready", "failed"):
            return False
        await self.set_status(document_id, "processing", summary_status="skipped")
        self._db.docs[document_id] = replace(
            self._db.docs[document_id], summary=None, suggested_questions=[]
        )
        return True

    async def fail_unfinished(self, error):
        for doc in list(self._db.docs.values()):
            if doc.status in ("pending", "processing"):
                await self.set_status(doc.id, "failed", error=error)
            if doc.summary_status == "pending":
                await self.set_summary(doc.id, "failed")
            if doc.enrich_status in ("pending", "running"):
                await self.set_enrich(doc.id, "partial", error)

    async def set_status(
        self, document_id, status, error=None, page_count=None, summary_status=None,
        enrich_status=None, width=None, height=None, duration_s=None,
    ):  # fmt: skip
        doc = self._db.docs[document_id]
        self._db.docs[document_id] = replace(
            doc,
            status=status,
            error=error,
            page_count=page_count or doc.page_count,
            summary_status=summary_status or doc.summary_status,
            enrich_status=enrich_status or doc.enrich_status,
            width=width or doc.width,
            height=height or doc.height,
            duration_s=duration_s or doc.duration_s,
            stage=None,
            progress_done=None,
            progress_total=None,
            cancel_requested=False,
        )

    async def set_summary(self, document_id, status, summary=None, questions=None):
        doc = self._db.docs[document_id]
        self._db.docs[document_id] = replace(
            doc,
            summary_status=status,
            summary=summary or doc.summary,
            suggested_questions=doc.suggested_questions if questions is None else questions,
        )

    async def mark_summary_pending(self, document_id):
        doc = self._db.docs[document_id]
        if doc.status != "ready" or doc.summary_status == "pending":
            return False
        await self.set_summary(document_id, "pending")
        return True


class FakeChunkStore:
    """ChunkStore + Retriever. Both rankings are the stored order (the vector one first n)."""

    def __init__(self, db: FakeDb, similarity: float | None = None):
        self._db = db
        self.similarity = similarity  # text cosine given to candidates that carry none of their own
        self.caption_vectors: dict = {}
        self.verdict_cache: dict[tuple, bool] = {}
        self.last_modalities: tuple[str, ...] | None = None

    async def replace(self, document_id, chunks, vectors):
        stored = [replace(c, id=uuid4(), document_id=document_id) for c in chunks]
        ids = {c.ordinal: c.id for c in stored}  # the real store resolves parent_ordinal the same
        self._db.chunks[document_id] = [
            replace(c, parent_id=ids.get(c.parent_ordinal, c.parent_id)) for c in stored
        ]

    async def for_document(self, document_id):
        return [
            replace(c, document_id=c.document_id or document_id)
            for c in self._db.chunks.get(document_id, [])
        ]

    def _set(self, chunk_id, **changes):
        for chunks in self._db.chunks.values():
            for i, chunk in enumerate(chunks):
                if chunk.id == chunk_id:
                    chunks[i] = replace(chunk, **changes)

    async def pending_captions(self, document_id, modalities=None):
        return [
            c for c in self._db.chunks.get(document_id, [])
            if c.modality != "text" and c.text_status == "pending"
            and not (c.modality == "frame" and c.parent_id is not None)
            and c.origin != "context"
            and (not modalities or c.modality in modalities)
        ]  # fmt: skip

    async def requeue_failed_captions(self, document_id, modalities=None):
        failed = [
            c for c in self._db.chunks.get(document_id, [])
            if c.modality != "text" and c.text_status == "failed"
            and (not modalities or c.modality in modalities)
        ]  # fmt: skip
        for c in failed:
            self._set(c.id, text_status="pending")
        return len(failed)

    async def save_caption(self, chunk_id, content, index_text, vector):
        self._set(chunk_id, content=content, index_text=index_text, text_status="ready")
        self.caption_vectors[chunk_id] = vector
        for chunks in self._db.chunks.values():  # frames / clips under it inherit the caption
            for i, c in enumerate(chunks):
                if c.parent_id == chunk_id and c.modality == "frame":
                    chunks[i] = replace(c, content=content, index_text="", text_status="ready")

    async def set_context(self, chunk_id, text):
        self._set(chunk_id, content=text, index_text="", text_status="ready")

    async def fail_caption(self, chunk_id):
        self._set(chunk_id, text_status="failed")

    async def skip_caption(self, chunk_id):
        self._set(chunk_id, text_status="skipped")

    async def count_media(self, document_id, modalities=None):
        return sum(
            1 for c in self._db.chunks.get(document_id, [])
            if c.modality != "text" and (not modalities or c.modality in modalities)
        )  # fmt: skip

    async def asset_of(self, chunk_id):
        for document_id, chunks in self._db.chunks.items():
            for chunk in chunks:
                if chunk.id == chunk_id:
                    return document_id, chunk.asset_key
        return None

    def _measured(self, chunk):
        """Chunks keep cosines they were built with; the rest get the store-wide `similarity`."""
        if chunk.similarity is None and chunk.native_similarity is None:
            return replace(chunk, similarity=self.similarity)
        return chunk

    def _in(self, document_ids, max_page):
        return [
            self._measured(replace(c, document_id=c.document_id or document_id))
            for document_id in document_ids
            for c in self._db.chunks.get(document_id, [])
            if max_page is None or c.page_start is None or c.page_start <= max_page
        ]

    async def candidates(self, document_ids, query, vector, n, max_page=None, modalities=None):
        self.last_modalities = modalities
        found = self._in(document_ids, max_page)

        def allowed(c):
            return modalities is None or c.modality in modalities

        text_space = [c for c in found if c.modality == "text" or c.text_status == "ready"]
        return Candidates(
            [c for c in text_space if allowed(c)][:n],
            [],
            [c for c in found if c.modality in ("image", "frame") and allowed(c)][:n],
            [c for c in found if c.modality == "audio" and allowed(c)][:n],
        )

    async def media_matches(self, document_ids, query, vector, modalities, max_page=None):
        words = set(normalize_query(query).split())
        return [
            MediaMatch(c, bool(words & set(normalize_query(c.content).split())))
            for c in self._in(document_ids, max_page)
            if c.modality in modalities
        ]

    async def verdicts(self, chunk_ids, query_key):
        return {i: self.verdict_cache[(i, query_key)] for i in chunk_ids
                if (i, query_key) in self.verdict_cache}  # fmt: skip

    async def save_verdict(self, chunk_id, query_key, verdict):
        self.verdict_cache[(chunk_id, query_key)] = verdict


class FakeUnitOfWork:
    def __init__(self, db: FakeDb):
        self.chats = FakeChatRepository(db)
        self.docs = FakeDocumentRepository(db)
        self.chunks = FakeChunkStore(db)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None
