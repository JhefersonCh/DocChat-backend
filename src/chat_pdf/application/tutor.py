"""Socratic tutor: generate questions with a verified rubric, grade answers, schedule reviews."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID

from chat_pdf.application.chats import require_chat
from chat_pdf.application.generation import INVALID_RESPONSE, _require_chat_profile
from chat_pdf.application.ports import (
    ChatRepository,
    ChunkStore,
    DocumentRepository,
    ProviderFactory,
    ProviderRepository,
)
from chat_pdf.application.sources import readable_chunks, readable_documents
from chat_pdf.domain.errors import Conflict, InvalidInput, NotFound, ProviderError
from chat_pdf.domain.tutor import (
    MAX_ANSWER_CHARS,
    MAX_QUESTIONS,
    Sched,
    due_at,
    grade,
    grade_messages,
    next_schedule,
    order_for_review,
    parse_grade,
    parse_questions,
    question_messages,
)

DEFAULT_COUNT = 5


@dataclass(frozen=True)
class TutorQuestion:
    id: UUID
    chat_id: UUID
    question: str
    rubric: list[dict[str, Any]]
    section: str | None
    page: int | None
    due_at: datetime
    ease: float
    interval_days: float
    reps: int
    lapses: int
    last_score: float | None
    created_at: datetime

    @property
    def sched(self) -> Sched:
        return Sched(self.ease, self.interval_days, self.reps, self.lapses)


class TutorRepository(Protocol):
    async def add_questions(self, chat_id: UUID, items: list[dict]) -> list[TutorQuestion]: ...

    async def questions(self, chat_id: UUID) -> list[TutorQuestion]: ...

    async def get(self, question_id: UUID, chat_id: UUID) -> TutorQuestion | None: ...

    async def record_attempt(
        self,
        question_id: UUID,
        answer: str,
        result: dict[str, Any],
        sched: Sched,
        due: datetime,
    ) -> None:
        """Stores the attempt and the question's new schedule/last score together."""
        ...


class GenerateTutorQuestions:
    def __init__(
        self,
        chats: ChatRepository,
        docs: DocumentRepository,
        chunks: ChunkStore,
        providers: ProviderRepository,
        factory: ProviderFactory,
        tutor: TutorRepository,
    ):
        self._chats, self._docs, self._chunks = chats, docs, chunks
        self._providers, self._factory, self._tutor = providers, factory, tutor

    async def execute(self, chat_id: UUID, count: int = DEFAULT_COUNT) -> list[TutorQuestion]:
        count = max(1, min(count, MAX_QUESTIONS))
        await require_chat(self._chats, chat_id)
        if not await readable_documents(self._chats, self._docs, chat_id):
            raise Conflict("document_not_ready", "El documento no está listo")
        profile = await _require_chat_profile(self._providers)
        chunks = await readable_chunks(self._chats, self._docs, self._chunks, chat_id)
        if not chunks:
            raise ProviderError("El documento no tiene fragmentos", "document_empty")
        existing = [q.question for q in await self._tutor.questions(chat_id)]
        result = await self._factory.chat(profile).complete(
            question_messages(chunks, count, existing)
        )
        items = parse_questions(result.content or "", chunks, existing)[:count]
        if not items:
            raise ProviderError("El modelo no devolvió preguntas válidas", INVALID_RESPONSE)
        return await self._tutor.add_questions(chat_id, items)


class ListTutorQuestions:
    def __init__(self, chats: ChatRepository, tutor: TutorRepository):
        self._chats, self._tutor = chats, tutor

    async def execute(self, chat_id: UUID) -> list[TutorQuestion]:
        """In review order: due ones first (failed before the rest), then upcoming."""
        await require_chat(self._chats, chat_id)
        return order_for_review(await self._tutor.questions(chat_id), datetime.now(UTC))


class AnswerTutorQuestion:
    def __init__(
        self,
        providers: ProviderRepository,
        factory: ProviderFactory,
        tutor: TutorRepository,
    ):
        self._providers, self._factory, self._tutor = providers, factory, tutor

    async def execute(self, chat_id: UUID, question_id: UUID, text: str) -> dict[str, Any]:
        answer = text.strip()
        if not answer or len(answer) > MAX_ANSWER_CHARS:
            raise InvalidInput("invalid_answer", "La respuesta está vacía o es demasiado larga")
        question = await self._tutor.get(question_id, chat_id)
        if question is None:
            raise NotFound("tutor_question_not_found", "Pregunta no encontrada")
        profile = await _require_chat_profile(self._providers)
        raw = await self._factory.chat(profile).complete(
            grade_messages(question.question, question.rubric, answer)
        )
        judgment = parse_grade(raw.content or "", len(question.rubric))
        if judgment is None:
            raise ProviderError("El modelo no devolvió una evaluación válida", INVALID_RESPONSE)
        result = grade(question.rubric, judgment, answer)
        now = datetime.now(UTC)
        sched = next_schedule(question.sched, result["score"])
        await self._tutor.record_attempt(question.id, answer, result, sched, due_at(now, sched))
        return {**result, "question_id": question.id, "due_at": due_at(now, sched)}
