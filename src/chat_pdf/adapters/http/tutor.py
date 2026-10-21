import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from chat_pdf.application.tutor import (
    DEFAULT_COUNT,
    AnswerTutorQuestion,
    GenerateTutorQuestions,
    ListTutorQuestions,
    TutorQuestion,
)
from chat_pdf.domain.tutor import MAX_ANSWER_CHARS, MAX_QUESTIONS

router = APIRouter(prefix="/api/v1")


class GenerateBody(BaseModel):
    count: Annotated[int, Field(ge=1, le=MAX_QUESTIONS)] = DEFAULT_COUNT


class AnswerBody(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=MAX_ANSWER_CHARS)]


class QuestionOut(BaseModel):
    """The rubric stays hidden until the student answers: only how many points it has."""

    id: uuid.UUID
    question: str
    section: str | None
    page: int | None
    points: int
    due: bool
    due_at: datetime
    reps: int
    lapses: int
    last_score: float | None


class PointOut(BaseModel):
    point: str
    covered: bool
    student_quote: str | None
    source_quote: str
    page: int | None
    chunk_id: uuid.UUID | None


class GradeOut(BaseModel):
    question_id: uuid.UUID
    points: list[PointOut]
    score: float
    feedback: str
    follow_up_question: str
    due_at: datetime


def _out(q: TutorQuestion) -> QuestionOut:
    return QuestionOut(
        id=q.id, question=q.question, section=q.section, page=q.page, points=len(q.rubric),
        due=q.due_at <= datetime.now(UTC), due_at=q.due_at, reps=q.reps, lapses=q.lapses,
        last_score=q.last_score,
    )  # fmt: skip


def _uc(request: Request) -> Any:
    return request.app.state.use_cases


@router.post("/chats/{chat_id}/tutor/questions", status_code=201, response_model=list[QuestionOut])
async def generate_questions(
    chat_id: uuid.UUID, request: Request, body: GenerateBody | None = None
):
    use_case: GenerateTutorQuestions = _uc(request).generate_tutor_questions
    created = await use_case.execute(chat_id, (body or GenerateBody()).count)
    return [_out(q) for q in created]


@router.get("/chats/{chat_id}/tutor/questions", response_model=list[QuestionOut])
async def list_questions(chat_id: uuid.UUID, request: Request):
    use_case: ListTutorQuestions = _uc(request).list_tutor_questions
    return [_out(q) for q in await use_case.execute(chat_id)]


@router.post("/chats/{chat_id}/tutor/questions/{question_id}/answer", response_model=GradeOut)
async def answer_question(
    chat_id: uuid.UUID, question_id: uuid.UUID, body: AnswerBody, request: Request
):
    use_case: AnswerTutorQuestion = _uc(request).answer_tutor_question
    return await use_case.execute(chat_id, question_id, body.text)
