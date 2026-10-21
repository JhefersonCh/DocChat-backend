import json
from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg.rows import class_row
from psycopg.types.json import Jsonb

from chat_pdf.adapters.persistence.connection import ConnectionSource
from chat_pdf.application.tutor import TutorQuestion
from chat_pdf.domain.tutor import Sched

COLUMNS = (
    "id, chat_id, question, rubric, section, page, due_at, ease, interval_days, reps, lapses, "
    "last_score, created_at"
)


def _jsonb(value: Any) -> Jsonb:
    return Jsonb(value, dumps=lambda obj: json.dumps(obj, default=str))


class PgTutorRepository:
    """tutor_questions + tutor_attempts. Implements the TutorRepository port."""

    def __init__(self, source: ConnectionSource):
        self._source = source

    async def add_questions(self, chat_id: UUID, items: list[dict]) -> list[TutorQuestion]:
        out = []
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(TutorQuestion))
            for item in items:
                first = item["rubric"][0]
                await cur.execute(
                    f"""insert into tutor_questions (chat_id, question, rubric, section, page)
                        values (%s, %s, %s, %s, %s) returning {COLUMNS}""",
                    (chat_id, item["question"], _jsonb(item["rubric"]),
                     first.get("section"), first.get("page")),
                )  # fmt: skip
                out.append(await cur.fetchone())
        return out

    async def questions(self, chat_id: UUID) -> list[TutorQuestion]:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(TutorQuestion))
            await cur.execute(
                f"select {COLUMNS} from tutor_questions where chat_id = %s order by created_at, id",
                (chat_id,),
            )
            return await cur.fetchall()

    async def get(self, question_id: UUID, chat_id: UUID) -> TutorQuestion | None:
        async with self._source.connection() as conn:
            cur = conn.cursor(row_factory=class_row(TutorQuestion))
            await cur.execute(
                f"select {COLUMNS} from tutor_questions where id = %s and chat_id = %s",
                (question_id, chat_id),
            )
            return await cur.fetchone()

    async def record_attempt(
        self, question_id: UUID, answer: str, result: dict[str, Any], sched: Sched, due: datetime
    ) -> None:
        async with self._source.connection() as conn:
            await conn.execute(
                """insert into tutor_attempts
                       (question_id, answer, points, score, feedback, follow_up)
                   values (%s, %s, %s, %s, %s, %s)""",
                (question_id, answer, _jsonb(result["points"]), result["score"],
                 result["feedback"], result["follow_up_question"]),
            )  # fmt: skip
            await conn.execute(
                """update tutor_questions set due_at = %s, ease = %s, interval_days = %s,
                       reps = %s, lapses = %s, last_score = %s where id = %s""",
                (due, sched.ease, sched.interval_days, sched.reps, sched.lapses,
                 result["score"], question_id),
            )  # fmt: skip
