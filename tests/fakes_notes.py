from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from chat_pdf.application.notes import Note


class FakeNoteRepository:
    def __init__(self):
        self.items: dict = {}

    async def create(self, chat_id, message_id, title, content, citations):
        n = Note(uuid4(), chat_id, message_id, title, content, citations, datetime.now(UTC))
        self.items[n.id] = n
        return n

    async def for_chat(self, chat_id):
        return [n for n in self.items.values() if n.chat_id == chat_id]

    async def for_message(self, chat_id, message_id):
        return next(
            (n for n in self.items.values() if n.chat_id == chat_id and n.message_id == message_id),
            None,
        )

    async def set_title(self, note_id, chat_id, title):
        n = self.items.get(note_id)
        if n is None or n.chat_id != chat_id:
            return None
        self.items[note_id] = replace(n, title=title)
        return self.items[note_id]

    async def delete(self, note_id, chat_id):
        n = self.items.get(note_id)
        if n is None or n.chat_id != chat_id:
            return False
        del self.items[note_id]
        return True
