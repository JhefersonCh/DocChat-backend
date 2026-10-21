from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from chat_pdf.application.artifacts import Artifact


class FakeArtifactRepository:
    def __init__(self):
        self.items: dict = {}

    async def create(self, chat_id, kind, options):
        a = Artifact(uuid4(), chat_id, kind, "pending", options, None, None, datetime.now(UTC))
        self.items[a.id] = a
        return a

    async def get(self, artifact_id):
        return self.items.get(artifact_id)

    async def for_chat(self, chat_id):
        return [a for a in self.items.values() if a.chat_id == chat_id]

    async def finish(self, artifact_id, data, error):
        status = "ready" if error is None else "failed"
        self.items[artifact_id] = replace(
            self.items[artifact_id], status=status, data=data, error=error
        )

    async def update_data(self, artifact_id, data):
        self.items[artifact_id] = replace(self.items[artifact_id], data=data)

    async def delete(self, artifact_id, chat_id):
        a = self.items.get(artifact_id)
        if a is None or a.chat_id != chat_id:
            return False
        del self.items[artifact_id]
        return True

    async def fail_pending(self, error):
        for a in list(self.items.values()):
            if a.status == "pending":
                await self.finish(a.id, None, error)
