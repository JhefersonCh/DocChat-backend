import asyncio
from uuid import uuid4

import pytest
from fakes import FakeChunkStore, FakeDb, FakeDocumentRepository

from chat_pdf.application.inspector import ListChunks
from chat_pdf.domain.errors import NotFound
from chat_pdf.domain.models import Chunk


def _chunks(n=5):
    return [
        Chunk(i, f"texto {i}", 1 + i // 2, 1 + (i + 1) // 2, i * 10, i * 10 + 8, uuid4(), "Sec")
        for i in range(n)
    ]  # chunk i spans pages [1 + i//2, 1 + (i+1)//2]


def _setup():
    db = FakeDb()
    doc = db.add_document(db.add_chat().id, uuid4())
    db.chunks[doc.id] = _chunks()
    return doc.id, ListChunks(FakeDocumentRepository(db), FakeChunkStore(db))


def test_lists_all_chunks_with_total_and_paginates():
    doc_id, use_case = _setup()
    total, items = asyncio.run(use_case.execute(doc_id, None, 0, 2))
    assert total == 5 and [c.ordinal for c in items] == [0, 1]
    _, rest = asyncio.run(use_case.execute(doc_id, None, 4, 10))
    assert [c.ordinal for c in rest] == [4]


def test_page_filter_keeps_chunks_overlapping_the_page():
    doc_id, use_case = _setup()
    total, items = asyncio.run(use_case.execute(doc_id, 2, 0, 50))
    assert total == 3 and [c.ordinal for c in items] == [1, 2, 3]


def test_unknown_document_is_not_found():
    _, use_case = _setup()
    with pytest.raises(NotFound):
        asyncio.run(use_case.execute(uuid4(), None, 0, 10))
