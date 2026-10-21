import asyncio
from uuid import uuid4

import pytest

from chat_pdf.adapters.storage.local import LocalFileStorage
from chat_pdf.domain.errors import InvalidInput


async def _blocks(*parts: bytes):
    for part in parts:
        yield part


def test_new_uploads_live_in_a_per_source_folder(tmp_path):
    storage, doc = LocalFileStorage(tmp_path), uuid4()

    asyncio.run(storage.save(doc, _blocks(b"ab", b"cd"), ".pdf"))

    assert storage.path(doc) == tmp_path / str(doc) / "original.pdf"
    assert storage.path(doc).read_bytes() == b"abcd"


def test_flat_legacy_file_is_still_found(tmp_path):
    storage, doc = LocalFileStorage(tmp_path), uuid4()
    (tmp_path / str(doc)).write_bytes(b"old")

    assert storage.path(doc) == tmp_path / str(doc)
    assert storage.path(doc).read_bytes() == b"old"


def test_unknown_document_resolves_to_a_missing_file(tmp_path):
    assert not LocalFileStorage(tmp_path).path(uuid4()).is_file()


def test_assets_and_thumbs_are_stored_beside_the_original(tmp_path):
    storage, doc = LocalFileStorage(tmp_path), uuid4()
    asyncio.run(storage.save(doc, _blocks(b"x"), ".pdf"))

    asyncio.run(storage.save_asset(doc, "assets", "s1.webp", b"big"))
    asyncio.run(storage.save_asset(doc, "thumbs", "s1.webp", b"small"))

    assert storage.asset_path(doc, "assets", "s1.webp").read_bytes() == b"big"
    assert storage.asset_path(doc, "thumbs", "s1.webp").read_bytes() == b"small"
    assert storage.path(doc).name == "original.pdf"  # assets do not shadow the original


@pytest.mark.parametrize("name", ["../x.webp", "a/b.webp", "", ".hidden", "/etc/passwd"])
def test_asset_names_cannot_escape_the_folder(tmp_path, name):
    with pytest.raises(InvalidInput):
        LocalFileStorage(tmp_path).asset_path(uuid4(), "assets", name)


def test_delete_removes_folder_or_legacy_file_and_tolerates_missing(tmp_path):
    storage, folder_doc, flat_doc = LocalFileStorage(tmp_path), uuid4(), uuid4()
    asyncio.run(storage.save(folder_doc, _blocks(b"x"), ".pdf"))
    asyncio.run(storage.save_asset(folder_doc, "thumbs", "t.webp", b"t"))
    (tmp_path / str(flat_doc)).write_bytes(b"old")

    for doc in (folder_doc, flat_doc, uuid4()):
        asyncio.run(storage.delete(doc))

    assert list(tmp_path.iterdir()) == []
