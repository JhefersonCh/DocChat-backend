import asyncio
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

from chat_pdf.application.ports import AssetKind
from chat_pdf.domain.errors import InvalidInput

ORIGINAL = "original"
PLAYABLE = "play.mp4"  # browser-playable transcode of a video, next to the original


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class LocalFileStorage:
    """Implements the FileStorage port on a local directory.

    Layout (plan §1.4): `{dir}/{id}/original.{ext}`, `{id}/assets/*`, `{id}/thumbs/*`. Files
    uploaded before it are flat (`{dir}/{id}`) and are served as they are, never migrated.
    """

    def __init__(self, directory: Path):
        self._dir = directory

    def _folder(self, document_id: UUID) -> Path:
        return self._dir / str(document_id)

    def path(self, document_id: UUID) -> Path:
        folder = self._folder(document_id)
        if folder.is_dir():
            return next(iter(sorted(folder.glob(f"{ORIGINAL}.*"))), folder / ORIGINAL)
        return folder  # legacy flat file (or nothing: callers check is_file())

    def playable_path(self, document_id: UUID) -> Path:
        return self._folder(document_id) / PLAYABLE

    async def save(self, document_id: UUID, content: AsyncIterator[bytes], ext: str = "") -> None:
        folder = self._folder(document_id)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f"{ORIGINAL}{ext.lower()}").open("wb") as out:
            async for block in content:
                out.write(block)

    async def delete(self, document_id: UUID) -> None:
        folder = self._folder(document_id)

        def remove() -> None:
            if folder.is_dir():
                shutil.rmtree(folder, ignore_errors=True)
            else:
                folder.unlink(missing_ok=True)

        await asyncio.to_thread(remove)

    async def clear_assets(self, document_id: UUID) -> None:
        def remove() -> None:
            for kind in ("assets", "thumbs"):
                shutil.rmtree(self._folder(document_id) / kind, ignore_errors=True)

        await asyncio.to_thread(remove)

    def asset_path(self, document_id: UUID, kind: AssetKind, name: str) -> Path:
        if not name or name != Path(name).name or name.startswith("."):
            raise InvalidInput("invalid_asset_name", "Nombre de archivo derivado inválido")
        return self._folder(document_id) / kind / name

    async def save_asset(self, document_id: UUID, kind: AssetKind, name: str, data: bytes) -> None:
        await asyncio.to_thread(_write, self.asset_path(document_id, kind, name), data)
