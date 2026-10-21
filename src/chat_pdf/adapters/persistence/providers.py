import logging
from typing import Any
from uuid import UUID

from cryptography.fernet import InvalidToken
from psycopg import errors
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from chat_pdf.adapters.persistence.connection import ConnectionSource
from chat_pdf.crypto import Crypto
from chat_pdf.domain.errors import Conflict, NotFound
from chat_pdf.domain.providers import ProviderKind, ProviderProfile

log = logging.getLogger(__name__)

PLAIN_FIELDS = (
    "kind", "name", "base_url", "model", "dimensions", "query_prefix", "document_prefix",
    "supports_tools", "default_k", "max_k", "max_tool_calls", "voice", "language", "protocol",
    "supports_vision", "modalities", "calibration",
)  # fmt: skip
COLUMNS = ", ".join(("id", "api_key_enc", *PLAIN_FIELDS))


def _db_value(column: str, value: Any) -> Any:
    return Jsonb(value) if column == "calibration" and value is not None else value


def _not_found(profile_id: UUID) -> NotFound:
    return NotFound("provider_not_found", f"Proveedor {profile_id} no encontrado")


class PgProviderRepository:
    """provider_profiles + app_settings. Implements the ProviderRepository port."""

    def __init__(self, source: ConnectionSource, crypto: Crypto):
        self._source = source
        self._crypto = crypto

    def _profile(self, row: dict[str, Any]) -> ProviderProfile:
        enc = row.pop("api_key_enc")
        try:
            key = self._crypto.decrypt(bytes(enc)) if enc else None
        except InvalidToken:  # APP_SECRET_KEY changed since the key was stored
            log.warning("api key of profile %s cannot be decrypted; re-enter it", row["id"])
            key = None
        return ProviderProfile(api_key=key, **row)

    async def create(self, data: dict[str, Any]) -> ProviderProfile:
        values = {k: _db_value(k, data[k]) for k in PLAIN_FIELDS if data.get(k) is not None}
        cols = [*values, "api_key_enc"]
        enc = self._crypto.encrypt(data["api_key"]) if data.get("api_key") else None
        sql = (
            f"insert into provider_profiles ({', '.join(cols)}) "
            f"values ({', '.join(['%s'] * len(cols))}) returning {COLUMNS}"
        )
        async with self._source.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, [*values.values(), enc])
            return self._profile(await cur.fetchone())

    async def list(self, kind: ProviderKind | None = None) -> list[ProviderProfile]:
        where, params = (" where kind = %s", [kind]) if kind else ("", [])
        sql = f"select {COLUMNS} from provider_profiles{where} order by created_at"
        async with self._source.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            return [self._profile(r) for r in await cur.fetchall()]

    async def get(self, profile_id: UUID) -> ProviderProfile | None:
        sql = f"select {COLUMNS} from provider_profiles where id = %s"
        async with self._source.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, [profile_id])
            row = await cur.fetchone()
        return self._profile(row) if row else None

    async def update(self, profile_id: UUID, changes: dict[str, Any]) -> ProviderProfile:
        """Apply only the keys present. `api_key` present but falsy clears the key."""
        sets = {k: _db_value(k, v) for k, v in changes.items() if k in PLAIN_FIELDS}
        if "api_key" in changes:
            key = changes["api_key"]
            sets["api_key_enc"] = self._crypto.encrypt(key) if key else None
        if not sets:
            profile = await self.get(profile_id)
            if profile is None:
                raise _not_found(profile_id)
            return profile
        assignments = ", ".join(f"{c} = %s" for c in sets)
        sql = f"update provider_profiles set {assignments} where id = %s returning {COLUMNS}"
        async with self._source.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, [*sets.values(), profile_id])
            row = await cur.fetchone()
        if row is None:
            raise _not_found(profile_id)
        return self._profile(row)

    async def delete(self, profile_id: UUID) -> None:
        try:
            async with self._source.connection() as conn:
                cur = await conn.execute(
                    "delete from provider_profiles where id = %s", [profile_id]
                )
        except errors.ForeignKeyViolation as exc:
            raise Conflict("provider_in_use", "Hay documentos indexados con este perfil") from exc
        if cur.rowcount == 0:
            raise _not_found(profile_id)

    async def in_use(self, profile_id: UUID) -> bool:
        async with self._source.connection() as conn:
            cur = await conn.execute(
                "select exists (select 1 from documents where embedding_profile_id = %s)",
                [profile_id],
            )
            return (await cur.fetchone())[0]

    async def active_ids(self) -> dict[str, UUID | None]:
        async with self._source.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "select active_chat_profile_id as chat, active_embedding_profile_id as embedding, "
                "active_stt_profile_id as stt, active_tts_profile_id as tts, "
                "active_vision_profile_id as vision from app_settings"
            )
            row = await cur.fetchone()
        return row or dict.fromkeys(("chat", "embedding", "stt", "tts", "vision"))

    async def active(self, kind: ProviderKind) -> ProviderProfile | None:
        profile_id = (await self.active_ids())[kind]
        return await self.get(profile_id) if profile_id else None

    async def set_active(self, kind: ProviderKind, profile_id: UUID | None) -> None:
        column = f"active_{kind}_profile_id"  # kind is a validated literal
        sql = (
            f"insert into app_settings (id, {column}) values (true, %s) "
            f"on conflict (id) do update set {column} = excluded.{column}"
        )
        async with self._source.connection() as conn:
            await conn.execute(sql, [profile_id])
