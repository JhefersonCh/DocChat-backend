from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Infrastructure settings only. Model/provider config lives in the DB."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    app_secret_key: str
    cors_origins: str = "http://localhost:3000"
    upload_dir: str = "./uploads"
    # Outbound HTTP timeouts (seconds)
    provider_timeout_s: float = 120.0
    provider_connect_timeout_s: float = 5.0
    health_probe_timeout_s: float = 3.0
    # Seed default only (plan §7.2): base_url of the local embeddings profile created on first boot.
    seed_embedding_base_url: str = "http://embeddings:8080/v1"
    # Same for the local TTS profile (compose service `tts`).
    seed_tts_base_url: str = "http://tts:7788/v1"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
