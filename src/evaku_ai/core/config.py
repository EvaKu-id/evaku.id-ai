from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Pydantic reserves the "model_" prefix; these settings need it.
        protected_namespaces=("settings_",),
    )

    app_env: Literal["local", "production"] = "local"
    log_level: str = "INFO"

    model_api_key: SecretStr | None = None

    data_version: str = "dev"
    model_version: str = "dev"

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    region_config: Path = Path("config/regions/example.yaml")

    supabase_db_url: SecretStr | None = None

    r2_endpoint: str | None = None
    r2_access_key_id: SecretStr | None = None
    r2_secret_access_key: SecretStr | None = None
    r2_bucket_public: str = "evaku-maps"
    r2_bucket_private: str = "evaku-private"

    next_base_url: str | None = None
    pipeline_secret: SecretStr | None = None

    healthchecks_url: str | None = None
    gee_service_account: str | None = None
    gee_key_file: Path | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
