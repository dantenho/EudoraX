from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "EudoraX Backend"
    app_env: str = Field(default="development", alias="APP_ENV")
    host: str = "0.0.0.0"
    port: int = 8000

    huggingface_model: str = "distilbert/distilbert-base-uncased"
    qdrant_url: str = "http://localhost:6333"
    milvus_uri: str = "http://localhost:19530"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
