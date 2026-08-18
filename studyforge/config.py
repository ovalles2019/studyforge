from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_base_url: str = "https://api.fireworks.ai/inference/v1"
    llm_api_key: str = ""
    llm_model: str = "accounts/fireworks/models/qwen3-235b-a22b"

    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_path: str = str(ROOT / "data" / "chroma")
    collection_name: str = "studyforge"
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieve_k: int = 5

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key.strip())


def get_settings() -> Settings:
    return Settings()
