"""Uygulama ayarları (.env) ve embedding profilleri (embedding_profiles.yaml)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class EmbeddingProfile(BaseModel):
    name: str
    model: str
    dim: int
    max_tokens: int
    doc_template: str = "{text}"
    query_template: str = "{text}"

    @property
    def collection_suffix(self) -> str:
        return self.name


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    lmstudio_url: str = "http://localhost:1234/v1"
    lmstudio_api_key: str = "lm-studio"
    llm_model: str = "meta-llama-3-8b-instruct"

    embedding_profile: str | None = None
    embedding_profiles_file: Path = Path("embedding_profiles.yaml")

    posts_dir: Path = Path("../buraksenyurt.github.io/_posts")
    exclude_categories: list[str] = ["Foto İpucu"]

    qdrant_url: str = "http://localhost:6333"

    data_dir: Path = Path("data")

    @field_validator("posts_dir", "data_dir", "embedding_profiles_file")
    @classmethod
    def _resolve_relative_to_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else (ROOT / value).resolve()

    @property
    def source_repo_root(self) -> Path:
        """Salt okunur kaynak reponun kökü (_posts klasörünün üstü)."""
        return self.posts_dir.parent


def load_embedding_profile(settings: Settings, name: str | None = None) -> EmbeddingProfile:
    raw = yaml.safe_load(settings.embedding_profiles_file.read_text(encoding="utf-8"))
    profiles: dict = raw.get("profiles") or {}
    selected = name or settings.embedding_profile or raw.get("active")
    if selected not in profiles:
        available = ", ".join(profiles) or "-"
        raise ValueError(f"Embedding profili bulunamadı: {selected!r} (mevcut: {available})")
    return EmbeddingProfile(name=selected, **profiles[selected])


def ensure_writable(path: Path, settings: Settings | None = None) -> Path:
    """Yazılacak bir yolun salt okunur kaynak repo içinde olmadığını garanti eder."""
    settings = settings or get_settings()
    target = path.resolve()
    protected = settings.source_repo_root.resolve()
    if target == protected or protected in target.parents:
        raise PermissionError(f"Kaynak repo salt okunurdur, yazma engellendi: {target}")
    return target


@lru_cache
def get_settings() -> Settings:
    return Settings()
