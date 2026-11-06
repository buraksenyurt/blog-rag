from pathlib import Path

import pytest

from common.config import ROOT, Settings, ensure_writable, load_embedding_profile
from common.embeddings import Embedder


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


def test_default_profile_is_embeddinggemma(settings: Settings) -> None:
    profile = load_embedding_profile(settings)
    assert profile.name == "embeddinggemma"
    assert profile.dim == 768


def test_unknown_profile_raises(settings: Settings) -> None:
    with pytest.raises(ValueError):
        load_embedding_profile(settings, "yok-boyle-bir-profil")


def test_templates_applied_without_network(settings: Settings) -> None:
    embedder = Embedder(settings, load_embedding_profile(settings))
    assert embedder.format_document("let x = 5;", "Rust") == "title: Rust | text: let x = 5;"
    assert embedder.format_query("soru") == "task: search result | query: soru"
    assert embedder.format_document("{braces}") == "title: none | text: {braces}"


def test_prefixless_profile(settings: Settings) -> None:
    embedder = Embedder(settings, load_embedding_profile(settings, "bge-m3"))
    assert embedder.format_document("metin", "Başlık") == "metin"
    assert embedder.format_query("soru") == "soru"


def test_source_repo_is_protected(settings: Settings) -> None:
    with pytest.raises(PermissionError):
        ensure_writable(settings.posts_dir / "deneme.md", settings)
    with pytest.raises(PermissionError):
        ensure_writable(settings.source_repo_root / "README.md", settings)


def test_data_dir_is_writable(settings: Settings) -> None:
    assert ensure_writable(settings.data_dir, settings) == (ROOT / "data").resolve()
    assert ensure_writable(Path(ROOT / "eval" / "results"), settings)
