from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from common.config import EmbeddingProfile
from common.embedding_cache import EmbeddingCache
from phase1_vector.ingest import embedding_title, ingest
from phase1_vector.store import collection_name, point_id

PROFILE = EmbeddingProfile(
    name="fake",
    model="fake-model",
    dim=4,
    max_tokens=100,
    doc_template="title: {title} | text: {text}",
    query_template="q: {text}",
)


class FakeEmbedder:
    """Ağa çıkmadan deterministik vektör üretir; kaç metin embed edildiğini sayar."""

    def __init__(self) -> None:
        self.profile = PROFILE
        self.batch_size = 2
        self.calls = 0

    def format_document(self, text: str, title: str | None = None) -> str:
        return PROFILE.doc_template.format(text=text, title=title or "none")

    def embed_raw(self, inputs: list[str]) -> list[list[float]]:
        self.calls += len(inputs)
        return [[float(len(t)), 1.0, float(i), 0.5] for i, t in enumerate(inputs)]


def chunk(i: int, slug: str = "2022-01-01-a") -> dict:
    return {
        "id": f"{slug}#{i:04d}",
        "post_slug": slug,
        "title": "A",
        "section_path": "B",
        "text": f"metin {i}",
        "year": 2022,
        "tags": ["rust"],
        "categories": [],
        "code_langs": [],
        "has_code": False,
        "chunk_index": i,
    }


@pytest.fixture
def setup(tmp_path: Path):
    return FakeEmbedder(), EmbeddingCache(tmp_path / "c.sqlite", PROFILE), QdrantClient(":memory:")


def test_point_id_is_deterministic() -> None:
    assert point_id("x#0001") == point_id("x#0001") != point_id("x#0002")


def test_embedding_title_includes_section() -> None:
    assert embedding_title(chunk(0)) == "A > B"
    assert embedding_title({**chunk(0), "section_path": ""}) == "A"


def test_ingest_is_idempotent_and_uses_cache(setup) -> None:
    embedder, cache, client = setup
    chunks = [chunk(i) for i in range(5)]
    first = ingest(chunks, embedder, cache, client)
    assert (first.embedded, first.cached) == (5, 0)
    second = ingest(chunks, embedder, cache, client)
    assert (second.embedded, second.cached) == (0, 5)
    assert embedder.calls == 5
    assert client.count(collection_name(PROFILE)).count == 5


def test_stale_points_are_deleted(setup) -> None:
    embedder, cache, client = setup
    ingest([chunk(i) for i in range(5)], embedder, cache, client)
    report = ingest([chunk(i) for i in range(3)], embedder, cache, client)
    assert report.deleted_stale == 2
    assert client.count(collection_name(PROFILE)).count == 3


def test_payload_filter_fields(setup) -> None:
    embedder, cache, client = setup
    ingest([chunk(0)], embedder, cache, client)
    point = client.retrieve(collection_name(PROFILE), [point_id(chunk(0)["id"])])[0]
    assert point.payload["tags"] == ["rust"] and point.payload["text"] == "metin 0"


class TruncatingEmbedder(FakeEmbedder):
    """İlk 50 karakterden sonrasını görmeyen model taklidi."""

    def embed_raw(self, inputs: list[str]) -> list[list[float]]:
        return [[float(len(t[:50])), float(100 - len(t[:50])), 1.0, 1.0] for t in inputs]


def test_truncation_probe() -> None:
    from phase1_vector.ingest import is_truncated

    assert is_truncated("x" * 80, TruncatingEmbedder())
    assert not is_truncated("kısa", TruncatingEmbedder())
