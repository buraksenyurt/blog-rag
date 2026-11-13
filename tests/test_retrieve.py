from qdrant_client import QdrantClient, models

from common.config import EmbeddingProfile
from phase1_vector.retrieve import RetrievalConfig, Retriever
from phase1_vector.store import collection_name, ensure_collection, point_id

PROFILE = EmbeddingProfile(name="fake", model="fake", dim=3, max_tokens=100)
QUERY_VECTOR = [1.0, 0.0, 0.0]


class FakeEmbedder:
    profile = PROFILE

    def embed_query(self, text: str) -> list[float]:
        return QUERY_VECTOR


def add(client: QdrantClient, cid: str, vector: list[float], **payload) -> None:
    slug = cid.split("#")[0]
    data = {
        "id": cid,
        "post_slug": slug,
        "title": slug,
        "date": "2022-01-01",
        "url": "u",
        "section_path": "",
        "n_tokens": 100,
        "tags": [],
        "code_langs": [],
        "text": cid,
    }
    data.update(payload)
    client.upsert(
        collection_name(PROFILE),
        [models.PointStruct(id=point_id(cid), vector=vector, payload=data)],
    )


def make_client() -> QdrantClient:
    client = QdrantClient(":memory:")
    ensure_collection(client, PROFILE)
    return client


def test_filter_boost_promotes_matching_chunk() -> None:
    client = make_client()
    add(client, "csharp-post#0000", [1.0, 0.10, 0.0], tags=["csharp"])
    add(client, "rust-post#0000", [1.0, 0.25, 0.0], tags=["rust"], code_langs=["rust"])
    retriever = Retriever(FakeEmbedder(), client, RetrievalConfig(n=2))

    boosted = retriever.retrieve("Rust örnekleri")
    assert boosted.hits[0].payload["id"] == "rust-post#0000"
    assert boosted.hits[0].boosted and boosted.filtered_count == 1

    plain = retriever.retrieve("Rust örnekleri", use_filter=False)
    assert plain.hits[0].payload["id"] == "csharp-post#0000"


def test_filtered_search_reaches_beyond_top_k() -> None:
    client = make_client()
    for i in range(5):
        add(client, f"c{i}#0000", [1.0, 0.01 * i, 0.0], tags=["csharp"])
    add(client, "rust-post#0000", [0.6, 0.8, 0.0], tags=["rust"])
    result = Retriever(FakeEmbedder(), client, RetrievalConfig(k=3)).retrieve("rust")
    assert "rust-post#0000" in {h.payload["id"] for h in result.candidates}


def test_max_per_post_and_token_budget() -> None:
    client = make_client()
    for i in range(5):
        add(client, f"same-post#{i:04d}", [1.0, 0.01 * i, 0.0])
    add(client, "other-post#0000", [1.0, 0.3, 0.0])
    cfg = RetrievalConfig(n=6, max_per_post=2, use_filter=False)
    result = Retriever(FakeEmbedder(), client, cfg).retrieve("soru")
    slugs = [h.post_slug for h in result.hits]
    assert slugs.count("same-post") == 2 and "other-post" in slugs

    small = Retriever(FakeEmbedder(), client, RetrievalConfig(token_budget=250, use_filter=False))
    assert len(small.retrieve("soru").hits) == 2


def test_mmr_prefers_diverse_chunk() -> None:
    client = make_client()
    add(client, "a#0000", [1.0, 0.0, 0.0])
    add(client, "b#0000", [1.0, 0.001, 0.0])  # a ile neredeyse aynı
    add(client, "c#0000", [0.9, 0.0, 0.45])  # biraz daha az alakalı ama farklı
    cfg = RetrievalConfig(n=2, use_filter=False, mmr_lambda=0.5)
    ids = [h.payload["id"] for h in Retriever(FakeEmbedder(), client, cfg).retrieve("q").hits]
    assert ids == ["a#0000", "c#0000"]


def test_code_boost_prefers_code_chunk_for_example_queries() -> None:
    client = make_client()
    add(client, "intro#0000", [1.0, 0.10, 0.0], has_code=False)
    add(client, "code#0000", [1.0, 0.20, 0.0], has_code=True)
    retriever = Retriever(FakeEmbedder(), client, RetrievalConfig(n=1, use_filter=False))
    assert retriever.retrieve("örnek kod yaz").hits[0].payload["id"] == "code#0000"
    assert retriever.retrieve("bu nedir").hits[0].payload["id"] == "intro#0000"


def test_short_textual_hit_is_expanded_with_next_chunk() -> None:
    client = make_client()
    add(client, "post#0000", [1.0, 0.0, 0.0], chunk_index=0, n_tokens=60, has_code=False)
    add(
        client,
        "post#0001",
        [0.0, 1.0, 0.0],
        chunk_index=1,
        n_tokens=300,
        has_code=True,
        text="```rust\nlet x = 5;\n```",
    )
    retriever = Retriever(FakeEmbedder(), client, RetrievalConfig(n=1, use_filter=False))
    hit = retriever.retrieve("q").hits[0]
    assert [e["id"] for e in hit.extras] == ["post#0001"]
    assert "let x = 5;" in hit.text and hit.total_tokens == 360

    no_expand = retriever.retrieve("q", expand_short=False).hits[0]
    assert no_expand.extras == []
    tight = retriever.retrieve("q", token_budget=100).hits[0]
    assert tight.extras == []
