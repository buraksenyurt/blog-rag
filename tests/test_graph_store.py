import pytest

from common.config import EmbeddingProfile
from phase2_graph.graph_model import build_graph_data
from phase2_graph.store import (
    CHUNK_BATCH,
    check_meta,
    delete_stale,
    ensure_schema,
    vector_index_query,
    write_chunks,
    write_posts,
)
from tests.test_graph_model import chunk

PROFILE = EmbeddingProfile(name="embeddinggemma", model="m", dim=768, max_tokens=2048)


class FakeRunner:
    """Neo4j yerine sorguları ve parametreleri kaydeder; istenen cevapları sırayla döner."""

    def __init__(self, responses: dict[str, list] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.responses = responses or {}

    def __call__(self, query: str, **params) -> list[dict]:
        self.calls.append((query, params))
        for marker, answers in self.responses.items():
            if marker in query and answers:
                return answers.pop(0)
        return []

    def queries(self, marker: str) -> list[dict]:
        return [p for q, p in self.calls if marker in q]


def test_vector_index_query_uses_profile_dimension() -> None:
    query = vector_index_query(768)
    assert "`vector.dimensions`: 768" in query and "'cosine'" in query
    with pytest.raises(ValueError):
        vector_index_query(0)


def test_ensure_schema_creates_constraints_and_vector_index() -> None:
    run = FakeRunner()
    ensure_schema(run, PROFILE)
    queries = [q for q, _ in run.calls]
    assert any("REQUIRE p.slug IS UNIQUE" in q for q in queries)
    assert queries[-1].startswith("CREATE VECTOR INDEX chunk_embedding")


def test_check_meta_rejects_other_profile() -> None:
    check_meta(FakeRunner(), PROFILE)  # meta yok -> sorun yok
    same = FakeRunner({"GraphMeta": [[{"profile": "embeddinggemma", "dim": 768}]]})
    check_meta(same, PROFILE)
    other = FakeRunner({"GraphMeta": [[{"profile": "bge-m3", "dim": 1024}]]})
    with pytest.raises(ValueError, match="--recreate"):
        check_meta(other, PROFILE)


def test_write_posts_clears_old_relations_before_linking() -> None:
    data = build_graph_data([chunk("p1", 0), chunk("p2", 0)], {"p1": ["p2"]})
    run = FakeRunner()
    write_posts(run, data)
    clear_at = next(i for i, (q, _) in enumerate(run.calls) if "DELETE r" in q)
    tag_at = next(i for i, (q, _) in enumerate(run.calls) if ":TAGGED]" in q and "MERGE" in q)
    assert clear_at < tag_at
    assert run.queries("LINKS_TO]->(b)")[0]["rows"] == [{"src": "p1", "dst": "p2"}]


def test_write_chunks_attaches_vectors_in_batches() -> None:
    chunks = [chunk("p1", i) for i in range(CHUNK_BATCH + 5)]
    data = build_graph_data(chunks, {})
    asked: list[int] = []

    def vectors_for(rows: list[dict]) -> list[list[float]]:
        asked.append(len(rows))
        return [[float(r["chunk_index"])] * 3 for r in rows]

    run = FakeRunner()
    write_chunks(run, data, vectors_for)
    assert asked == [CHUNK_BATCH, 5]
    written = [row for p in run.queries("SET c.post_slug") for row in p["rows"]]
    assert len(written) == CHUNK_BATCH + 5
    assert written[3]["embedding"] == [3.0, 3.0, 3.0]
    assert sum(len(p["rows"]) for p in run.queries("MERGE (a)-[:NEXT]->(b)")) == CHUNK_BATCH + 4


def test_delete_stale_repeats_until_nothing_left() -> None:
    data = build_graph_data([chunk("p1", 0)], {})
    run = FakeRunner(
        {
            "NOT c.id IN $ids": [[{"deleted": 10000}], [{"deleted": 3}], [{"deleted": 0}]],
            "NOT p.slug IN $slugs": [[{"deleted": 2}]],
            "NOT (n)--()": [[{"deleted": 5}]],
        }
    )
    assert delete_stale(run, data) == {"chunks": 10003, "posts": 2, "orphans": 5}


def test_vector_index_quantization_is_disabled_by_default() -> None:
    assert "`vector.quantization.enabled`: false" in vector_index_query(768)
    assert "`vector.quantization.enabled`: true" in vector_index_query(768, quantization=True)


def test_rebuild_vector_index_drops_creates_and_waits() -> None:
    from phase2_graph.store import rebuild_vector_index

    run = FakeRunner()
    rebuild_vector_index(run, PROFILE, quantization=False)
    queries = [q for q, _ in run.calls]
    assert queries[0].startswith("DROP INDEX chunk_embedding")
    assert queries[1].startswith("CREATE VECTOR INDEX") and "enabled`: false" in queries[1]
    assert "awaitIndexes" in queries[2]
