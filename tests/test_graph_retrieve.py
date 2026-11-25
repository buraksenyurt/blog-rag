import pytest

from common.config import EmbeddingProfile
from common.prompts import build_messages
from common.query_filters import QueryFilter
from eval.run_eval import GRAPH_VARIANTS, structural_ok
from phase2_graph.retrieve import (
    GraphRetrievalConfig,
    GraphRetriever,
    lowercase_filter,
    to_cosine,
)
from phase2_graph.structural import StructuralIntent, answer_structural, detect_structural
from tests.test_graph_store import FakeRunner

SERIES = ["Rust Pratikleri", "WCF", "GoLang", "Ruby Kod Parçacıkları"]


@pytest.mark.parametrize(
    ("question", "kind", "detail"),
    [
        ("Blogda Rust ile ilgili kaç yazı var?", "count", ["rust", "rust-lang"]),
        ("Go dili ile ilgili yazılar hangi yıllarda yayınlanmış?", "years", ["golang"]),
        ("Rust Pratikleri serisinde hangi yazılar var?", "series", "Rust Pratikleri"),
        ("Docker ile ilgili hangi yazılar var?", "list", ["docker"]),
    ],
)
def test_detect_structural(question: str, kind: str, detail) -> None:
    intent = detect_structural(question, SERIES)
    assert intent is not None and intent.kind == kind
    assert (intent.series if kind == "series" else intent.tags) == detail


@pytest.mark.parametrize(
    "question",
    [
        "Rust dilinde değişken kullanımları ile ilgili örnekler yaz.",
        "Rust'ta kaç tür akıllı pointer vardır?",  # 'kaç' var ama yazı sayısı sorulmuyor
        "WCF servislerinde binding türleri nelerdir?",  # seri adı geçiyor ama 'seri' yok
        "Bu blogda kaç yazı var?",  # teknoloji yok -> şablon yok
    ],
)
def test_detect_structural_ignores_normal_questions(question: str) -> None:
    assert detect_structural(question, SERIES) is None


def test_answer_structural_templates() -> None:
    run = FakeRunner(
        {
            "count(p) AS n": [[{"n": 46, "first": 2020, "last": 2026}]],
            "WITH p.year AS year": [[{"year": 2017, "n": 23}, {"year": 2018, "n": 1}]],
            "{name: $name}": [
                [
                    {"slug": "a", "title": "Loglama", "date": "2022-01-30", "position": 1},
                    {"slug": "b", "title": "Channels", "date": "2022-02-20", "position": 2},
                ]
            ],
        }
    )
    count = answer_structural(run, StructuralIntent("count", tags=["rust", "rust-lang"]))
    assert count.facts == ["Blogda rust veya rust-lang tag'li 46 yazı var (2020–2026)."]
    years = answer_structural(run, StructuralIntent("years", tags=["golang"]))
    assert years.facts == ["golang tag'li 24 yazının yıllara göre dağılımı: 2017: 23, 2018: 1."]
    series = answer_structural(run, StructuralIntent("series", series="Rust Pratikleri"))
    assert series.posts == ["a", "b"]
    assert (
        "'Rust Pratikleri' serisi 2 yazıdan oluşuyor (2022-01-30 → 2022-02-20)" in series.facts[0]
    )


def test_answer_structural_returns_none_without_results() -> None:
    run = FakeRunner({"count(p) AS n": [[{"n": 0, "first": None, "last": None}]]})
    assert answer_structural(run, StructuralIntent("count", tags=["yok"])) is None


def test_to_cosine_and_lowercase_filter() -> None:
    assert to_cosine(1.0) == 1.0 and to_cosine(0.5) == 0.0
    qf = lowercase_filter(QueryFilter(["grpc"], ["gRPC"], ["protobuf"]))
    assert qf.tags == ["grpc"] and qf.code_langs == ["protobuf"]


PROFILE = EmbeddingProfile(name="fake", model="fake", dim=3, max_tokens=100)


class FakeEmbedder:
    profile = PROFILE

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def row(
    cid: str,
    score: float,
    vector,
    *,
    tags=(),
    has_code=False,
    n_tokens=100,
    reasons=(),
    series=None,
    position=None,
) -> dict:
    slug, index = cid.split("#")
    return {
        "chunk": {
            "id": cid,
            "post_slug": slug,
            "text": cid,
            "section_path": "",
            "chunk_index": int(index),
            "n_tokens": n_tokens,
            "has_code": has_code,
            "code_langs": ["rust"] if has_code else [],
        },
        "embedding": list(vector),
        "post": {
            "slug": slug,
            "title": slug.title(),
            "url": "u",
            "year": 2022,
            "date": "2022-01-01",
        },
        "tags": list(tags),
        "series": series,
        "position": position,
        "score": score,
        "reasons": list(reasons),
    }


def make_retriever(responses: dict) -> tuple[GraphRetriever, FakeRunner]:
    responses = {"MATCH (s:Series) RETURN": [[{"name": n} for n in SERIES]], **responses}
    run = FakeRunner(responses)
    # Testler index yolunu kullanır; tam cosine yolu ayrıca test ediliyor.
    return GraphRetriever(FakeEmbedder(), run, GraphRetrievalConfig(n=3, exact_search=False)), run


def test_graph_retriever_merges_filtered_and_related_candidates() -> None:
    retriever, run = make_retriever(
        {
            "db.index.vector.queryNodes": [[row("csharp#0000", 0.90, [1, 0.1, 0])]],
            "ft.name IN $tags": [
                [
                    row(
                        "rust#0000",
                        0.88,
                        [1, 0.3, 0],
                        tags=["rust"],
                        series="Rust Pratikleri",
                        position=3,
                    )
                ]
            ],
            "UNWIND $seeds": [
                [
                    row(
                        "rust2#0000",
                        0.86,
                        [1, 0, 0.4],
                        tags=["rust"],
                        reasons=["seri: Rust Pratikleri"],
                    )
                ]
            ],
        }
    )
    result = retriever.retrieve("Rust örnekleri")
    ids = [h.id for h in result.candidates]
    assert ids[0] == "rust#0000"  # +0.05 filtre boost'u csharp'ı geçirir
    assert set(ids) == {"csharp#0000", "rust#0000", "rust2#0000"}
    related = next(h for h in result.candidates if h.id == "rust2#0000")
    assert related.graph_bonus == 0.02 and related.reasons == ["seri: Rust Pratikleri"]
    assert related.score == pytest.approx(0.72)  # Neo4j (1+cos)/2 -> cos
    top = result.candidates[0]
    assert top.payload["context_note"] == "Rust Pratikleri serisi, 3. yazı"
    seeds = run.queries("UNWIND $seeds")[0]["seeds"]
    assert seeds == ["rust", "csharp"]  # related sorgusu ilk iki aday yazıdan başlar
    assert run.queries("ft.name IN $tags")[0]["tags"] == ["rust", "rust-lang"]


def test_graph_retriever_without_expansion_skips_related_query() -> None:
    retriever, run = make_retriever(
        {"db.index.vector.queryNodes": [[row("a#0000", 0.9, [1, 0, 0])]]}
    )
    retriever.retrieve("soru", graph_expand=False, use_filter=False)
    assert run.queries("UNWIND $seeds") == []


def test_graph_retriever_adds_structural_facts() -> None:
    retriever, _ = make_retriever(
        {
            "db.index.vector.queryNodes": [[row("rust#0000", 0.9, [1, 0, 0], tags=["rust"])]],
            "count(p) AS n": [[{"n": 46, "first": 2020, "last": 2026}]],
        }
    )
    result = retriever.retrieve("Blogda Rust ile ilgili kaç yazı var?", graph_expand=False)
    assert result.facts == ["Blogda rust veya rust-lang tag'li 46 yazı var (2020–2026)."]
    off = retriever.retrieve("Blogda Rust ile ilgili kaç yazı var?", structural=False)
    assert off.facts == []


def test_short_hit_is_expanded_via_next_relation() -> None:
    nxt = row("a#0001", 0.0, [0, 1, 0], n_tokens=300)
    retriever, _ = make_retriever(
        {
            "db.index.vector.queryNodes": [[row("a#0000", 0.9, [1, 0, 0], n_tokens=50)]],
            "-[:NEXT]->(n:Chunk)": [[{"chunk": nxt["chunk"], "post": nxt["post"]}]],
        }
    )
    hit = retriever.retrieve("soru", use_filter=False, graph_expand=False).hits[0]
    assert [e["id"] for e in hit.extras] == ["a#0001"]


def test_prompt_includes_graph_facts() -> None:
    messages = build_messages("kaç yazı?", [], ["Blogda 46 yazı var."])
    assert (
        "GRAPH BİLGİSİ" in messages[1]["content"]
        and "- Blogda 46 yazı var." in (messages[1]["content"])
    )


def test_structural_ok_compares_expected_numbers() -> None:
    q = {"type": "structural", "expected": "golang tag'li 27 yazı; 2017 (23), 2018 (1)"}
    assert structural_ok(q, ["golang tag'li 27 yazının dağılımı: 2017: 23, 2018: 1."])
    assert structural_ok(q, ["golang tag'li 26 yazı: 2017: 23, 2018: 1."]) is False
    assert structural_ok(q, []) is False
    assert structural_ok({"type": "example", "expected": None}, ["x"]) is None


def test_graph_variants_are_valid_overrides() -> None:
    from dataclasses import fields

    names = {f.name for f in fields(GraphRetrievalConfig)}
    assert all(set(v) <= names for v in GRAPH_VARIANTS.values())


def test_exact_search_uses_full_scan_instead_of_index() -> None:
    retriever, run = make_retriever(
        {"MATCH (c:Chunk)\nWITH c, vector.similarity": [[row("a#0000", 0.9, [1, 0, 0])]]}
    )
    result = retriever.retrieve("soru", exact_search=True, use_filter=False, graph_expand=False)
    assert [h.id for h in result.candidates] == ["a#0000"]
    assert run.queries("db.index.vector.queryNodes") == []


def test_exact_search_is_the_default_entry() -> None:
    assert GraphRetrievalConfig().exact_search is True
    assert GRAPH_VARIANTS["vector"]["exact_search"] is False


def test_concept_expansion_adds_mentioning_chunks() -> None:
    retriever, run = make_retriever(
        {
            "db.index.vector.queryNodes": [[row("a#0000", 0.9, [1, 0, 0])]],
            "RETURN k.name AS name, k.aliases": [
                [
                    {"name": "shadowing", "aliases": ["gölgeleme"], "df": 1},
                    {"name": "rust", "aliases": [], "df": 40},  # merkez kavram, kullanılmaz
                ]
            ],
            "WHERE k.name IN $names": [
                [row("kimdir#0009", 0.85, [1, 0.2, 0], reasons=["kavram: shadowing"])]
            ],
        }
    )
    result = retriever.retrieve(
        "Rust'ta gölgeleme nedir?", concept_expand=True, use_filter=False, graph_expand=False
    )
    assert run.queries("WHERE k.name IN $names")[0]["names"] == ["shadowing"]
    hit = next(h for h in result.candidates if h.id == "kimdir#0009")
    assert hit.graph_bonus == 0.03 and hit.reasons == ["kavram: shadowing"]
    assert retriever.query_concepts("rust nedir", max_df=8) == []


def test_concept_candidates_are_limited_to_query_technology() -> None:
    retriever, run = make_retriever(
        {
            "db.index.vector.queryNodes": [[row("go#0000", 0.9, [1, 0, 0], tags=["golang"])]],
            "RETURN k.name AS name, k.aliases": [[{"name": "channel", "aliases": [], "df": 3}]],
        }
    )
    retriever.retrieve("Go'da channel örneği", concept_expand=True, graph_expand=False)
    params = run.queries("WHERE k.name IN $names")[0]
    assert params["names"] == ["channel"] and params["tags"] == ["golang"]
