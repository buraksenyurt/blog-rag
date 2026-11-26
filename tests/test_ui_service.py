import json
from pathlib import Path

import openai

from common.config import EmbeddingProfile, Settings
from common.retrieval import Hit, RetrievalConfig, RetrievalResult, candidate_rows
from phase2_graph.retrieve import GraphRetrievalConfig
from ui.service import (
    Engine,
    QueryOutcome,
    ServiceStatus,
    UISettings,
    compare_contexts,
    configure,
    example_questions,
    run_query,
    source_rows,
    timing,
)

PROFILE = EmbeddingProfile(name="fake", model="text-embedding-fake", dim=3, max_tokens=100)


def hit(slug: str, i: int = 0, **extra) -> Hit:
    payload = {
        "id": f"{slug}#{i:04d}",
        "post_slug": slug,
        "title": slug.title(),
        "date": "2022-01-01",
        "url": f"u/{slug}",
        "section_path": "",
        "text": "metin",
        "n_tokens": 100,
        "code_langs": [],
        **extra,
    }
    h = Hit(id=payload["id"], score=0.8, payload=payload, vector=[1.0, 0, 0], final_score=0.85)
    return h


class FakeRetriever:
    def __init__(self, config: RetrievalConfig, hits: list[Hit], facts=None, fail=False) -> None:
        self.config = config
        self.hits = hits
        self.facts = facts or []
        self.fail = fail
        self.calls = 0

    def retrieve(self, question: str) -> RetrievalResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("Neo4j kapalı")
        return RetrievalResult(
            question, None, False, 0, list(self.hits), list(self.hits), facts=list(self.facts)
        )


class FakeLLM:
    model = "fake-llm"
    last_reasoning_chars = 0

    def __init__(self, answer: str = "Cevap [#1].", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error

    def stream(self, messages, temperature=0.2, max_tokens=1024):
        if self.error:
            raise self.error
        yield from self.answer.split(" ")


def make_engine(tmp_path: Path, retrievers: dict, llm: FakeLLM) -> Engine:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    return Engine(
        settings, PROFILE, retriever_factory=lambda m: retrievers[m], llm_factory=lambda model: llm
    )


def test_mode_available_reasons() -> None:
    status = ServiceStatus(
        lmstudio=True,
        models=["text-embedding-fake", "gemma"],
        embedding_model="text-embedding-fake",
        qdrant=True,
        neo4j=False,
        neo4j_error="bağlantı yok",
    )
    assert status.chat_models == ["gemma"]
    assert status.mode_available("vector") == (True, "")
    ok, reason = status.mode_available("graph")
    assert not ok and "Neo4j" in reason
    assert not ServiceStatus(lmstudio=False, lmstudio_error="x").mode_available("vector")[0]
    no_embed = ServiceStatus(lmstudio=True, models=["gemma"], embedding_model="e", qdrant=True)
    assert "Embedding" in no_embed.mode_available("vector")[1]


def test_configure_applies_only_matching_fields() -> None:
    ui = UISettings(n=4, budget=2000, graph_expand=False, concept_expand=True)
    vector = configure(RetrievalConfig(), ui)
    assert (vector.n, vector.token_budget) == (4, 2000)
    graph = configure(GraphRetrievalConfig(), ui)
    assert graph.graph_expand is False and graph.concept_expand is True and graph.n == 4


def test_run_query_streams_answer_and_logs(tmp_path: Path) -> None:
    retriever = FakeRetriever(RetrievalConfig(), [hit("a"), hit("b")])
    engine = make_engine(tmp_path, {"vector": retriever}, FakeLLM())
    tokens: list[str] = []
    shown: list[RetrievalResult] = []
    outcome = run_query(
        engine,
        "vector",
        "soru",
        UISettings(n=2),
        "fake-llm",
        on_token=tokens.append,
        on_retrieval=shown.append,
    )
    assert outcome.error is None and outcome.warning is None
    assert retriever.calls == 1  # retrieval bir kez yapılır, cevap aynı sonucu kullanır
    assert tokens == ["Cevap", "[#1]."] and len(shown) == 1
    assert outcome.result.cited == [1] and outcome.result.messages[1]["role"] == "user"
    assert [r["id"] for r in outcome.rows] == ["a#0000", "b#0000"]
    assert retriever.config.n == 2
    log = next((tmp_path / "runs").glob("ask-*.jsonl")).read_text(encoding="utf-8")
    assert json.loads(log.splitlines()[0])["phase"] == "ui-vector"


def test_run_query_reports_retrieval_and_llm_errors(tmp_path: Path) -> None:
    failing = FakeRetriever(GraphRetrievalConfig(), [], fail=True)
    engine = make_engine(tmp_path, {"graph": failing}, FakeLLM())
    outcome = run_query(engine, "graph", "soru", UISettings(), "m", log=False)
    assert outcome.error.startswith("Retrieval hatası: RuntimeError")

    timeout = FakeLLM(error=openai.APITimeoutError(request=None))
    engine = make_engine(
        tmp_path, {"vector": FakeRetriever(RetrievalConfig(), [hit("a")])}, timeout
    )
    outcome = run_query(engine, "vector", "soru", UISettings(), "m", log=False)
    assert "saniyede cevap vermedi" in outcome.error and outcome.rows


def test_run_query_warns_on_empty_answer(tmp_path: Path) -> None:
    engine = make_engine(
        tmp_path, {"vector": FakeRetriever(RetrievalConfig(), [hit("a")])}, FakeLLM(answer="")
    )
    outcome = run_query(engine, "vector", "soru", UISettings(), "m", log=False)
    assert outcome.warning.startswith("Boş cevap")


def test_sources_timing_and_compare(tmp_path: Path) -> None:
    vector_retriever = FakeRetriever(RetrievalConfig(), [hit("a"), hit("b")])
    graph_retriever = FakeRetriever(
        GraphRetrievalConfig(), [hit("b", context_note="X serisi"), hit("c")], facts=["46 yazı"]
    )
    engine = make_engine(
        tmp_path, {"vector": vector_retriever, "graph": graph_retriever}, FakeLLM()
    )
    v = run_query(engine, "vector", "soru", UISettings(), "m", log=False)
    g = run_query(engine, "graph", "soru", UISettings(), "m", log=False)
    sources = source_rows(v.result)
    assert sources[0]["cited"] and not sources[1]["cited"] and sources[0]["url"] == "u/a"
    assert source_rows(g.result)[0]["note"] == "X serisi"
    assert timing(v.result)["context_tokens"] == 200
    diff = compare_contexts(v, g)
    assert diff["common"] == [("b", "B")]
    assert diff["only_vector"] == [("a", "A")] and diff["only_graph"] == [("c", "C")]
    assert diff["graph_facts"] == ["46 yazı"]
    assert compare_contexts(QueryOutcome("vector", "q"), g)["only_graph"] == [
        ("b", "B"),
        ("c", "C"),
    ]


def test_candidate_rows_mark_context_and_extras() -> None:
    a, b = hit("a"), hit("b")
    a.extras = [{"id": "a#0001", "n_tokens": 50, "text": "x"}]
    b.graph_bonus, b.reasons = 0.02, ["seri: X"]
    rows = candidate_rows(RetrievalResult("q", None, False, 0, [a, b], [a]))
    assert rows[0]["context"] == 1 and rows[0]["extras"] == ["a#0001"]
    assert rows[1]["context"] is None and rows[1]["boosts"] == "G"
    assert rows[1]["reasons"] == ["seri: X"]


def test_example_questions_come_from_eval_set() -> None:
    examples = example_questions()
    assert [e["label"] for e in examples].count("Yapısal") == 2
    assert all(e["question"] for e in examples)
    assert example_questions(picks=[("x", "yok")]) == []  # bilinmeyen id atlanır
