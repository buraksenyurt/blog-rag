from common.config import get_settings
from eval.metrics import hit, recall, reciprocal_rank, turkish_ratio, unique_posts
from eval.run_eval import VARIANTS, load_questions


def test_unique_posts_keeps_order() -> None:
    assert unique_posts(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_retrieval_metrics() -> None:
    assert recall(["a", "b"], ["x", "a"]) == 0.5
    assert hit(["a"], ["x", "a"]) == 1.0 and hit(["a"], ["x"]) == 0.0
    assert reciprocal_rank(["a", "b"], ["x", "b", "a"]) == 0.5
    assert reciprocal_rank(["a"], ["x"]) == 0.0
    assert recall([], ["a"]) == 0.0


def test_turkish_ratio_ignores_code() -> None:
    assert turkish_ratio("Bu bir örnek ve kod aşağıdaki gibi.\n```rust\nthe and is\n```") == 1.0
    assert turkish_ratio("This is the example and it is short.") == 0.0
    assert turkish_ratio("```rust\nlet x = 1;\n```") is None


def test_question_set_is_well_formed() -> None:
    questions = load_questions()
    assert len(questions) >= 25
    assert len({q["id"] for q in questions}) == len(questions)
    for q in questions:
        assert q["type"] in {"example", "relational", "structural"}
        assert q["question"].strip()
        if q["type"] != "structural":
            assert q["relevant"], q["id"]


def test_relevant_slugs_exist_in_posts() -> None:
    posts_dir = get_settings().posts_dir
    if not posts_dir.is_dir():
        import pytest

        pytest.skip("kaynak blog klasörü yok")
    for q in load_questions():
        for slug in q.get("relevant") or []:
            assert (posts_dir / f"{slug}.md").exists(), f"{q['id']}: {slug}"


def test_variants_are_valid_retrieval_overrides() -> None:
    from dataclasses import fields

    from phase1_vector.retrieve import RetrievalConfig

    names = {f.name for f in fields(RetrievalConfig)}
    for overrides in VARIANTS.values():
        assert set(overrides) <= names
    assert list(VARIANTS)[-1] == "full"


def test_run_retrieval_end_to_end_with_fake_store() -> None:
    from eval.run_eval import run_retrieval
    from phase1_vector.retrieve import Retriever
    from tests.test_retrieve import FakeEmbedder, add, make_client

    client = make_client()
    add(client, "rust-post#0000", [1.0, 0.1, 0.0], tags=["rust"], has_code=True)
    add(client, "other-post#0000", [1.0, 0.0, 0.0])
    questions = [
        {"id": "q1", "type": "example", "question": "Rust örnek", "relevant": ["rust-post"]},
        {"id": "q2", "type": "structural", "question": "kaç yazı", "relevant": []},
    ]
    report = run_retrieval(questions, Retriever(FakeEmbedder(), client), ["plain", "full"])
    assert report["summary"]["full"]["hit_ctx"] == 1.0
    assert report["summary"]["full"]["mrr"] == 1.0  # filtre boost'u rust yazısını öne alır
    assert report["summary"]["plain"]["mrr"] == 0.5
    assert report["questions"][1]["variants"]["full"]["hit_ctx"] == 0.0
