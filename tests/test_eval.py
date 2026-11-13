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
        assert q.get("difficulty", "normal") in {"normal", "hard"}
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


def test_question_set_has_hard_questions() -> None:
    hard = [q for q in load_questions() if q.get("difficulty") == "hard"]
    assert len(hard) >= 10
    assert all(q["type"] != "structural" and q["relevant"] for q in hard)


class _Hit:
    def __init__(self, slug: str) -> None:
        self.payload = {
            "id": f"{slug}#0000",
            "title": slug,
            "date": "2022-01-01",
            "section_path": "",
            "text": "metin",
            "url": "u",
        }
        self.post_slug = slug
        self.extras: list = []

    @property
    def text(self) -> str:
        return self.payload["text"]


class _Retriever:
    def retrieve(self, question: str):
        class Result:
            hits = [_Hit("beklenen-yazi"), _Hit("diger-yazi")]

        return Result()


class _FlakyLLM:
    """İlk soruda zaman aşımı fırlatır, sonrakilerde stream eder."""

    model = "fake-llm"

    def __init__(self) -> None:
        self.calls = 0

    def stream(self, messages, temperature=0.2, max_tokens=1024):
        import openai

        self.calls += 1
        if self.calls == 1:
            raise openai.APITimeoutError(request=None)
        yield "Cevap "
        yield "[#1]."

    def chat(self, *args, **kwargs):  # run_llm stream kullanmalı
        raise AssertionError("run_llm chat() yerine stream() kullanmalı")


def test_run_llm_continues_after_timeout_and_saves_each_row() -> None:
    from eval.run_eval import run_llm, summarize_answers

    questions = [
        {"id": "q1", "type": "example", "question": "a", "relevant": ["beklenen-yazi"]},
        {"id": "q2", "type": "example", "question": "b", "relevant": ["beklenen-yazi"]},
    ]
    saved: list[int] = []
    rows = run_llm(questions, _Retriever(), _FlakyLLM(), save=lambda r: saved.append(len(r)))
    assert saved == [1, 2]
    assert rows[0]["error"].startswith("APITimeoutError") and rows[0]["answer"] == ""
    assert rows[1]["error"] is None and rows[1]["answer"] == "Cevap [#1]."
    assert rows[1]["cited"] == [1] and rows[1]["sources"][0]["relevant"] is True
    assert rows[1]["first_token_seconds"] is not None
    summary = summarize_answers(rows)
    assert summary["answered"] == 1 and summary["errors"] == 1 and summary["citation_rate"] == 1.0


def test_answers_markdown_marks_errors(tmp_path) -> None:
    from eval.run_eval import write_answers_markdown

    row = {
        "id": "q1",
        "type": "example",
        "difficulty": "hard",
        "question": "a",
        "expected": None,
        "error": "APITimeoutError: timed out",
    }
    path = tmp_path / "a.md"
    write_answers_markdown([row], path, "m", "p")
    text = path.read_text(encoding="utf-8")
    assert "## q1 (example, zor)" in text and "**HATA:** APITimeoutError" in text
