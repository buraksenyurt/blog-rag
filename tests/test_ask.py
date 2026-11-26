from dataclasses import dataclass, field

from common.answering import answer_question
from common.prompts import SYSTEM_PROMPT, build_messages, extract_citations, format_context


@dataclass
class FakeHit:
    payload: dict
    extras: list = field(default_factory=list)
    score: float = 0.8
    final_score: float = 0.8

    @property
    def text(self) -> str:
        return self.payload["text"]


def hit(n: int, section: str = "Değişkenler") -> FakeHit:
    return FakeHit(
        {
            "id": f"p#{n}",
            "title": f"Yazı {n}",
            "date": "2022-04-17",
            "section_path": section,
            "text": f"let x{n} = {n};",
            "url": "u",
        }
    )


class FakeRetriever:
    def retrieve(self, question: str):
        class Result:
            hits = [hit(1), hit(2, section="")]

        return Result()


class FakeLLM:
    model = "fake-llm"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.messages: list[dict] = []

    def chat(self, messages, temperature=0.2, max_tokens=1024) -> str:
        self.messages = messages
        return self.answer

    def stream(self, messages, temperature=0.2, max_tokens=1024):
        self.messages = messages
        yield self.answer[:5]
        yield self.answer[5:]


def test_context_is_numbered_with_title_and_section() -> None:
    context = format_context([hit(1), hit(2, section="")])
    assert context.startswith("[#1] Yazı 1 (2022-04-17) — Değişkenler\nlet x1 = 1;")
    assert "[#2] Yazı 2 (2022-04-17) — Giriş" in context


def test_messages_contain_rules_question_and_context() -> None:
    messages = build_messages("Rust değişkenleri?", [hit(1)])
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "Türkçe" in messages[0]["content"]
    assert "SORU: Rust değişkenleri?" in messages[1]["content"]
    assert "[#1] Yazı 1" in messages[1]["content"]


def test_extract_citations() -> None:
    assert extract_citations("A [#1]. B [#3][#1]. C [#9]", 3) == ([1, 3], [9])
    assert extract_citations("atıf yok", 3) == ([], [])


def test_answer_question_with_rag() -> None:
    llm = FakeLLM("Rust'ta değişkenler değişmezdir [#1].")
    result = answer_question("soru", llm, FakeRetriever())
    assert result.use_rag and result.cited == [1]
    assert "BAĞLAM" in llm.messages[1]["content"]


def test_answer_question_streaming_and_no_rag() -> None:
    llm = FakeLLM("bağlamsız cevap")
    received: list[str] = []
    result = answer_question("soru", llm, None, on_token=received.append)
    assert result.answer == "bağlamsız cevap" and received == ["bağla", "msız cevap"]
    assert result.first_token_seconds is not None
    assert not result.use_rag and result.hits == []
    assert "BAĞLAM" not in llm.messages[1]["content"]


def test_retrieval_config_overrides() -> None:
    from common.answering import retrieval_config

    default = retrieval_config()
    assert (default.n, default.token_budget) == (6, 3500)
    small = retrieval_config(n=4, budget=2000)
    assert (small.n, small.token_budget) == (4, 2000)


def test_llm_uses_configured_timeout_without_retries() -> None:
    from common.config import Settings
    from common.llm import LLM

    llm = LLM(Settings(_env_file=None, llm_timeout=42))
    assert llm._client.timeout == 42 and llm._client.max_retries == 0


def test_answer_question_reuses_given_retrieval() -> None:
    class Result:
        hits = [hit(1)]
        facts = ["graph bilgisi"]

    class NeverRetrieve:
        def retrieve(self, question):
            raise AssertionError("retrieval tekrar yapılmamalı")

    llm = FakeLLM("cevap [#1]")
    result = answer_question(
        "soru", llm, NeverRetrieve(), retrieval=Result(), retrieval_seconds=1.5
    )
    assert result.retrieval_seconds == 1.5 and result.cited == [1]
    assert "GRAPH BİLGİSİ" in result.messages[1]["content"]
