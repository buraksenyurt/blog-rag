from types import SimpleNamespace

from common.config import Settings
from common.llm import LLM, parse_json_answer


def event(content=None, reasoning=None, empty=False):
    if empty:
        return SimpleNamespace(choices=[])
    delta = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class FakeCompletions:
    def __init__(self, events=None, content="") -> None:
        self.events = events or []
        self.content = content
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        if kwargs.get("stream"):
            return iter(self.events)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_llm(completions: FakeCompletions) -> LLM:
    llm = LLM(Settings(_env_file=None), model="m")
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return llm


def test_llm_exposes_all_methods() -> None:
    # Adım 9'daki bir düzenleme stream()'i yanlışlıkla sınıfın dışına itmişti.
    for name in ("chat", "chat_json", "stream"):
        assert callable(getattr(LLM, name, None)), name


def test_stream_yields_content_and_counts_reasoning() -> None:
    completions = FakeCompletions(
        [event(reasoning="düşün"), event(empty=True), event("Mer"), event("haba")]
    )
    llm = make_llm(completions)
    assert list(llm.stream([{"role": "user", "content": "x"}])) == ["Mer", "haba"]
    assert llm.last_reasoning_chars == len("düşün")
    assert completions.kwargs["stream"] is True and completions.kwargs["model"] == "m"


def test_chat_and_chat_json() -> None:
    assert make_llm(FakeCompletions(content="cevap")).chat([]) == "cevap"
    completions = FakeCompletions(content='```json\n{"concepts": []}\n```')
    assert make_llm(completions).chat_json([], {"type": "object"}) == {"concepts": []}
    assert completions.kwargs["response_format"]["type"] == "json_schema"


def test_parse_json_answer_rejects_missing_json() -> None:
    import pytest

    with pytest.raises(ValueError):
        parse_json_answer("JSON yok")
