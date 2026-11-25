"""Soru cevaplama akışı (retrieval -> prompt -> LLM -> atıf kontrolü). Faz 1 ve Faz 2 ortak."""

from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Protocol

from common.config import Settings, ensure_writable
from common.llm import LLM
from common.prompts import build_messages, build_no_rag_messages, extract_citations
from common.retrieval import Hit, RetrievalConfig, RetrievalResult


class Retriever(Protocol):
    """Faz 1 (Qdrant) ve Faz 2 (Neo4j) retriever'larının ortak arayüzü."""

    def retrieve(self, query: str) -> RetrievalResult: ...


@dataclass
class AskResult:
    question: str
    model: str
    use_rag: bool
    answer: str
    hits: list[Hit] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0
    first_token_seconds: float | None = None  # stream'de ilk token'a kadar geçen süre (prefill)


def answer_question(
    question: str,
    llm: LLM,
    retriever: Retriever | None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    on_token: Callable[[str], None] | None = None,
) -> AskResult:
    hits: list = []
    facts: list[str] = []
    started = time.perf_counter()
    if retriever is not None:
        retrieval = retriever.retrieve(question)
        hits, facts = retrieval.hits, list(getattr(retrieval, "facts", []) or [])
        messages = build_messages(question, hits, facts)
    else:
        messages = build_no_rag_messages(question)
    retrieval_seconds = time.perf_counter() - started

    started = time.perf_counter()
    first_token_seconds: float | None = None
    if on_token is not None:
        parts: list[str] = []
        for part in llm.stream(messages, temperature=temperature, max_tokens=max_tokens):
            if first_token_seconds is None:
                first_token_seconds = round(time.perf_counter() - started, 2)
            parts.append(part)
            on_token(part)
        answer = "".join(parts)
    else:
        answer = llm.chat(messages, temperature=temperature, max_tokens=max_tokens)
    generation_seconds = time.perf_counter() - started

    cited, invalid = extract_citations(answer, len(hits))
    return AskResult(
        question=question,
        model=llm.model,
        use_rag=retriever is not None,
        answer=answer,
        hits=hits,
        facts=facts,
        cited=cited,
        invalid_citations=invalid,
        retrieval_seconds=round(retrieval_seconds, 2),
        generation_seconds=round(generation_seconds, 2),
        first_token_seconds=first_token_seconds,
    )


def empty_answer_reason(llm: LLM) -> str:
    """Model hiç cevap metni üretmediğinde kullanıcıya gösterilecek açıklama."""
    reasoning = getattr(llm, "last_reasoning_chars", 0) or 0
    detail = f" ({reasoning} karakter düşünme metni üretti)" if reasoning else ""
    return (
        f"Boş cevap{detail}: model çıktısını düşünme (reasoning) bölümüne yazmış ya da "
        "max_tokens dolmuş olabilir. Thinking modeli yerine Instruct sürümünü kullanın."
    )


def print_sources(result: AskResult) -> None:
    if not result.use_rag:
        return
    print("\n\nKaynaklar")
    print("---------")
    for fact in result.facts:
        print(f"  [graph] {fact}")
    for number, hit in enumerate(result.hits, start=1):
        p = hit.payload
        mark = "✓" if number in result.cited else " "
        section = p.get("section_path") or "Giriş"
        print(f"{mark} [#{number}] {p['title']} ({p['date']}) — {section}")
        print(f"         {p['url']}")
    if not result.cited:
        print("\n! Cevapta hiç [#n] atfı yok. Model bağlam talimatını izlememiş olabilir.")
    if result.invalid_citations:
        print(f"\n! Bağlamda olmayan atıf numaraları: {result.invalid_citations}")


def log_run(
    result: AskResult,
    settings: Settings,
    profile_name: str,
    cfg: RetrievalConfig,
    phase: int = 1,
) -> None:
    runs_dir = ensure_writable(settings.data_dir / "runs", settings)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"ask-{dt.date.today().isoformat()}.jsonl"
    record = {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "question": result.question,
        "model": result.model,
        "embedding_profile": profile_name,
        "use_rag": result.use_rag,
        "retrieval": asdict(cfg),
        "hits": [
            {
                "id": h.payload["id"],
                "score": round(h.score, 4),
                "final": round(h.final_score, 4),
                "extras": [e["id"] for e in h.extras],
            }
            for h in result.hits
        ],
        "facts": result.facts,
        "answer": result.answer,
        "cited": result.cited,
        "invalid_citations": result.invalid_citations,
        "retrieval_seconds": result.retrieval_seconds,
        "generation_seconds": result.generation_seconds,
        "first_token_seconds": result.first_token_seconds,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def retrieval_config(
    n: int | None = None, budget: int | None = None, base: RetrievalConfig | None = None
) -> RetrievalConfig:
    """CLI'dan gelen --n/--budget değerleriyle retrieval ayarlarını günceller."""
    overrides = {}
    if n is not None:
        overrides["n"] = n
    if budget is not None:
        overrides["token_budget"] = budget
    return replace(base or RetrievalConfig(), **overrides)
