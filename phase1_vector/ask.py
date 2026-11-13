"""Faz 1 uçtan uca: soru -> retrieval -> LLM -> cevap + kaynaklar.

Kullanım:
    uv run python -m phase1_vector.ask "Rust dilinde değişken kullanımları ile ilgili örnekler yaz"
    uv run python -m phase1_vector.ask "..." --debug            # retrieval tablosunu da göster
    uv run python -m phase1_vector.ask "..." --no-rag           # bağlamsız cevap (kıyas için)
    uv run python -m phase1_vector.ask "..." --model qwen3-8b   # başka bir LLM
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace

from openai import APIError, APITimeoutError
from qdrant_client import QdrantClient

from common.config import Settings, ensure_writable, get_settings, load_embedding_profile
from common.embeddings import Embedder
from common.llm import LLM
from common.prompts import build_messages, build_no_rag_messages, extract_citations
from phase1_vector.retrieve import Hit, RetrievalConfig, Retriever, print_result


@dataclass
class AskResult:
    question: str
    model: str
    use_rag: bool
    answer: str
    hits: list[Hit] = field(default_factory=list)
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
    hits: list[Hit] = []
    started = time.perf_counter()
    if retriever is not None:
        hits = retriever.retrieve(question).hits
        messages = build_messages(question, hits)
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
        cited=cited,
        invalid_citations=invalid,
        retrieval_seconds=round(retrieval_seconds, 2),
        generation_seconds=round(generation_seconds, 2),
        first_token_seconds=first_token_seconds,
    )


def print_sources(result: AskResult) -> None:
    if not result.use_rag:
        return
    print("\n\nKaynaklar")
    print("---------")
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


def log_run(result: AskResult, settings: Settings, profile_name: str, cfg: RetrievalConfig) -> None:
    runs_dir = ensure_writable(settings.data_dir / "runs", settings)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"ask-{dt.date.today().isoformat()}.jsonl"
    record = {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
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
        "answer": result.answer,
        "cited": result.cited,
        "invalid_citations": result.invalid_citations,
        "retrieval_seconds": result.retrieval_seconds,
        "generation_seconds": result.generation_seconds,
        "first_token_seconds": result.first_token_seconds,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def retrieval_config(n: int | None = None, budget: int | None = None) -> RetrievalConfig:
    """CLI'dan gelen --n/--budget değerleriyle varsayılan retrieval ayarlarını günceller."""
    overrides = {}
    if n is not None:
        overrides["n"] = n
    if budget is not None:
        overrides["token_budget"] = budget
    return replace(RetrievalConfig(), **overrides)


def main() -> int:
    parser = argparse.ArgumentParser(description="Faz 1 RAG: soru sor")
    parser.add_argument("question")
    parser.add_argument("--model", help="LM Studio model id (varsayılan: .env LLM_MODEL)")
    parser.add_argument("--profile", help="embedding profili")
    parser.add_argument("--no-rag", action="store_true", help="bağlam olmadan sor (kıyas için)")
    parser.add_argument("--debug", action="store_true", help="retrieval tablosunu göster")
    parser.add_argument("--show-context", action="store_true", help="LLM'e giden bağlamı bas")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--n", type=int, help="LLM'e gidecek chunk sayısı (varsayılan 6)")
    parser.add_argument("--budget", type=int, help="bağlam token bütçesi (varsayılan 3500)")
    args = parser.parse_args()

    settings = get_settings()
    profile = load_embedding_profile(settings, args.profile)
    llm = LLM(settings, model=args.model)
    cfg = retrieval_config(args.n, args.budget)
    retriever = None
    if not args.no_rag:
        retriever = Retriever(
            Embedder(settings, profile), QdrantClient(url=settings.qdrant_url, timeout=30), cfg
        )
        if args.debug or args.show_context:
            retrieval = retriever.retrieve(args.question)
            print_result(retrieval, show_text=args.show_context)
            print("\n" + "=" * 80 + "\n")

    mode = f"RAG ({profile.name})" if retriever else "bağlamsız"
    print(f"Model: {llm.model} | Mod: {mode}\n")
    try:
        result = answer_question(
            args.question,
            llm,
            retriever,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            on_token=None if args.no_stream else lambda t: print(t, end="", flush=True),
        )
    except APITimeoutError:
        print(f"\nLLM {settings.llm_timeout:.0f} saniyede cevap vermedi (LLM_TIMEOUT).")
        print("İpucu: daha küçük bir model ya da --budget/--n ile daha kısa bağlam deneyin.")
        return 1
    except APIError as exc:
        print(f"\nLLM hatası: {exc}")
        print("İpucu: LM Studio'da modelin context length ayarı en az 8192 olmalı.")
        return 1

    if args.no_stream:
        print(result.answer)
    print_sources(result)
    first = f" (ilk token {result.first_token_seconds}s)" if result.first_token_seconds else ""
    context_tokens = sum(h.total_tokens for h in result.hits)
    print(
        f"\nSüre: retrieval {result.retrieval_seconds}s, "
        f"üretim {result.generation_seconds}s{first} | bağlam ~{context_tokens} token"
    )
    log_run(result, settings, profile.name, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
