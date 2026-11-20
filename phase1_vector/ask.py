"""Faz 1 uçtan uca: soru -> retrieval -> LLM -> cevap + kaynaklar.

Kullanım:
    uv run python -m phase1_vector.ask "Rust dilinde değişken kullanımları ile ilgili örnekler yaz"
    uv run python -m phase1_vector.ask "..." --debug            # retrieval tablosunu da göster
    uv run python -m phase1_vector.ask "..." --no-rag           # bağlamsız cevap (kıyas için)
    uv run python -m phase1_vector.ask "..." --model qwen3-8b   # başka bir LLM
"""

from __future__ import annotations

import argparse
import sys

from openai import APIError, APITimeoutError
from qdrant_client import QdrantClient

from common.answering import (  # noqa: F401  (AskResult geriye dönük import için)
    AskResult,
    answer_question,
    log_run,
    print_sources,
    retrieval_config,
)
from common.config import get_settings, load_embedding_profile
from common.embeddings import Embedder
from common.llm import LLM
from phase1_vector.retrieve import Retriever, print_result


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
    log_run(result, settings, profile.name, cfg, phase=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
