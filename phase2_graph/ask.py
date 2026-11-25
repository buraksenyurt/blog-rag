"""Faz 2 uçtan uca: soru -> graph retrieval (+ yapısal sorgu) -> LLM -> cevap + kaynaklar.

Kullanım:
    uv run python -m phase2_graph.ask "Blogda Rust ile ilgili kaç yazı var?"
    uv run python -m phase2_graph.ask "..." --debug          # retrieval tablosu ve graph bilgisi
    uv run python -m phase2_graph.ask "..." --no-graph       # sadece vector index (kıyas için)
"""

from __future__ import annotations

import argparse
import sys

from openai import APIError, APITimeoutError

from common.answering import (
    answer_question,
    empty_answer_reason,
    log_run,
    print_sources,
    retrieval_config,
)
from common.config import get_settings, load_embedding_profile
from common.embeddings import Embedder
from common.llm import LLM
from common.retrieval import print_result
from phase2_graph.retrieve import GraphRetrievalConfig, GraphRetriever
from phase2_graph.store import connect, make_runner


def main() -> int:
    parser = argparse.ArgumentParser(description="Faz 2 Graph RAG: soru sor")
    parser.add_argument("question")
    parser.add_argument("--model", help="LM Studio model id (varsayılan: .env LLM_MODEL)")
    parser.add_argument("--profile", help="embedding profili")
    parser.add_argument("--debug", action="store_true", help="retrieval tablosunu göster")
    parser.add_argument("--show-context", action="store_true", help="LLM'e giden bağlamı bas")
    parser.add_argument("--no-graph", action="store_true", help="graph genişletmesini kapat")
    parser.add_argument("--no-structural", action="store_true", help="yapısal sorguları kapat")
    parser.add_argument("--concepts", action="store_true", help="kavram genişletmesini aç")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--n", type=int, help="LLM'e gidecek chunk sayısı (varsayılan 6)")
    parser.add_argument("--budget", type=int, help="bağlam token bütçesi (varsayılan 3500)")
    args = parser.parse_args()

    settings = get_settings()
    profile = load_embedding_profile(settings, args.profile)
    llm = LLM(settings, model=args.model)
    base = GraphRetrievalConfig(
        graph_expand=not args.no_graph,
        structural=not args.no_structural,
        concept_expand=args.concepts,
    )
    cfg = retrieval_config(args.n, args.budget, base=base)

    driver = connect(settings)
    try:
        retriever = GraphRetriever(
            Embedder(settings, profile), make_runner(driver, settings.neo4j_database), cfg
        )
        if args.debug or args.show_context:
            print_result(retriever.retrieve(args.question), show_text=args.show_context)
            print("\n" + "=" * 80 + "\n")

        print(f"Model: {llm.model} | Mod: Graph RAG ({profile.name})\n")
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
    finally:
        driver.close()

    if args.no_stream:
        print(result.answer)
    if not result.answer.strip():
        print(f"\n! {empty_answer_reason(llm)}")
    print_sources(result)
    first = f" (ilk token {result.first_token_seconds}s)" if result.first_token_seconds else ""
    context_tokens = sum(h.total_tokens for h in result.hits)
    print(
        f"\nSüre: retrieval {result.retrieval_seconds}s, "
        f"üretim {result.generation_seconds}s{first} | bağlam ~{context_tokens} token"
    )
    log_run(result, settings, profile.name, cfg, phase=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
