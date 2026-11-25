"""Adım 9 pilotu: seçili tag'lerdeki yazılardan LLM ile kavram çıkarıp graph'a ekler.

Kullanım:
    uv run python -m phase2_graph.extract_entities --limit 3 --dry-run   # önce 3 yazıyla dene
    uv run python -m phase2_graph.extract_entities                       # Rust yazıları (46)
    uv run python -m phase2_graph.extract_entities --tags golang --model gemma-3-4b-it-qat

Çıkarım sonuçları data/concepts/{model}/ altında cache'lenir; tekrar çalıştırınca sadece
eksik yazılar için LLM çağrılır (--refresh hepsini yeniden çıkarır).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter, defaultdict

from openai import APIError

from common.chunk_store import read_chunks
from common.config import ensure_writable, get_settings
from common.llm import LLM
from phase2_graph.concepts import (
    DEFAULT_BUDGET_CHARS,
    PROMPT_VERSION,
    SCHEMA,
    PostConcepts,
    build_concept_graph,
    build_messages,
    build_post_excerpt,
    clean_post_concepts,
    load_cached,
    save_cached,
)
from phase2_graph.store import connect, make_runner, write_concepts


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", text.lower()).strip("-")


def select_posts(chunks: list[dict], tags: set[str]) -> dict[str, list[dict]]:
    posts: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        if {t.lower() for t in chunk.get("tags") or []} & tags:
            posts[chunk["post_slug"]].append(chunk)
    return dict(sorted(posts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM ile kavram çıkarımı (pilot)")
    parser.add_argument("--tags", default="rust,rust-lang", help="virgülle ayrılmış tag'ler")
    parser.add_argument("--limit", type=int, help="ilk N yazı (deneme için)")
    parser.add_argument("--model", help="LM Studio model id (varsayılan .env LLM_MODEL)")
    parser.add_argument("--refresh", action="store_true", help="cache'i yok say, yeniden çıkar")
    parser.add_argument(
        "--cache-version",
        type=int,
        default=PROMPT_VERSION,
        help=f"cache/prompt sürümü klasörü (varsayılan v{PROMPT_VERSION})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Neo4j'ye yazma, sadece raporla")
    parser.add_argument(
        "--budget", type=int, default=DEFAULT_BUDGET_CHARS, help="yazı özeti karakter bütçesi"
    )
    args = parser.parse_args()

    started = time.perf_counter()
    settings = get_settings()
    llm = LLM(settings, model=args.model)
    chunks = read_chunks(settings.data_dir / "chunks.jsonl")
    tags = {t.strip().lower() for t in args.tags.split(",") if t.strip()}
    posts = select_posts(chunks, tags)
    slugs = list(posts)[: args.limit] if args.limit else list(posts)
    cache_dir = ensure_writable(
        settings.data_dir / "concepts" / slugify(llm.model) / f"v{args.cache_version}", settings
    )
    print(f"Model: {llm.model} | tag: {sorted(tags)} | yazı: {len(slugs)} | cache: {cache_dir}\n")

    extracted: list[PostConcepts] = []
    failures: list[tuple[str, str]] = []
    llm_calls = 0
    for i, slug in enumerate(slugs, start=1):
        raw = None if args.refresh else load_cached(cache_dir, slug)
        source = "cache"
        if raw is None:
            source = "LLM"
            t0 = time.perf_counter()
            messages = build_messages(build_post_excerpt(posts[slug], args.budget))
            try:
                try:
                    raw = llm.chat_json(messages, SCHEMA, name="concepts", max_tokens=2048)
                except ValueError:
                    # Çıktı max_tokens'ta kesilip JSON yarım kalmış olabilir: bir kez daha dene
                    raw = llm.chat_json(messages, SCHEMA, name="concepts", max_tokens=3072)
            except (APIError, ValueError) as exc:
                failures.append((slug, f"{type(exc).__name__}: {exc}"[:160]))
                print(f"  [{i}/{len(slugs)}] {slug[:60]} HATA ({type(exc).__name__})")
                continue
            llm_calls += 1
            save_cached(cache_dir, slug, raw)
            source = f"LLM {time.perf_counter() - t0:.0f}s"
        post = clean_post_concepts(slug, raw)
        # Seçimin konusu olan dil/tag (ör. rust) kavram olarak her yazıya bağlanıp
        # merkez düğüm oluşturur; Tag düğümü bu işi zaten görüyor.
        post.concepts = [c for c in post.concepts if c["name"] not in tags]
        kept = {c["name"] for c in post.concepts}
        post.relations = [r for r in post.relations if {r["source"], r["target"]} <= kept]
        extracted.append(post)
        names = ", ".join(c["name"] for c in post.concepts[:6])
        print(f"  [{i}/{len(slugs)}] {slug[:60]} ({source}) -> {names}", flush=True)

    graph = build_concept_graph(extracted, chunks)
    merged = [c for c in graph.concepts if c["aliases"]]
    kinds = Counter(r["kind"] for r in graph.relations)
    print(
        f"\nYazı: {len(extracted)}/{len(slugs)} (hata {len(failures)}, LLM çağrısı {llm_calls})"
        f"\nYazıda geçmediği için atılan kavram ataması: {graph.unsupported}"
        f"\nKavram: {len(graph.concepts)} | ABOUT: {len(graph.about)} | "
        f"MENTIONS: {len(graph.mentions)} | ilişki: {len(graph.relations)} {dict(kinds)}"
    )
    top = sorted(graph.concepts, key=lambda c: (-c["df"], c["name"]))[:15]
    print("En çok yazıda geçen kavramlar: " + ", ".join(f"{c['name']} ({c['df']})" for c in top))
    unverified = sum(len(set(c["aliases"]) - set(c["mention_terms"])) for c in graph.concepts)
    print(f"Takma ad (sadece sorgu eşleştirmesinde kullanılır): {unverified}")
    print(
        "Takma adla birleşenler (ilk 10): "
        + "; ".join(f"{c['name']} <- {', '.join(c['aliases'][:3])}" for c in merged[:10])
    )
    for slug, error in failures:
        print(f"  ! {slug}: {error}")

    if args.dry_run:
        print("\n--dry-run: Neo4j'ye yazılmadı.")
    else:
        driver = connect(settings)
        try:
            run = make_runner(driver, settings.neo4j_database)
            write_concepts(run, graph)
            rows = run(
                """
                MATCH (k:Concept)<-[:MENTIONS]-(c:Chunk)
                WITH k, count(c) AS n ORDER BY n DESC LIMIT 8
                RETURN collect(k.name + ' (' + toString(n) + ')') AS value
                """
            )
            print(
                f"\nNeo4j: en çok chunk'ta geçen kavramlar -> {rows[0]['value'] if rows else '-'}"
            )
        finally:
            driver.close()
    print(f"\nSüre: {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
