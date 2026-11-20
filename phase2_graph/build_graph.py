"""data/chunks.jsonl + embedding cache -> Neo4j deterministik graph (LLM yok).

Vektörler Faz 1'in embedding cache'inden okunur; cache'te olmayan chunk varsa
LM Studio ile embed edilip cache'e eklenir (--no-embed ile kapatılabilir).

Kullanım:
    docker compose up -d neo4j
    uv run python -m phase2_graph.build_graph
    uv run python -m phase2_graph.build_graph --recreate      # graph'ı silip baştan kur
    uv run python -m phase2_graph.build_graph --rebuild-index # sadece vector index'i yeniden kur
"""

from __future__ import annotations

import argparse
import sys
import time

from common.chunk_store import embedding_input, read_chunks, read_post_links
from common.config import ensure_writable, get_settings, load_embedding_profile
from common.embedding_cache import EmbeddingCache
from common.embeddings import Embedder
from phase2_graph.graph_model import build_graph_data
from phase2_graph.store import (
    VECTOR_INDEX,
    Runner,
    check_meta,
    connect,
    delete_stale,
    ensure_schema,
    make_runner,
    rebuild_vector_index,
    vector_index_options,
    wipe,
    write_chunks,
    write_meta,
    write_posts,
)

VERIFY_QUERIES = {
    "Rust yazıları (rust/rust-lang tag)": """
        MATCH (p:Post)-[:TAGGED]->(t:Tag) WHERE t.name IN ['rust', 'rust-lang']
        RETURN count(DISTINCT p) AS value""",
    "Go yazıları yıllara göre (golang tag)": """
        MATCH (p:Post)-[:TAGGED]->(:Tag {name: 'golang'})
        WITH p.year AS year, count(*) AS n ORDER BY year
        RETURN collect(toString(year) + ': ' + toString(n)) AS value""",
    "Rust Pratikleri serisi": """
        MATCH (p:Post)-[r:IN_SERIES]->(:Series {name: 'Rust Pratikleri'})
        WITH p ORDER BY r.position
        WITH collect(p) AS posts
        RETURN toString(size(posts)) + ' yazı, ' + toString(posts[0].date) + ' → '
               + toString(posts[-1].date) AS value""",
    "En büyük seriler": """
        MATCH (s:Series)<-[:IN_SERIES]-(p)
        WITH s.name AS name, count(p) AS n ORDER BY n DESC LIMIT 6
        RETURN collect(name + ' (' + toString(n) + ')') AS value""",
}


def print_counts(run: Runner) -> None:
    labels = run(
        "MATCH (n) WITH labels(n)[0] AS label, count(*) AS n RETURN label, n ORDER BY n DESC"
    )
    rels = run(
        "MATCH ()-[r]->() WITH type(r) AS type, count(*) AS n RETURN type, n ORDER BY n DESC"
    )
    print("\nDüğümler : " + ", ".join(f"{r['label']} {r['n']}" for r in labels))
    print("İlişkiler: " + ", ".join(f"{r['type']} {r['n']}" for r in rels))


def verify(run: Runner) -> None:
    print("\nDoğrulama sorguları")
    for title, query in VERIFY_QUERIES.items():
        rows = run(query)
        print(f"  {title}: {rows[0]['value'] if rows else '-'}")
    sanity = run(
        f"""
        MATCH (c:Chunk) WITH c ORDER BY c.id LIMIT 1
        CALL db.index.vector.queryNodes('{VECTOR_INDEX}', 3, c.embedding) YIELD node, score
        RETURN c.id AS query, collect(node.id)[0] AS top, max(score) AS score
        """
    )
    if sanity:
        ok = "OK" if sanity[0]["top"] == sanity[0]["query"] else "BEKLENMEDİK"
        print(f"  Vector index: kendi vektörüyle arama -> {ok} (skor {sanity[0]['score']:.4f})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Neo4j deterministik graph kurulumu")
    parser.add_argument("--profile", help="embedding profili (varsayılan aktif profil)")
    parser.add_argument("--recreate", action="store_true", help="graph'ı silip baştan kur")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="veriye dokunmadan sadece vector index'i silip yeniden kur",
    )
    parser.add_argument(
        "--quantization",
        action="store_true",
        help="vector index'te quantization'ı aç (varsayılan kapalı)",
    )
    parser.add_argument(
        "--no-embed", action="store_true", help="cache'te olmayan vektör varsa embed etme, dur"
    )
    args = parser.parse_args()

    started = time.perf_counter()
    settings = get_settings()
    profile = load_embedding_profile(settings, args.profile)
    chunks = read_chunks(settings.data_dir / "chunks.jsonl")
    data = build_graph_data(chunks, read_post_links(settings.data_dir / "post_links.jsonl"))
    print(
        f"Profil: {profile.name} ({profile.dim}) | yazı: {len(data.posts)} | chunk: "
        f"{len(data.chunks)} | tag: {len(data.tag_variants)} | seri: "
        f"{len({s['series'] for s in data.series})} | link: {len(data.links)} "
        f"(kapsam dışına giden: {data.missing_links})"
    )

    embedder = Embedder(settings, profile)
    cache_path = ensure_writable(settings.data_dir / "embeddings" / f"{profile.name}.sqlite")
    cache = EmbeddingCache(cache_path, profile)
    by_id = {c["id"]: c for c in chunks}
    stats = {"cached": 0, "embedded": 0}

    def vectors_for(rows: list[dict]) -> list[list[float]]:
        inputs = [embedding_input(by_id[r["id"]], embedder) for r in rows]
        keys = [cache.key(i) for i in inputs]
        found = cache.get_many(keys)
        missing = [i for i, k in enumerate(keys) if k not in found]
        if missing:
            if args.no_embed:
                raise RuntimeError(
                    f"{len(missing)} chunk cache'te yok. "
                    "Önce: uv run python -m phase1_vector.ingest"
                )
            new = embedder.embed_raw([inputs[i] for i in missing])
            cache.put_many((keys[i], v) for i, v in zip(missing, new, strict=True))
            found.update({keys[i]: v for i, v in zip(missing, new, strict=True)})
        stats["cached"] += len(rows) - len(missing)
        stats["embedded"] += len(missing)
        return [found[k] for k in keys]

    def progress(done: int, total: int) -> None:
        if done % 2000 < 200 or done == total:
            print(f"  chunk {done}/{total} ({time.perf_counter() - started:.0f}s)", flush=True)

    driver = connect(settings)
    run = make_runner(driver, settings.neo4j_database)
    if args.rebuild_index:
        try:
            print(f"Önceki index: {vector_index_options(run)}")
            print("Vector index yeniden kuruluyor (veri yeniden yazılmıyor)...")
            rebuild_vector_index(run, profile, args.quantization)
            print(f"Yeni index  : {vector_index_options(run)}")
            verify(run)
        finally:
            cache.close()
            driver.close()
        print(f"\nSüre: {time.perf_counter() - started:.0f}s")
        return 0
    try:
        if args.recreate:
            print("Graph siliniyor...")
            wipe(driver, settings.neo4j_database)
        check_meta(run, profile)
        ensure_schema(run, profile, args.quantization)
        print("Yazılar, tag'ler, kategoriler, seriler ve linkler yazılıyor...")
        write_posts(run, data)
        print("Chunk'lar ve vektörler yazılıyor...")
        write_chunks(run, data, vectors_for, progress)
        removed = delete_stale(run, data)
        write_meta(run, profile, data)
        run("CALL db.awaitIndexes(600)")
        print(
            f"\nVektör: cache {stats['cached']}, yeni embed {stats['embedded']} | silinen eski "
            f"kayıt: chunk {removed['chunks']}, yazı {removed['posts']}, "
            f"boşta kalan düğüm {removed['orphans']}"
        )
        print_counts(run)
        print(f"Vector index: {vector_index_options(run)}")
        verify(run)
    finally:
        cache.close()
        driver.close()
    print(f"\nSüre: {time.perf_counter() - started:.0f}s | Neo4j Browser: http://localhost:7474")
    return 0


if __name__ == "__main__":
    sys.exit(main())
