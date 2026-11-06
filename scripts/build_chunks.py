"""Yazıları okur, chunk'lar ve data/chunks.jsonl + data/chunk_stats.json üretir.

Kullanım:
    uv run python -m scripts.build_chunks
    uv run python -m scripts.build_chunks --show 2026-09-01-rust-kiwi-app   # tek yazıyı incele
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import asdict

from common.chunker import Chunk, ChunkerConfig, chunk_post
from common.config import ensure_writable, get_settings
from common.loader import iter_posts, load_post


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def show(slug: str, cfg: ChunkerConfig) -> int:
    settings = get_settings()
    path = settings.posts_dir / f"{slug}.md"
    if not path.exists():
        print(f"Bulunamadı: {path}")
        return 1
    post = load_post(path)
    print(f"{post.title} | {post.url}\ntags={post.tags} links={post.links}\n")
    for c in chunk_post(post, cfg):
        print(f"=== {c.id} | ~{c.n_tokens} tok | {c.section_path or '-'} | {c.code_langs}")
        print(c.text[:600] + (" …" if len(c.text) > 600 else ""))
        print()
    return 0


def build(cfg: ChunkerConfig) -> int:
    settings = get_settings()
    out_dir = ensure_writable(settings.data_dir, settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = ensure_writable(out_dir / "chunks.jsonl", settings)
    links_path = ensure_writable(out_dir / "post_links.jsonl", settings)

    total_files = len(list(settings.posts_dir.glob("*.md")))
    chunks: list[Chunk] = []
    posts = 0
    with links_path.open("w", encoding="utf-8") as links_file:
        for post in iter_posts(settings):
            posts += 1
            chunks.extend(chunk_post(post, cfg))
            if post.links:
                links_file.write(
                    json.dumps({"slug": post.slug, "links": post.links}, ensure_ascii=False) + "\n"
                )

    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    sizes = [c.n_tokens for c in chunks]
    langs = Counter(lang for c in chunks for lang in c.code_langs)
    largest = sorted(chunks, key=lambda c: c.n_tokens, reverse=True)[:5]
    stats = {
        "config": asdict(cfg),
        "files": total_files,
        "posts_indexed": posts,
        "posts_excluded": total_files - posts,
        "chunks": len(chunks),
        "chunks_per_post_avg": round(len(chunks) / max(posts, 1), 2),
        "tokens": {
            "min": min(sizes),
            "p50": int(statistics.median(sizes)),
            "p90": percentile(sizes, 0.9),
            "max": max(sizes),
            "total": sum(sizes),
        },
        "over_max": sum(1 for s in sizes if s > cfg.max_tokens),
        "with_code": sum(1 for c in chunks if c.has_code),
        "code_split": sum(1 for c in chunks if c.code_split),
        "code_langs": dict(langs.most_common(15)),
        "rust_chunks": sum(1 for c in chunks if "rust" in c.code_langs or "rust" in c.tags),
        "largest": [{"id": c.id, "tokens": c.n_tokens} for c in largest],
    }
    stats_path = ensure_writable(out_dir / "chunk_stats.json", settings)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n-> {chunks_path}\n-> {links_path}\n-> {stats_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="chunks.jsonl üretir")
    parser.add_argument("--show", metavar="SLUG", help="tek bir yazının chunk'larını göster")
    parser.add_argument("--target", type=int, default=ChunkerConfig.target_tokens)
    parser.add_argument("--max", type=int, default=ChunkerConfig.max_tokens)
    args = parser.parse_args()
    cfg = ChunkerConfig(target_tokens=args.target, max_tokens=args.max)
    return show(args.show, cfg) if args.show else build(cfg)


if __name__ == "__main__":
    sys.exit(main())
