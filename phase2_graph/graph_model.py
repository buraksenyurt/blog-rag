"""Chunk'lardan deterministik graph verisi üretir (LLM yok, Neo4j bağımsız, test edilebilir).

Düğümler:  Post, Chunk, Tag, Category, Language, Series
İlişkiler: (Post)-[:HAS_CHUNK]->(Chunk), (Chunk)-[:NEXT]->(Chunk),
           (Post)-[:TAGGED]->(Tag), (Post)-[:IN_CATEGORY]->(Category),
           (Chunk)-[:HAS_CODE_IN]->(Language), (Post)-[:LINKS_TO]->(Post),
           (Post)-[:IN_SERIES {position}]->(Series)
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

# "Rust Pratikleri - Lifetimes", "WCF 4.0 Yenilikleri - ...", "Tasarım Desenleri – Observer"
TITLE_SEPARATOR_RE = re.compile(r"\s+[-–—]\s+|\s*:\s+")
TRAILING_NOISE_RE = re.compile(r"[\s\d.#:–-]+$")
MIN_SERIES_SIZE = 3


@dataclass
class GraphData:
    posts: list[dict] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)
    tagged: list[dict] = field(default_factory=list)  # {slug, tag}
    tag_variants: dict[str, list[str]] = field(default_factory=dict)  # tag -> yazılışlar
    in_category: list[dict] = field(default_factory=list)  # {slug, category}
    code_langs: list[dict] = field(default_factory=list)  # {chunk_id, lang}
    next_pairs: list[dict] = field(default_factory=list)  # {a, b}
    links: list[dict] = field(default_factory=list)  # {src, dst}
    missing_links: int = 0  # kapsam dışı (ör. Foto İpucu) hedeflere giden linkler
    series: list[dict] = field(default_factory=list)  # {slug, series, position}


def normalize_tag(tag: str) -> str:
    return tag.strip().lower()


def series_prefix(title: str) -> str | None:
    """Başlığın ayraçtan önceki kısmı; sondaki numaralar atılır (ör. 'Ruby Kod Parçacıkları 16')."""
    parts = TITLE_SEPARATOR_RE.split(title, maxsplit=1)
    if len(parts) < 2:
        return None
    prefix = parts[0]
    if "(" in prefix and ")" not in prefix:  # "Oyun Programlamaya Giriş (Çarpışma ..." gibi
        prefix = prefix.split("(", 1)[0]
    prefix = TRAILING_NOISE_RE.sub("", prefix).strip()
    return prefix if len(prefix) >= 3 else None


def detect_series(posts: list[dict], min_size: int = MIN_SERIES_SIZE) -> list[dict]:
    """Aynı başlık önekini paylaşan en az min_size yazıyı tarih sırasıyla bir seri sayar."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for post in posts:
        prefix = series_prefix(post["title"])
        if prefix:
            groups[prefix].append(post)
    rows: list[dict] = []
    for name, members in groups.items():
        if len(members) < min_size:
            continue
        ordered = sorted(members, key=lambda p: (p["date"], p["slug"]))
        rows.extend(
            {"slug": p["slug"], "series": name, "position": i}
            for i, p in enumerate(ordered, start=1)
        )
    return rows


def build_graph_data(chunks: list[dict], post_links: dict[str, list[str]]) -> GraphData:
    """chunks.jsonl satırları ve post linklerinden graph satırlarını üretir.

    Chunk satırlarındaki vektörler burada yoktur; yazma sırasında cache'ten eklenir.
    """
    data = GraphData()
    posts: dict[str, dict] = {}
    by_post: dict[str, list[dict]] = defaultdict(list)
    variants: dict[str, set[str]] = defaultdict(set)

    for c in chunks:
        slug = c["post_slug"]
        if slug not in posts:
            posts[slug] = {
                "slug": slug,
                "title": c["title"],
                "date": c["date"],
                "year": c["year"],
                "url": c["url"],
            }
            for tag in c.get("tags") or []:
                name = normalize_tag(tag)
                if name:
                    variants[name].add(tag)
                    data.tagged.append({"slug": slug, "tag": name})
            for category in c.get("categories") or []:
                data.in_category.append({"slug": slug, "category": category.strip()})
        by_post[slug].append(c)
        data.chunks.append(
            {
                "id": c["id"],
                "post_slug": slug,
                "text": c["text"],
                "section_path": c.get("section_path") or "",
                "chunk_index": c["chunk_index"],
                "n_tokens": c.get("n_tokens", 0),
                "has_code": bool(c.get("has_code")),
                "code_langs": c.get("code_langs") or [],
            }
        )
        data.code_langs.extend(
            {"chunk_id": c["id"], "lang": lang} for lang in c.get("code_langs") or []
        )

    for members in by_post.values():
        ordered = sorted(members, key=lambda c: c["chunk_index"])
        data.next_pairs.extend(
            {"a": a["id"], "b": b["id"]} for a, b in zip(ordered, ordered[1:], strict=False)
        )

    for src, targets in post_links.items():
        if src not in posts:
            continue
        for dst in targets:
            if dst in posts and dst != src:
                data.links.append({"src": src, "dst": dst})
            else:
                data.missing_links += 1

    data.posts = list(posts.values())
    data.tag_variants = {name: sorted(v) for name, v in variants.items()}
    data.series = detect_series(data.posts)
    return data
