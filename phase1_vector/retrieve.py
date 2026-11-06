"""Faz 1 retrieval: sorgu embedding -> Qdrant top-k -> filtre boost -> post limiti + MMR.

Kullanım:
    uv run python -m phase1_vector.retrieve "Rust dilinde değişken kullanımları ile ilgili örnekler"
    uv run python -m phase1_vector.retrieve "..." --no-filter --show-text
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field, replace

from qdrant_client import QdrantClient, models

from common.config import get_settings, load_embedding_profile
from common.embeddings import Embedder
from common.vectors import cosine
from phase1_vector.query_filters import QueryFilter, detect_filter
from phase1_vector.store import collection_name


@dataclass(frozen=True)
class RetrievalConfig:
    k: int = 20  # Qdrant'tan alınacak aday sayısı
    n: int = 6  # LLM'e gidecek chunk sayısı
    use_filter: bool = True
    boost: float = 0.05  # filtreye uyan adayların skoruna eklenir
    max_per_post: int = 3
    mmr_lambda: float = 0.7  # 1.0 = sadece alaka, 0.0 = sadece çeşitlilik
    token_budget: int = 3500  # seçilen chunk'ların toplam (tahmini) token üst sınırı


@dataclass
class Hit:
    id: str
    score: float
    payload: dict
    vector: list[float] = field(repr=False)
    boosted: bool = False
    final_score: float = 0.0

    @property
    def post_slug(self) -> str:
        return self.payload["post_slug"]

    @property
    def n_tokens(self) -> int:
        return int(self.payload.get("n_tokens", 0))


@dataclass
class RetrievalResult:
    query: str
    query_filter: QueryFilter | None
    filtered_count: int
    candidates: list[Hit]
    hits: list[Hit]

    @property
    def context_tokens(self) -> int:
        return sum(h.n_tokens for h in self.hits)


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        client: QdrantClient,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.client = client
        self.config = config or RetrievalConfig()
        self.collection = collection_name(embedder.profile)

    def _search(self, vector: list[float], limit: int, flt: models.Filter | None) -> list[Hit]:
        points = self.client.query_points(
            self.collection,
            query=vector,
            query_filter=flt,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        ).points
        return [Hit(str(p.id), p.score, p.payload or {}, list(p.vector)) for p in points]

    def retrieve(self, query: str, **overrides) -> RetrievalResult:
        cfg = replace(self.config, **overrides) if overrides else self.config
        query_filter = detect_filter(query) if cfg.use_filter else None
        vector = self.embedder.embed_query(query)

        candidates = {h.id: h for h in self._search(vector, cfg.k, None)}
        filtered_count = 0
        if query_filter:
            filtered = self._search(vector, cfg.k, query_filter.to_qdrant())
            filtered_count = len(filtered)
            for hit in filtered:
                candidates.setdefault(hit.id, hit)
        for hit in candidates.values():
            hit.boosted = bool(query_filter) and query_filter.matches(hit.payload)
            hit.final_score = hit.score + (cfg.boost if hit.boosted else 0.0)

        ranked = sorted(candidates.values(), key=lambda h: h.final_score, reverse=True)
        return RetrievalResult(
            query=query,
            query_filter=query_filter,
            filtered_count=filtered_count,
            candidates=ranked,
            hits=select_mmr(ranked, cfg),
        )


def select_mmr(candidates: list[Hit], cfg: RetrievalConfig) -> list[Hit]:
    """Alaka ve çeşitliliği dengeleyerek n chunk seçer (post limiti ve token bütçesiyle)."""
    selected: list[Hit] = []
    per_post: Counter[str] = Counter()
    tokens = 0
    remaining = list(candidates)
    while remaining and len(selected) < cfg.n:
        best: Hit | None = None
        best_value = float("-inf")
        for hit in remaining:
            if per_post[hit.post_slug] >= cfg.max_per_post:
                continue
            if selected and tokens + hit.n_tokens > cfg.token_budget:
                continue
            redundancy = max((cosine(hit.vector, s.vector) for s in selected), default=0.0)
            value = cfg.mmr_lambda * hit.final_score - (1 - cfg.mmr_lambda) * redundancy
            if value > best_value:
                best, best_value = hit, value
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
        per_post[best.post_slug] += 1
        tokens += best.n_tokens
    return selected


def print_result(result: RetrievalResult, show_text: bool = False) -> None:
    qf = result.query_filter
    if qf:
        print(
            f"Filtre: {qf.technologies} -> tags={qf.tags} code_langs={qf.code_langs} "
            f"({result.filtered_count} filtreli aday)"
        )
    else:
        print("Filtre: yok")
    print(
        f"Aday: {len(result.candidates)} | Seçilen: {len(result.hits)} | "
        f"Context: ~{result.context_tokens} token\n"
    )
    selected = {h.id for h in result.hits}
    print(f"{'#':>2} {'final':>6} {'skor':>6} B  {'tok':>5}  chunk / bölüm / diller")
    for rank, hit in enumerate(result.candidates, start=1):
        mark = "*" if hit.id in selected else " "
        p = hit.payload
        print(
            f"{rank:>2}{mark}{hit.final_score:6.3f} {hit.score:6.3f} {'+' if hit.boosted else ' '} "
            f"{hit.n_tokens:>5}  {p['id']}  [{p.get('section_path') or '-'}] {p.get('code_langs')}"
        )
    print("\n* = LLM'e gidecek chunk, + = filtre boost'u aldı")
    if show_text:
        for rank, hit in enumerate(result.hits, start=1):
            p = hit.payload
            print(f"\n===== [{rank}] {p['title']} ({p['date']}) — {p.get('section_path') or '-'}")
            print(f"{p['url']}\n")
            print(p["text"])


def main() -> int:
    defaults = RetrievalConfig()
    parser = argparse.ArgumentParser(description="Faz 1 retrieval (LLM yok)")
    parser.add_argument("query")
    parser.add_argument("--profile")
    parser.add_argument("--k", type=int, default=defaults.k)
    parser.add_argument("--n", type=int, default=defaults.n)
    parser.add_argument("--boost", type=float, default=defaults.boost)
    parser.add_argument("--max-per-post", type=int, default=defaults.max_per_post)
    parser.add_argument("--mmr-lambda", type=float, default=defaults.mmr_lambda)
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--show-text", action="store_true", help="seçilen chunk'ların metnini bas")
    args = parser.parse_args()

    settings = get_settings()
    profile = load_embedding_profile(settings, args.profile)
    retriever = Retriever(
        Embedder(settings, profile),
        QdrantClient(url=settings.qdrant_url, timeout=30),
        RetrievalConfig(
            k=args.k,
            n=args.n,
            use_filter=not args.no_filter,
            boost=args.boost,
            max_per_post=args.max_per_post,
            mmr_lambda=args.mmr_lambda,
        ),
    )
    print_result(retriever.retrieve(args.query), show_text=args.show_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
