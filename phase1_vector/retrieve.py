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
from phase1_vector.query_filters import QueryFilter, detect_code_intent, detect_filter
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
    code_boost: float = 0.02  # sorgu kod örneği istiyorsa kod içeren adaylara eklenir
    expand_short: bool = True  # kısa ve kodsuz chunk'lara aynı yazıdaki sonraki chunk eklenir
    short_tokens: int = 150


@dataclass
class Hit:
    id: str
    score: float
    payload: dict
    vector: list[float] = field(repr=False)
    boosted: bool = False
    code_boosted: bool = False
    final_score: float = 0.0
    extras: list[dict] = field(default_factory=list)  # genişletmeyle eklenen komşu chunk'lar

    @property
    def post_slug(self) -> str:
        return self.payload["post_slug"]

    @property
    def n_tokens(self) -> int:
        return int(self.payload.get("n_tokens", 0))

    @property
    def total_tokens(self) -> int:
        return self.n_tokens + sum(int(e.get("n_tokens", 0)) for e in self.extras)

    @property
    def text(self) -> str:
        return "\n\n".join([self.payload["text"], *(e["text"] for e in self.extras)])


@dataclass
class RetrievalResult:
    query: str
    query_filter: QueryFilter | None
    code_intent: bool
    filtered_count: int
    candidates: list[Hit]
    hits: list[Hit]

    @property
    def context_tokens(self) -> int:
        return sum(h.total_tokens for h in self.hits)


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
        code_intent = cfg.code_boost > 0 and detect_code_intent(query)
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
            hit.code_boosted = code_intent and bool(hit.payload.get("has_code"))
            hit.final_score = (
                hit.score
                + (cfg.boost if hit.boosted else 0.0)
                + (cfg.code_boost if hit.code_boosted else 0.0)
            )

        ranked = sorted(candidates.values(), key=lambda h: h.final_score, reverse=True)
        hits = select_mmr(ranked, cfg)
        if cfg.expand_short:
            self._expand_short_hits(hits, cfg)
        return RetrievalResult(
            query=query,
            query_filter=query_filter,
            code_intent=code_intent,
            filtered_count=filtered_count,
            candidates=ranked,
            hits=hits,
        )

    def _next_chunk(self, payload: dict) -> dict | None:
        points, _ = self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="post_slug", match=models.MatchValue(value=payload["post_slug"])
                    ),
                    models.FieldCondition(
                        key="chunk_index",
                        match=models.MatchValue(value=int(payload["chunk_index"]) + 1),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return points[0].payload if points else None

    def _expand_short_hits(self, hits: list[Hit], cfg: RetrievalConfig) -> None:
        """Kısa, kodsuz chunk'lar çoğu zaman bir sonraki kod örneğinin girişidir.

        Böyle chunk'lara aynı yazıdaki sonraki chunk eklenir (token bütçesi izin verirse).
        """
        selected = {h.payload["id"] for h in hits}
        tokens = sum(h.total_tokens for h in hits)
        for hit in hits:
            if hit.n_tokens >= cfg.short_tokens or hit.payload.get("has_code"):
                continue
            if "chunk_index" not in hit.payload:
                continue
            neighbor = self._next_chunk(hit.payload)
            if not neighbor or neighbor["id"] in selected:
                continue
            size = int(neighbor.get("n_tokens", 0))
            if tokens + size > cfg.token_budget:
                continue
            hit.extras.append(neighbor)
            selected.add(neighbor["id"])
            tokens += size


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
    print(f"Kod niyeti: {'evet (kod içeren adaylara boost)' if result.code_intent else 'hayır'}")
    print(
        f"Aday: {len(result.candidates)} | Seçilen: {len(result.hits)} | "
        f"Context: ~{result.context_tokens} token\n"
    )
    selected = {h.id: h for h in result.hits}
    print(f"{'#':>2} {'final':>6} {'skor':>6} FK {'tok':>5}  chunk / bölüm / diller")
    for rank, hit in enumerate(result.candidates, start=1):
        mark = "*" if hit.id in selected else " "
        p = hit.payload
        print(
            f"{rank:>2}{mark}{hit.final_score:6.3f} {hit.score:6.3f} "
            f"{'F' if hit.boosted else ' '}{'k' if hit.code_boosted else ' '} "
            f"{hit.n_tokens:>5}  {p['id']}  [{p.get('section_path') or '-'}] {p.get('code_langs')}"
        )
        for extra in selected[hit.id].extras if hit.id in selected else []:
            print(f"{'':>24}↳ +{extra['n_tokens']} tok  {extra['id']}  (sonraki chunk eklendi)")
    print("\n* = LLM'e gidecek chunk, F = teknoloji filtresi boost'u, k = kod boost'u")
    if show_text:
        for rank, hit in enumerate(result.hits, start=1):
            p = hit.payload
            print(f"\n===== [{rank}] {p['title']} ({p['date']}) — {p.get('section_path') or '-'}")
            print(f"{p['url']}\n")
            print(hit.text)


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
    parser.add_argument("--no-code-boost", action="store_true")
    parser.add_argument("--no-expand", action="store_true", help="kısa chunk genişletmesini kapat")
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
            code_boost=0.0 if args.no_code_boost else defaults.code_boost,
            expand_short=not args.no_expand,
        ),
    )
    print_result(retriever.retrieve(args.query), show_text=args.show_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
