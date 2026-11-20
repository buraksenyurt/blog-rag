"""Faz 1 retrieval: sorgu embedding -> Qdrant top-k -> filtre boost -> post limiti + MMR.

Kullanım:
    uv run python -m phase1_vector.retrieve "Rust dilinde değişken kullanımları ile ilgili örnekler"
    uv run python -m phase1_vector.retrieve "..." --no-filter --show-text
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from qdrant_client import QdrantClient, models

from common.config import get_settings, load_embedding_profile
from common.embeddings import Embedder
from common.query_filters import detect_code_intent, detect_filter
from common.retrieval import (  # noqa: F401  (Hit, RetrievalResult geriye dönük import için)
    Hit,
    RetrievalConfig,
    RetrievalResult,
    apply_boosts,
    expand_short_hits,
    print_result,
    select_mmr,
)
from phase1_vector.store import collection_name


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

    def retrieve(
        self, query: str, vector: list[float] | None = None, **overrides
    ) -> RetrievalResult:
        """vector verilirse sorgu yeniden embed edilmez (eval'de varyant kıyası için)."""
        cfg = replace(self.config, **overrides) if overrides else self.config
        query_filter = detect_filter(query) if cfg.use_filter else None
        code_intent = cfg.code_boost > 0 and detect_code_intent(query)
        if vector is None:
            vector = self.embedder.embed_query(query)

        candidates = {h.id: h for h in self._search(vector, cfg.k, None)}
        filtered_count = 0
        if query_filter:
            filtered = self._search(vector, cfg.k, query_filter.to_qdrant())
            filtered_count = len(filtered)
            for hit in filtered:
                candidates.setdefault(hit.id, hit)
        ranked = apply_boosts(list(candidates.values()), query_filter, code_intent, cfg)
        hits = select_mmr(ranked, cfg)
        if cfg.expand_short:
            expand_short_hits(hits, cfg, self._next_chunk)
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
