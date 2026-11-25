"""Faz 2 retrieval: Neo4j vector index + graph genişletme + yapısal sorular.

Akış:
  1. Giriş noktası: vector index ile top-k chunk (Faz 1'deki Qdrant aramasının karşılığı).
  2. Teknoloji filtresi varsa ilgili tag/dil altındaki chunk'lar arasında tam cosine araması.
  3. Graph genişletme: en iyi yazıların LINKS_TO ve seri komşularından en alakalı chunk'lar
     aday havuzuna eklenir (+graph_boost).
  4. Faz 1 ile aynı boost'lar, MMR seçimi ve kısa chunk -> NEXT genişletmesi.
  5. Yapısal soru ise (sayma, yıllar, listeleme, seri) Cypher şablonuyla kesin sonuç.

Kullanım:
    uv run python -m phase2_graph.retrieve "Rust Pratikleri serisinde hangi yazılar var?"
    uv run python -m phase2_graph.retrieve "..." --no-graph --show-text
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, replace

from common.config import get_settings, load_embedding_profile
from common.embeddings import Embedder
from common.query_filters import QueryFilter, detect_code_intent, detect_filter, normalize_query
from common.retrieval import (
    Hit,
    RetrievalConfig,
    RetrievalResult,
    apply_boosts,
    expand_short_hits,
    print_result,
    select_mmr,
)
from phase2_graph.concepts import mention_pattern
from phase2_graph.store import VECTOR_INDEX, Runner, connect, make_runner
from phase2_graph.structural import answer_structural, detect_structural

# Tüm aday sorguları aynı alanları döner; c ve p değişkenleri ile score bağlı olmalı.
RETURN_CANDIDATE = """
OPTIONAL MATCH (p)-[:TAGGED]->(t:Tag)
OPTIONAL MATCH (p)-[sr:IN_SERIES]->(ser:Series)
WITH c, p, score, reasons, ser.name AS series, sr.position AS position,
     collect(DISTINCT t.name) AS tags
RETURN c {.id, .post_slug, .text, .section_path, .chunk_index, .n_tokens, .has_code,
          .code_langs} AS chunk,
       c.embedding AS embedding,
       p {.slug, .title, .url, .year, date: toString(p.date)} AS post,
       tags, series, position, score, reasons
"""

Q_VECTOR = (
    """
CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node AS c, score
MATCH (p:Post)-[:HAS_CHUNK]->(c)
WITH c, p, score, [] AS reasons
"""
    + RETURN_CANDIDATE
)

# Index kullanmadan bütün chunk'larla tam cosine (ANN/quantization etkisini ölçmek için)
Q_EXACT = (
    """
MATCH (c:Chunk)
WITH c, vector.similarity.cosine(c.embedding, $vec) AS score
ORDER BY score DESC LIMIT $k
MATCH (p:Post)-[:HAS_CHUNK]->(c)
WITH c, p, score, [] AS reasons
"""
    + RETURN_CANDIDATE
)

Q_FILTERED = (
    """
CALL {
  MATCH (fp:Post)-[:TAGGED]->(ft:Tag) WHERE ft.name IN $tags
  MATCH (fp)-[:HAS_CHUNK]->(fc:Chunk)
  RETURN fc AS c
  UNION
  MATCH (lc:Chunk)-[:HAS_CODE_IN]->(l:Language) WHERE l.name IN $langs
  RETURN lc AS c
}
WITH DISTINCT c
WITH c, vector.similarity.cosine(c.embedding, $vec) AS score
ORDER BY score DESC LIMIT $k
MATCH (p:Post)-[:HAS_CHUNK]->(c)
WITH c, p, score, [] AS reasons
"""
    + RETURN_CANDIDATE
)

Q_RELATED = (
    """
UNWIND $seeds AS seed
MATCH (s:Post {slug: seed})
CALL {
  WITH s
  MATCH (s)-[:LINKS_TO]-(o:Post)
  RETURN o, 'link: ' + s.title AS reason
  UNION
  WITH s
  MATCH (s)-[:IN_SERIES]->(sr:Series)<-[:IN_SERIES]-(o:Post)
  RETURN o, 'seri: ' + sr.name AS reason
}
WITH o, collect(DISTINCT reason) AS reasons
WHERE NOT o.slug IN $seeds
MATCH (o)-[:HAS_CHUNK]->(oc:Chunk)
WITH o, reasons, oc, vector.similarity.cosine(oc.embedding, $vec) AS sim
ORDER BY sim DESC
WITH o, reasons, collect({chunk: oc, sim: sim})[0] AS best
ORDER BY best.sim DESC LIMIT $limit
WITH best.chunk AS c, o AS p, best.sim AS score, reasons
"""
    + RETURN_CANDIDATE
)

# Adım 9: sorguda geçen kavramlardan (MENTIONS) aday chunk'lar
Q_CONCEPT_VOCAB = "MATCH (k:Concept) RETURN k.name AS name, k.aliases AS aliases, k.df AS df"
Q_CONCEPT_CANDIDATES = (
    """
MATCH (k:Concept) WHERE k.name IN $names
MATCH (c:Chunk)-[:MENTIONS]->(k)
MATCH (cp:Post)-[:HAS_CHUNK]->(c)
// Sorguda teknoloji varsa kavram adayları o teknolojinin yazılarından gelir
// (Go sorusundaki "channel" Rust yazılarındaki channel chunk'larını çekmesin).
WHERE size($tags) = 0 OR EXISTS { (cp)-[:TAGGED]->(t:Tag) WHERE t.name IN $tags }
WITH c, collect(k.name) AS concept_names
WITH c, concept_names, vector.similarity.cosine(c.embedding, $vec) AS score
ORDER BY score DESC LIMIT $k
MATCH (p:Post)-[:HAS_CHUNK]->(c)
WITH c, p, score, [n IN concept_names | 'kavram: ' + n] AS reasons
"""
    + RETURN_CANDIDATE
)

Q_NEXT = """
MATCH (:Chunk {id: $id})-[:NEXT]->(n:Chunk)
MATCH (p:Post)-[:HAS_CHUNK]->(n)
RETURN n {.id, .post_slug, .text, .section_path, .chunk_index, .n_tokens, .has_code,
          .code_langs} AS chunk,
       p {.slug, .title, .url, .year, date: toString(p.date)} AS post
"""


@dataclass(frozen=True)
class GraphRetrievalConfig(RetrievalConfig):
    graph_expand: bool = True  # seed yazıların link/seri komşularını aday havuzuna ekle
    seed_posts: int = 3  # genişletmenin başladığı en iyi yazı sayısı
    related_limit: int = 4  # eklenecek en fazla komşu yazı
    graph_boost: float = 0.02  # graph ilişkisiyle gelen adaylara eklenen puan
    structural: bool = True  # sayma/listeleme sorularını Cypher şablonlarıyla cevapla
    # Giriş araması: 10 bin chunk için tam cosine hem hızlı hem kesin. Neo4j'nin HNSW
    # index'i quantization kapalıyken bile bazı adayları kaçırıyor (eval: vector / exact).
    # Böylece Faz 2'nin giriş noktası Faz 1 ile aynı olur; fark yalnızca graph'tan gelir.
    exact_search: bool = True
    concept_expand: bool = False  # Adım 9: sorgudaki kavramların geçtiği chunk'ları ekle
    concept_boost: float = 0.03
    max_concept_df: int = 8  # çok yazıda geçen "merkez" kavramlar (ör. rust) kullanılmaz


def to_cosine(score: float) -> float:
    """Neo4j cosine skorunu [0, 1] aralığından gerçek cosine'e çevirir: (1 + cos) / 2."""
    return 2.0 * score - 1.0


def make_payload(chunk: dict, post: dict, tags: list[str], series: str | None, position) -> dict:
    payload = {
        **chunk,
        "title": post["title"],
        "date": post["date"],
        "year": post["year"],
        "url": post["url"],
        "tags": tags,
    }
    if series:
        payload["series"] = series
        payload["context_note"] = f"{series} serisi, {position}. yazı"
    return payload


def row_to_hit(row: dict) -> Hit:
    payload = make_payload(
        row["chunk"], row["post"], row["tags"], row.get("series"), row.get("position")
    )
    return Hit(
        id=row["chunk"]["id"],
        score=to_cosine(row["score"]),
        payload=payload,
        vector=list(row["embedding"]),
        reasons=list(row.get("reasons") or []),
    )


def lowercase_filter(query_filter: QueryFilter) -> QueryFilter:
    """Graph'ta tag adları küçük harfle tutulur (gRPC -> grpc)."""
    tags = sorted({t.lower() for t in query_filter.tags})
    return replace(query_filter, tags=tags)


class GraphRetriever:
    def __init__(
        self, embedder: Embedder, run: Runner, config: GraphRetrievalConfig | None = None
    ) -> None:
        self.embedder = embedder
        self.run = run
        self.config = config or GraphRetrievalConfig()
        self.series_names = [r["name"] for r in run("MATCH (s:Series) RETURN s.name AS name")]
        self._concept_vocab: list[tuple[str, int, re.Pattern]] | None = None

    def query_concepts(self, query: str, max_df: int) -> list[str]:
        """Sorguda adı ya da takma adı geçen (ve merkez olmayan) kavramlar."""
        if self._concept_vocab is None:
            self._concept_vocab = [
                (r["name"], r["df"] or 0, pattern)
                for r in self.run(Q_CONCEPT_VOCAB)
                if (pattern := mention_pattern([r["name"], *(r["aliases"] or [])]))
            ]
        text = normalize_query(query)
        return [
            name
            for name, df, pattern in self._concept_vocab
            if df <= max_df and pattern.search(text)
        ]

    def _candidates(self, query: str, params: dict) -> list[Hit]:
        return [row_to_hit(r) for r in self.run(query, **params)]

    def _next_chunk(self, payload: dict) -> dict | None:
        rows = self.run(Q_NEXT, id=payload["id"])
        if not rows:
            return None
        return make_payload(rows[0]["chunk"], rows[0]["post"], [], None, None)

    def retrieve(
        self, query: str, vector: list[float] | None = None, **overrides
    ) -> RetrievalResult:
        cfg = replace(self.config, **overrides) if overrides else self.config
        query_filter = lowercase_filter(detect_filter(query)) if cfg.use_filter else None
        code_intent = cfg.code_boost > 0 and detect_code_intent(query)
        if vector is None:
            vector = self.embedder.embed_query(query)

        if cfg.exact_search:
            entry = self._candidates(Q_EXACT, {"k": cfg.k, "vec": vector})
        else:
            entry = self._candidates(Q_VECTOR, {"index": VECTOR_INDEX, "k": cfg.k, "vec": vector})
        candidates = {h.id: h for h in entry}
        filtered_count = 0
        if query_filter:
            filtered = self._candidates(
                Q_FILTERED,
                {
                    "tags": query_filter.tags,
                    "langs": query_filter.code_langs,
                    "k": cfg.k,
                    "vec": vector,
                },
            )
            filtered_count = len(filtered)
            for hit in filtered:
                candidates.setdefault(hit.id, hit)

        if cfg.concept_expand:
            names = self.query_concepts(query, cfg.max_concept_df)
            if names:
                tags = query_filter.tags if query_filter else []
                concept_hits = self._candidates(
                    Q_CONCEPT_CANDIDATES,
                    {"names": names, "vec": vector, "k": cfg.k, "tags": tags},
                )
                for hit in concept_hits:
                    target = candidates.setdefault(hit.id, hit)
                    target.graph_bonus = max(target.graph_bonus, cfg.concept_boost)
                    target.reasons = list(dict.fromkeys([*target.reasons, *hit.reasons]))

        ranked = apply_boosts(list(candidates.values()), query_filter, code_intent, cfg)
        if cfg.graph_expand and ranked:
            self._expand_graph(ranked, candidates, vector, cfg)
            ranked = apply_boosts(list(candidates.values()), query_filter, code_intent, cfg)

        hits = select_mmr(ranked, cfg)
        if cfg.expand_short:
            expand_short_hits(hits, cfg, self._next_chunk)

        facts: list[str] = []
        fact_posts: list[str] = []
        if cfg.structural:
            intent = detect_structural(query, self.series_names)
            answer = answer_structural(self.run, intent) if intent else None
            if answer:
                facts, fact_posts = answer.facts, answer.posts

        return RetrievalResult(
            query=query,
            query_filter=query_filter,
            code_intent=code_intent,
            filtered_count=filtered_count,
            candidates=ranked,
            hits=hits,
            facts=facts,
            fact_posts=fact_posts,
        )

    def _expand_graph(
        self,
        ranked: list[Hit],
        candidates: dict[str, Hit],
        vector: list[float],
        cfg: GraphRetrievalConfig,
    ) -> None:
        seeds: list[str] = []
        for hit in ranked:
            if hit.post_slug not in seeds:
                seeds.append(hit.post_slug)
            if len(seeds) >= cfg.seed_posts:
                break
        related = self._candidates(
            Q_RELATED, {"seeds": seeds, "vec": vector, "limit": cfg.related_limit}
        )
        related_posts = {h.post_slug: h.reasons for h in related}
        for hit in related:
            candidates.setdefault(hit.id, hit)
        for hit in candidates.values():
            if hit.post_slug in related_posts:
                hit.graph_bonus = max(hit.graph_bonus, cfg.graph_boost)
                hit.reasons = list(dict.fromkeys([*hit.reasons, *related_posts[hit.post_slug]]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Faz 2 graph retrieval (LLM yok)")
    parser.add_argument("query")
    parser.add_argument("--profile")
    parser.add_argument("--n", type=int)
    parser.add_argument("--no-graph", action="store_true", help="graph genişletmesini kapat")
    parser.add_argument("--no-structural", action="store_true", help="yapısal sorguları kapat")
    parser.add_argument("--index", action="store_true", help="tam cosine yerine vector index")
    parser.add_argument("--concepts", action="store_true", help="kavram genişletmesini aç")
    parser.add_argument("--show-text", action="store_true", help="seçilen chunk'ların metnini bas")
    args = parser.parse_args()

    settings = get_settings()
    profile = load_embedding_profile(settings, args.profile)
    overrides = {
        "graph_expand": not args.no_graph,
        "structural": not args.no_structural,
        "exact_search": not args.index,
        "concept_expand": args.concepts,
    }
    if args.n:
        overrides["n"] = args.n
    driver = connect(settings)
    try:
        retriever = GraphRetriever(
            Embedder(settings, profile),
            make_runner(driver, settings.neo4j_database),
            GraphRetrievalConfig(**overrides),
        )
        print_result(retriever.retrieve(args.query), show_text=args.show_text)
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
