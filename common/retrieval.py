"""Retrieval ortak yapıları: aday (Hit), sonuç, skor boost'ları, MMR seçimi, debug çıktısı.

Faz 1 (Qdrant) ve Faz 2 (Neo4j) aynı skor mantığını ve seçimi kullanır; böylece iki faz
arasındaki fark yalnızca aday üretimi ve graph genişletmesinden gelir.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from common.query_filters import QueryFilter
from common.vectors import cosine


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
    graph_bonus: float = 0.0  # Faz 2: graph ilişkisiyle gelen adaylara eklenen puan
    reasons: list[str] = field(default_factory=list)  # Faz 2: adayın graph'tan gelme nedeni

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
    facts: list[str] = field(default_factory=list)  # Faz 2: yapısal sorgu sonuçları
    fact_posts: list[str] = field(default_factory=list)  # Faz 2: yapısal sonuçtaki yazılar

    @property
    def context_tokens(self) -> int:
        return sum(h.total_tokens for h in self.hits)


def apply_boosts(
    candidates: list[Hit],
    query_filter: QueryFilter | None,
    code_intent: bool,
    cfg: RetrievalConfig,
) -> list[Hit]:
    """Teknoloji filtresi, kod niyeti ve graph boost'larını uygular, final skora göre sıralar."""
    for hit in candidates:
        hit.boosted = bool(query_filter) and query_filter.matches(hit.payload)
        hit.code_boosted = code_intent and bool(hit.payload.get("has_code"))
        hit.final_score = (
            hit.score
            + (cfg.boost if hit.boosted else 0.0)
            + (cfg.code_boost if hit.code_boosted else 0.0)
            + hit.graph_bonus
        )
    return sorted(candidates, key=lambda h: h.final_score, reverse=True)


def expand_short_hits(
    hits: list[Hit], cfg: RetrievalConfig, next_chunk: Callable[[dict], dict | None]
) -> None:
    """Kısa, kodsuz chunk'lar çoğu zaman bir sonraki kod örneğinin girişidir.

    Böyle chunk'lara aynı yazıdaki sonraki chunk eklenir (token bütçesi izin verirse).
    next_chunk, bir chunk payload'u için sonraki chunk'ın payload'unu döner (yoksa None).
    """
    selected = {h.payload["id"] for h in hits}
    tokens = sum(h.total_tokens for h in hits)
    for hit in hits:
        if hit.n_tokens >= cfg.short_tokens or hit.payload.get("has_code"):
            continue
        if "chunk_index" not in hit.payload:
            continue
        neighbor = next_chunk(hit.payload)
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


def candidate_rows(result: RetrievalResult) -> list[dict]:
    """Retrieval tablosunun yapısal hali (konsol ve arayüz aynı veriyi kullanır)."""
    selected = {h.id: (i, h) for i, h in enumerate(result.hits, start=1)}
    rows = []
    for rank, hit in enumerate(result.candidates, start=1):
        p = hit.payload
        context_no, chosen = selected.get(hit.id, (None, None))
        rows.append(
            {
                "rank": rank,
                "context": context_no,  # LLM bağlamındaki [#n] numarası, yoksa None
                "final": round(hit.final_score, 4),
                "score": round(hit.score, 4),
                "boosts": ("F" if hit.boosted else "")
                + ("k" if hit.code_boosted else "")
                + ("G" if hit.graph_bonus else ""),
                "tokens": hit.n_tokens,
                "id": p["id"],
                "title": p.get("title", ""),
                "section": p.get("section_path") or "",
                "code_langs": list(p.get("code_langs") or []),
                "reasons": list(hit.reasons),
                "extras": [e["id"] for e in chosen.extras] if chosen else [],
                "url": p.get("url", ""),
            }
        )
    return rows


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
    for fact in result.facts:
        print(f"Graph bilgisi: {fact}")
    if result.facts:
        print()
    print(f"{'#':>2} {'final':>6} {'skor':>6} FKG {'tok':>5}  chunk / bölüm / diller")
    for row in candidate_rows(result):
        mark = "*" if row["context"] else " "
        print(
            f"{row['rank']:>2}{mark}{row['final']:6.3f} {row['score']:6.3f} "
            f"{row['boosts']:<3} {row['tokens']:>5}  {row['id']}  "
            f"[{row['section'] or '-'}] {row['code_langs']}"
        )
        if row["reasons"]:
            print(f"{'':>25}↳ graph: {', '.join(row['reasons'])}")
        for extra in row["extras"]:
            print(f"{'':>25}↳ {extra}  (sonraki chunk eklendi)")
    print(
        "\n* = LLM'e gidecek chunk, F = teknoloji filtresi boost'u, k = kod boost'u, "
        "G = graph ilişkisi boost'u"
    )
    if show_text:
        for rank, hit in enumerate(result.hits, start=1):
            p = hit.payload
            print(f"\n===== [{rank}] {p['title']} ({p['date']}) — {p.get('section_path') or '-'}")
            print(f"{p['url']}\n")
            print(hit.text)
