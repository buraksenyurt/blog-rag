"""Retrieval ve cevap metrikleri (post seviyesinde)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

CODE_BLOCK_RE = re.compile(r"```.*?```", re.S)
WORD_RE = re.compile(r"[a-zçğıöşü]+")
TR_WORDS = {
    "ve",
    "bir",
    "bu",
    "için",
    "ile",
    "olarak",
    "değil",
    "da",
    "de",
    "çok",
    "gibi",
    "daha",
    "olan",
    "ise",
    "veya",
    "ancak",
    "şekilde",
    "kullanılır",
    "aşağıdaki",
}
EN_WORDS = {
    "the",
    "and",
    "is",
    "of",
    "to",
    "in",
    "that",
    "this",
    "for",
    "with",
    "are",
    "can",
    "be",
    "as",
    "it",
    "you",
    "which",
    "example",
    "following",
}


def unique_posts(slugs: Iterable[str]) -> list[str]:
    """Sırayı koruyarak tekrarları atar (chunk listesinden post sıralaması üretir)."""
    seen: dict[str, None] = {}
    for slug in slugs:
        seen.setdefault(slug, None)
    return list(seen)


def recall(relevant: Sequence[str], retrieved: Sequence[str]) -> float:
    if not relevant:
        return 0.0
    return len(set(relevant) & set(retrieved)) / len(set(relevant))


def hit(relevant: Sequence[str], retrieved: Sequence[str]) -> float:
    return 1.0 if set(relevant) & set(retrieved) else 0.0


def reciprocal_rank(relevant: Sequence[str], ranked: Sequence[str]) -> float:
    targets = set(relevant)
    for rank, slug in enumerate(ranked, start=1):
        if slug in targets:
            return 1.0 / rank
    return 0.0


def turkish_ratio(text: str) -> float | None:
    """Kod blokları dışındaki metinde Türkçe/İngilizce ortak kelime oranı (1.0 = tamamen Türkçe)."""
    words = WORD_RE.findall(CODE_BLOCK_RE.sub(" ", text.lower()))
    tr = sum(1 for w in words if w in TR_WORDS)
    en = sum(1 for w in words if w in EN_WORDS)
    return None if tr + en == 0 else round(tr / (tr + en), 2)
