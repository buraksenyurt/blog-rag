"""Yapısal sorular (sayma, yıllara göre dağılım, listeleme, seri) için Cypher şablonları.

Vektör araması "kaç yazı var?" sorusunu cevaplayamaz; graph bunu kesin olarak sayar.
Niyet basit kurallarla tespit edilir, teknoloji adları ortak teknoloji sözlüğünden gelir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from common.query_filters import detect_filter, normalize_query
from phase2_graph.store import Runner

COUNT_RE = re.compile(r"(?<!\w)kaç\s+(tane\s+)?(yazı|makale|içerik|post|blog yazısı)")
YEARS_RE = re.compile(r"hangi\s+yıl|yıllara\s+göre|yıl\s+yıl|ne\s+zaman(lar)?")
LIST_RE = re.compile(r"hangi\s+yazı|yazıları\s+listele|yazılar(ı)?\s+ne(ler)?(dir)?|listele")
SERIES_RE = re.compile(r"(?<!\w)seri(si|sinde|sindeki|nin|ye)?(?!\w)")
LIST_LIMIT = 40

Q_COUNT = """
MATCH (p:Post)-[:TAGGED]->(t:Tag) WHERE t.name IN $tags
WITH DISTINCT p
RETURN count(p) AS n, min(p.year) AS first, max(p.year) AS last
"""
Q_YEARS = """
MATCH (p:Post)-[:TAGGED]->(t:Tag) WHERE t.name IN $tags
WITH DISTINCT p
WITH p.year AS year, count(p) AS n ORDER BY year
RETURN year, n
"""
Q_LIST = """
MATCH (p:Post)-[:TAGGED]->(t:Tag) WHERE t.name IN $tags
WITH DISTINCT p ORDER BY p.date
RETURN p.slug AS slug, p.title AS title, toString(p.date) AS date
"""
Q_SERIES = """
MATCH (p:Post)-[r:IN_SERIES]->(:Series {name: $name})
RETURN p.slug AS slug, p.title AS title, toString(p.date) AS date, r.position AS position
ORDER BY r.position
"""


@dataclass
class StructuralIntent:
    kind: str  # count | years | list | series
    tags: list[str] = field(default_factory=list)
    series: str | None = None


@dataclass
class StructuralAnswer:
    kind: str
    facts: list[str]
    posts: list[str] = field(default_factory=list)


def detect_structural(question: str, series_names: list[str]) -> StructuralIntent | None:
    text = normalize_query(question)
    if SERIES_RE.search(text):
        matches = [s for s in series_names if len(s) >= 4 and s.lower() in text]
        if matches:
            return StructuralIntent("series", series=max(matches, key=len))
    tags = sorted({t.lower() for t in detect_filter(question).tags})
    if not tags:
        return None
    if COUNT_RE.search(text):
        return StructuralIntent("count", tags=tags)
    if YEARS_RE.search(text):
        return StructuralIntent("years", tags=tags)
    if LIST_RE.search(text):
        return StructuralIntent("list", tags=tags)
    return None


def _tags_text(tags: list[str]) -> str:
    return " veya ".join(tags)


def answer_structural(run: Runner, intent: StructuralIntent) -> StructuralAnswer | None:
    if intent.kind == "count":
        row = run(Q_COUNT, tags=intent.tags)[0]
        if not row["n"]:
            return None
        span = f"{row['first']}–{row['last']}" if row["first"] != row["last"] else row["first"]
        fact = f"Blogda {_tags_text(intent.tags)} tag'li {row['n']} yazı var ({span})."
        return StructuralAnswer("count", [fact])

    if intent.kind == "years":
        rows = run(Q_YEARS, tags=intent.tags)
        if not rows:
            return None
        total = sum(r["n"] for r in rows)
        dist = ", ".join(f"{r['year']}: {r['n']}" for r in rows)
        fact = f"{_tags_text(intent.tags)} tag'li {total} yazının yıllara göre dağılımı: {dist}."
        return StructuralAnswer("years", [fact])

    if intent.kind == "list":
        rows = run(Q_LIST, tags=intent.tags)
        if not rows:
            return None
        shown = rows[:LIST_LIMIT]
        items = "; ".join(f"{i}) {r['title']} ({r['date']})" for i, r in enumerate(shown, 1))
        more = f" (ilk {LIST_LIMIT} tanesi)" if len(rows) > LIST_LIMIT else ""
        fact = f"{_tags_text(intent.tags)} tag'li {len(rows)} yazı{more}: {items}."
        return StructuralAnswer("list", [fact], [r["slug"] for r in rows])

    if intent.kind == "series" and intent.series:
        rows = run(Q_SERIES, name=intent.series)
        if not rows:
            return None
        items = "; ".join(f"{r['position']}) {r['title']} ({r['date']})" for r in rows)
        fact = (
            f"'{intent.series}' serisi {len(rows)} yazıdan oluşuyor "
            f"({rows[0]['date']} → {rows[-1]['date']}): {items}."
        )
        return StructuralAnswer("series", [fact], [r["slug"] for r in rows])
    return None
