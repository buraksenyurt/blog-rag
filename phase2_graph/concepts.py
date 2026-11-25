"""LLM ile yazı başına kavram çıkarımı (Adım 9, pilot).

- Girdi: yazının başlığı, etiketleri, bölüm başlıkları ve kısaltılmış metni.
- Çıktı: JSON şemasına uyan kavram listesi (ad, tür, Türkçe/İngilizce takma adlar) ve
  bu kavramlar arasındaki ilişkiler.
- Sonuçlar data/concepts/{model}/{slug}.json olarak cache'lenir (bu makinede her çağrı
  dakikalar sürebilir; yarıda kalan çalıştırma kaldığı yerden devam eder).
- Kavram adları sadece metin normalizasyonuyla birleştirilir. embeddinggemma skorları
  çok dar aralıkta olduğu için embedding benzerliğiyle otomatik birleştirme yapılmaz.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CONCEPT_TYPES = ["language", "library", "concept", "tool", "pattern"]
RELATION_KINDS = ["uses", "part_of", "related_to", "alternative_to"]
MAX_CONCEPTS = 15
DEFAULT_BUDGET_CHARS = 12000  # ~4,5 bin token; 8192 context'e cevapla birlikte sığar
# Prompt ya da özet mantığı değişince artırılır; cache ayrı klasöre yazılır.
PROMPT_VERSION = 4

SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": CONCEPT_TYPES},
                    "aliases": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
                },
                "required": ["name", "type", "aliases"],
                "additionalProperties": False,
            },
            "maxItems": MAX_CONCEPTS,
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "kind": {"type": "string", "enum": RELATION_KINDS},
                },
                "required": ["source", "target", "kind"],
                "additionalProperties": False,
            },
            "maxItems": 20,
        },
    },
    "required": ["concepts", "relations"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = f"""Sen teknik blog yazılarından bilgi grafiği için kavram çıkaran bir asistansın.

Kurallar:
1. Yalnızca yazıda gerçekten anlatılan ya da kodda kullanılan teknik kavramları çıkar. \
En fazla {MAX_CONCEPTS} kavram yaz, en önemlileri seç.
2. Kavram adını kısa, küçük harfli ve yaygın İngilizce teknik terim olarak yaz. \
Yazıda Türkçe geçse bile adı İngilizce yaz (ör. metinde "çöp toplayıcı" geçiyorsa ad \
"garbage collection" olur).
3. Kavramın Türkçe karşılığını ve yazıda geçen farklı yazılışlarını aliases listesine ekle \
(ör. "garbage collection" için "çöp toplayıcı"). Yoksa boş liste bırak.
4. Bir kavramı ancak yazıda adı ya da Türkçe karşılığı geçiyorsa yaz. Yazıda geçmeyen \
kavram ekleme.
5. type değeri şunlardan biri olmalı: {", ".join(CONCEPT_TYPES)}.
6. İlişkileri yalnızca çıkardığın kavramlar arasında yaz. kind değeri şunlardan biri \
olmalı: {", ".join(RELATION_KINDS)}.
7. "programlama", "kod", "örnek", "uygulama" gibi genel kelimeleri ve println!, \
String::from gibi kod ifadelerini kavram olarak yazma.
8. Sadece istenen JSON nesnesini döndür."""

CODE_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
SPACES_RE = re.compile(r"\s+")
EDGE_PUNCT_RE = re.compile(r"^[\s`'\"“”‘’.,;:()\[\]-]+|[\s`'\"“”‘’.,;:()\[\]-]+$")
GENERIC_NAMES = {
    "programlama", "programming", "kod", "code", "örnek", "example", "uygulama",
    "main", "function", "fonksiyon",
    # Türkçe metinde sık geçen kelimelerle çakışan adlar ("bir paket var" -> var)
    "var", "yok", "bir", "ile", "için", "gibi", "olan", "daha", "çok", "ana", "dil",
}  # fmt: skip
# Kavram değil kod ifadesi: println!, String::from, main(), Vec<T> gibi
CODE_LIKE_RE = re.compile(r"::|[!(){}\[\]<>;=&*]")


@dataclass
class PostConcepts:
    slug: str
    concepts: list[dict] = field(default_factory=list)  # {name, type, aliases}
    relations: list[dict] = field(default_factory=list)  # {source, target, kind}


@dataclass
class ConceptGraph:
    unsupported: int = 0  # yazıda adı/karşılığı geçmediği için atılan kavram atamaları
    concepts: list[dict] = field(default_factory=list)  # {name, type, aliases, df}
    about: list[dict] = field(default_factory=list)  # {slug, concept}
    mentions: list[dict] = field(default_factory=list)  # {chunk_id, concept}
    relations: list[dict] = field(default_factory=list)  # {source, target, kind, count}


def shorten_code(text: str, max_lines: int = 8) -> str:
    """Kod bloklarını ilk satırlarıyla sınırlar (kavram ipucu kalır, bütçe korunur)."""

    def cut(match: re.Match) -> str:
        lines = match.group(1).splitlines()
        body = "\n".join(lines[:max_lines]) + ("\n// ..." if len(lines) > max_lines else "")
        return f"```\n{body}\n```"

    return CODE_BLOCK_RE.sub(cut, text)


def build_post_excerpt(chunks: list[dict], budget_chars: int = DEFAULT_BUDGET_CHARS) -> str:
    """Yazının LLM'e gidecek özeti: başlık, etiketler, bölümler ve kısaltılmış metin."""
    ordered = sorted(chunks, key=lambda c: c["chunk_index"])
    first = ordered[0]
    sections = list(dict.fromkeys(c["section_path"] for c in ordered if c.get("section_path")))
    # Tag'ler bilerek verilmiyor: verildiğinde model kavram listesini tag'lerle dolduruyor
    # (dry-run v2). Tag'ler graph'ta zaten Tag düğümü; kavram katmanı metinden yeni bilgi
    # çıkarmalı.
    header = [f"Başlık: {first['title']}"]
    if sections:
        header.append("Bölümler: " + "; ".join(sections))
    text = "\n\n".join(shorten_code(c["text"]) for c in ordered)
    excerpt = "\n".join(header) + "\n\nMetin:\n" + text
    return excerpt[:budget_chars]


def build_messages(excerpt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": excerpt},
    ]


def normalize_name(name: str) -> str:
    name = EDGE_PUNCT_RE.sub("", SPACES_RE.sub(" ", name.strip().lower()))
    return name.replace("_", " ")


def clean_post_concepts(slug: str, raw: dict) -> PostConcepts:
    """Model çıktısını doğrular ve normalize eder; geçersiz/genel kavramları atar."""
    result = PostConcepts(slug)
    seen: dict[str, dict] = {}
    for item in (raw.get("concepts") or [])[:MAX_CONCEPTS]:
        raw_name = str(item.get("name", ""))
        name = normalize_name(raw_name)
        if CODE_LIKE_RE.search(raw_name) or not (2 <= len(name) <= 40) or name in GENERIC_NAMES:
            continue
        ctype = item.get("type") if item.get("type") in CONCEPT_TYPES else "concept"
        aliases = sorted({normalize_name(str(a)) for a in item.get("aliases") or []} - {name, ""})
        if name in seen:
            seen[name]["aliases"] = sorted(set(seen[name]["aliases"]) | set(aliases))
            continue
        seen[name] = {"name": name, "type": ctype, "aliases": aliases}
    result.concepts = list(seen.values())
    for rel in raw.get("relations") or []:
        source, target = (
            normalize_name(str(rel.get("source", ""))),
            normalize_name(str(rel.get("target", ""))),
        )
        kind = rel.get("kind") if rel.get("kind") in RELATION_KINDS else "related_to"
        if source in seen and target in seen and source != target:
            result.relations.append({"source": source, "target": target, "kind": kind})
    return result


def canonical_names(posts: list[PostConcepts]) -> dict[str, str]:
    """Ad/takma ad eşleşmesiyle kavramları birleştirir: ad -> en sık kullanılan kanonik ad.

    'ownership' bir yazıda ad, başka bir yazıda 'sahiplik'in takma adıysa ikisi birleşir.
    Basit tekil/çoğul farkı (trait objects / trait object) da birleştirilir.
    """
    freq = Counter(c["name"] for p in posts for c in p.concepts)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # daha sık geçen (eşitse daha kısa) ad kanonik olur
        keep, drop = sorted([ra, rb], key=lambda n: (-freq[n], len(n), n))
        parent[drop] = keep

    names = set(freq)
    for p in posts:
        for c in p.concepts:
            for alias in c["aliases"]:
                if alias in names:
                    union(c["name"], alias)
    for name in names:
        if name.endswith("s") and name[:-1] in names:
            union(name, name[:-1])
    return {name: find(name) for name in names}


def build_concept_graph(posts: list[PostConcepts], chunks: list[dict]) -> ConceptGraph:
    """Kanonik kavramlar, ABOUT, MENTIONS (metin eşleşmesi) ve kavram ilişkileri.

    Takma adların hepsi kavramda tutulur; sorgudaki Türkçe terimi ("sahiplik") kavrama
    bağlamak için kullanılır. MENTIONS ise yalnızca kavram adlarıyla kurulur: kanonik ad
    ve onunla birleşen diğer adlar (lifetimes, borrow checker gibi), o da yazının metninde
    gerçekten geçiyorsa. Takma adlar chunk eşleştirmesine sokulmaz, çünkü model metinde
    olmayan ya da yanlış eşleşen takma adlar üretebiliyor ("kopyalama" -> borrowing,
    "package" -> crate gibi). Yazar İngilizce terimleri çoğu zaman parantez içinde
    verdiği için ("Sahiplenme (Ownership)") ad bazlı eşleştirme yeterli kapsama sağlıyor.
    """
    canon = canonical_names(posts)
    post_text: dict[str, str] = defaultdict(str)
    for ch in chunks:
        post_text[ch["post_slug"]] += " " + ch["text"]
    evidence: dict[str, set[str]] = defaultdict(set)
    info: dict[str, dict] = {}
    about: set[tuple[str, str]] = set()
    rel_count: Counter[tuple[str, str, str]] = Counter()
    types: dict[str, Counter] = defaultdict(Counter)

    unsupported = 0
    for p in posts:
        supported: set[str] = set()
        for c in p.concepts:
            # Kanıt kontrolü: kavramın adı ya da bir karşılığı yazının metninde geçmeli.
            # Küçük modeller prompt'taki örnekleri ya da konu hakkındaki genel bilgisini
            # yazıda olmayan kavramlar olarak ekleyebiliyor (v3 tam çalıştırma: %38).
            pattern = mention_pattern([c["name"], *c["aliases"]])
            if not (pattern and pattern.search(post_text[p.slug])):
                unsupported += 1
                continue
            supported.add(c["name"])
            name = canon[c["name"]]
            entry = info.setdefault(name, {"name": name, "aliases": set()})
            entry["aliases"] |= set(c["aliases"]) | ({c["name"]} - {name})
            evidence[name].add(name)
            for term in [c["name"]]:
                pattern = mention_pattern([term])
                if pattern and pattern.search(post_text[p.slug]):
                    evidence[name].add(term)
            types[name][c["type"]] += 1
            about.add((p.slug, name))
        for r in p.relations:
            if r["source"] not in supported or r["target"] not in supported:
                continue  # kavramı atılmış ilişki
            a, b = canon[r["source"]], canon[r["target"]]
            if a != b:
                rel_count[(a, b, r["kind"])] += 1

    df = Counter(name for _, name in about)
    concepts = [
        {
            "name": name,
            "type": types[name].most_common(1)[0][0],
            "aliases": sorted(entry["aliases"]),
            "df": df[name],
            "mention_terms": sorted(evidence[name]),
        }
        for name, entry in sorted(info.items())
    ]

    patterns = {c["name"]: mention_pattern(c["mention_terms"]) for c in concepts}
    pilot_posts = {p.slug for p in posts}
    mentions = [
        {"chunk_id": ch["id"], "concept": name}
        for ch in chunks
        if ch["post_slug"] in pilot_posts
        for name, pattern in patterns.items()
        if pattern and pattern.search(ch["text"])
    ]
    return ConceptGraph(
        unsupported=unsupported,
        concepts=concepts,
        about=[{"slug": s, "concept": n} for s, n in sorted(about)],
        mentions=mentions,
        relations=[
            {"source": a, "target": b, "kind": k, "count": n}
            for (a, b, k), n in sorted(rel_count.items())
        ],
    )


def mention_pattern(terms: list[str]) -> re.Pattern | None:
    """Kavram adı/takma adları için kelime sınırlı, büyük/küçük harf duyarsız desen."""
    usable = sorted({t for t in terms if len(t) >= 3}, key=len, reverse=True)
    if not usable:
        return None
    alternation = "|".join(re.escape(t).replace(r"\ ", r"[\s_-]+") for t in usable)
    return re.compile(rf"(?<![\w])(?:{alternation})(?![\w])", re.IGNORECASE)


def cache_path(cache_dir: Path, slug: str) -> Path:
    return cache_dir / f"{slug}.json"


def load_cached(cache_dir: Path, slug: str) -> dict | None:
    path = cache_path(cache_dir, slug)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_cached(cache_dir: Path, slug: str, raw: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path(cache_dir, slug).write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
