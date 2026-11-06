"""Markdown farkındalıklı chunking.

Kurallar (PLAN.md 1.3):
- Önce başlıklara göre bölümlenir; her chunk `section_path` taşır.
- Bölümler hedef boyutu aşarsa paragraf sınırlarından bölünür, ardışık chunk'lar
  arasında cümle bazlı küçük bir overlap bırakılır.
- Kod blokları ortadan bölünmez. Kod yeni bir chunk'a düşüyorsa önündeki açıklama
  paragrafı da kodla birlikte o chunk'a taşınır. Sadece embedding modelinin sınırını aşan
  dev kod blokları satır sınırlarından parçalanır (`code_split=True`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from common.loader import FENCE_RE, Post

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…:])\s+")
LANG_ALIASES = {
    "golang": "go",
    "yml": "yaml",
    "cs": "csharp",
    "c#": "csharp",
    "js": "javascript",
    "ts": "typescript",
    "sh": "bash",
    "shell": "bash",
    "powershell": "powershell",
    "ps1": "powershell",
    "c++": "cpp",
}
NON_LANGS = {"", "text", "plaintext", "txt", "console", "output"}


@dataclass(frozen=True)
class ChunkerConfig:
    target_tokens: int = 450
    max_tokens: int = 1500
    min_tokens: int = 60
    overlap_tokens: int = 70
    intro_max_tokens: int = 250
    chars_per_token: float = 3.2


@dataclass
class Block:
    kind: str  # "heading" | "text" | "code"
    text: str
    level: int = 0
    lang: str | None = None
    carried: bool = False
    split: bool = False


@dataclass
class Chunk:
    id: str
    post_slug: str
    title: str
    date: str
    year: int
    url: str
    tags: list[str]
    categories: list[str]
    section_path: str
    chunk_index: int
    text: str
    n_tokens: int
    has_code: bool
    code_langs: list[str] = field(default_factory=list)
    code_split: bool = False


def estimate_tokens(text: str, cfg: ChunkerConfig) -> int:
    return math.ceil(len(text) / cfg.chars_per_token)


def normalize_lang(info: str) -> str | None:
    lang = info.strip().split()[0].lower() if info.strip() else ""
    lang = LANG_ALIASES.get(lang, lang)
    return None if lang in NON_LANGS else lang


def parse_blocks(markdown: str) -> list[Block]:
    """Markdown'ı başlık / paragraf / kod bloklarına ayırır (kod içindeki # başlık sayılmaz)."""
    blocks: list[Block] = []
    para: list[str] = []
    code: list[str] | None = None
    lang: str | None = None

    def flush_para() -> None:
        if para:
            text = "\n".join(para).strip()
            if text:
                blocks.append(Block("text", text))
            para.clear()

    for line in markdown.splitlines():
        if code is not None:
            code.append(line)
            if FENCE_RE.match(line):
                blocks.append(Block("code", "\n".join(code), lang=lang))
                code, lang = None, None
            continue
        if FENCE_RE.match(line):
            flush_para()
            stripped = line.strip()
            code = [stripped]
            lang = normalize_lang(stripped.lstrip("`~"))
            continue
        heading = HEADING_RE.match(line)
        if heading:
            flush_para()
            blocks.append(Block("heading", heading.group(2), level=len(heading.group(1))))
            continue
        if not line.strip():
            flush_para()
            continue
        para.append(line)

    flush_para()
    if code is not None:  # kapanmamış fence: dosya sonuna kadar kod say
        fence = code[0][:3]
        blocks.append(Block("code", "\n".join([*code, fence]), lang=lang))
    return blocks


def _split_sentences(text: str, limit: int, cfg: ChunkerConfig) -> list[str]:
    parts: list[str] = []
    current = ""
    for sentence in SENTENCE_SPLIT_RE.split(text):
        candidate = f"{current} {sentence}".strip()
        if current and estimate_tokens(candidate, cfg) > limit:
            parts.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _split_code(block: Block, cfg: ChunkerConfig) -> list[Block]:
    lines = block.text.splitlines()
    opening, closing, body = lines[0], lines[-1], lines[1:-1]
    limit = cfg.max_tokens - 20
    parts: list[list[str]] = []
    current: list[str] = []
    last_blank = -1
    for line in body:
        current.append(line)
        if not line.strip():
            last_blank = len(current) - 1
        if estimate_tokens("\n".join(current), cfg) > limit and len(current) > 1:
            cut = last_blank if last_blank > 0 else len(current) - 1
            parts.append(current[:cut])
            current = current[cut:]
            last_blank = -1
    if current:
        parts.append(current)
    return [
        Block("code", "\n".join([opening, *p, closing]), lang=block.lang, split=True)
        for p in parts
        if any(x.strip() for x in p)
    ]


def _expand_oversized(blocks: list[Block], cfg: ChunkerConfig) -> list[Block]:
    out: list[Block] = []
    for block in blocks:
        if estimate_tokens(block.text, cfg) <= cfg.max_tokens:
            out.append(block)
        elif block.kind == "code":
            out.extend(_split_code(block, cfg))
        elif _is_table(block.text):
            out.extend(_split_table(block, cfg))
        else:
            out.extend(
                Block("text", p) for p in _split_sentences(block.text, cfg.target_tokens, cfg)
            )
    return out


def _is_table(text: str) -> bool:
    lines = text.splitlines()
    return len(lines) > 2 and all(line.lstrip().startswith("|") for line in lines[:2])


def _split_table(block: Block, cfg: ChunkerConfig) -> list[Block]:
    """Büyük tabloları satır sınırından böler; başlık satırları her parçada tekrarlanır."""
    lines = block.text.splitlines()
    header, rows = lines[:2], lines[2:]
    parts: list[list[str]] = []
    current: list[str] = []
    for row in rows:
        if (
            current
            and estimate_tokens("\n".join([*header, *current, row]), cfg) > cfg.target_tokens
        ):
            parts.append(current)
            current = []
        current.append(row)
    if current:
        parts.append(current)
    return [Block("text", "\n".join([*header, *p])) for p in parts]


def _tail_overlap(previous: list[Block], cfg: ChunkerConfig) -> list[Block]:
    """Önceki chunk bir paragrafla bitiyorsa son cümlelerini yeni chunk'a taşır."""
    last = previous[-1]
    if last.kind != "text" or last.carried:
        return []
    tail: list[str] = []
    for sentence in reversed(SENTENCE_SPLIT_RE.split(last.text)):
        if estimate_tokens(" ".join([sentence, *tail]), cfg) > cfg.overlap_tokens:
            break
        tail.insert(0, sentence)
    if not tail:
        return []
    return [Block("text", " ".join(tail), carried=True)]


def _sections(blocks: list[Block]) -> list[tuple[str, list[Block]]]:
    sections: list[tuple[str, list[Block]]] = []
    stack: list[tuple[int, str]] = []
    current: list[Block] = []
    path = ""
    for block in blocks:
        if block.kind == "heading":
            if current:
                sections.append((path, current))
                current = []
            stack = [(lvl, t) for lvl, t in stack if lvl < block.level]
            stack.append((block.level, block.text))
            path = " > ".join(t for _, t in stack)
        else:
            current.append(block)
    if current:
        sections.append((path, current))
    return sections


def _pack(blocks: list[Block], cfg: ChunkerConfig) -> list[list[Block]]:
    groups: list[list[Block]] = []
    current: list[Block] = []
    tokens = 0

    def emit(group: list[Block]) -> None:
        if group and not all(b.carried for b in group):
            groups.append(group)

    for block in blocks:
        size = estimate_tokens(block.text, cfg)
        if current and tokens + size > cfg.target_tokens:
            last = current[-1]
            last_size = estimate_tokens(last.text, cfg)
            if (
                block.kind == "code"
                and last.kind == "text"
                and not last.carried
                and last_size <= cfg.intro_max_tokens
                and last_size + size <= cfg.max_tokens
            ):
                # Kodun açıklama paragrafı kodla birlikte yeni chunk'a taşınır.
                current.pop()
                emit(current)
                current, tokens = [last], last_size
            else:
                emit(current)
                carry = _tail_overlap(current, cfg)
                carry_size = sum(estimate_tokens(b.text, cfg) for b in carry)
                if carry_size + size > cfg.max_tokens:
                    carry, carry_size = [], 0
                current, tokens = carry, carry_size
        current.append(block)
        tokens += size
    emit(current)
    return groups


def chunk_post(post: Post, cfg: ChunkerConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkerConfig()
    raw: list[tuple[str, list[Block]]] = []
    for path, blocks in _sections(parse_blocks(post.body)):
        for group in _pack(_expand_oversized(blocks, cfg), cfg):
            raw.append((path, group))

    # Çok küçük parçaları komşusuyla birleştir.
    merged: list[tuple[str, list[Block]]] = []
    for path, group in raw:
        size = sum(estimate_tokens(b.text, cfg) for b in group)
        if merged and size < cfg.min_tokens:
            prev_path, prev = merged[-1]
            if sum(estimate_tokens(b.text, cfg) for b in prev) + size <= cfg.max_tokens:
                merged[-1] = (prev_path, prev + group)
                continue
        merged.append((path, group))
    if len(merged) > 1:
        first_size = sum(estimate_tokens(b.text, cfg) for b in merged[0][1])
        if first_size < cfg.min_tokens:
            merged[1] = (merged[1][0], merged[0][1] + merged[1][1])
            merged.pop(0)

    chunks: list[Chunk] = []
    for index, (path, group) in enumerate(merged):
        text = "\n\n".join(b.text for b in group)
        langs = sorted({b.lang for b in group if b.kind == "code" and b.lang})
        chunks.append(
            Chunk(
                id=f"{post.slug}#{index:04d}",
                post_slug=post.slug,
                title=post.title,
                date=post.date.date().isoformat(),
                year=post.year,
                url=post.url,
                tags=post.tags,
                categories=post.categories,
                section_path=path,
                chunk_index=index,
                text=text,
                n_tokens=estimate_tokens(text, cfg),
                has_code=any(b.kind == "code" for b in group),
                code_langs=langs,
                code_split=any(b.split for b in group),
            )
        )
    return chunks
