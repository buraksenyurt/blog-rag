"""_posts klasöründeki Jekyll yazılarını okur (SALT OKUNUR) ve gövdeyi temizler."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter

from common.config import Settings, get_settings

BASE_URL = "https://buraksenyurt.github.io"

FILENAME_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(.+)$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
LIQUID_RAW_RE = re.compile(r"\{%-?\s*(?:raw|endraw)\s*-?%\}")
INTERNAL_LINK_RE = re.compile(
    r"\]\((?:https?://buraksenyurt\.github\.io)?/(\d{4})/(\d{2})/(\d{2})/([^/)#?\s]+)/?[^)]*\)"
)
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HTML_IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
MD_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\([^)]*\)")
# Sadece bilinen HTML etiketleri temizlenir; `List<T>` gibi metinler korunur.
HTML_TAG_RE = re.compile(
    r"</?(?:br|p|div|span|center|b|i|u|strong|em|font|a|sup|sub|hr)\b[^>]*/?>", re.IGNORECASE
)
BLANK_LINES_RE = re.compile(r"\n{3,}")


@dataclass
class Post:
    slug: str
    title: str
    date: dt.datetime
    tags: list[str]
    categories: list[str]
    body: str
    path: Path
    links: list[str] = field(default_factory=list)

    @property
    def year(self) -> int:
        return self.date.year

    @property
    def url(self) -> str:
        match = FILENAME_RE.match(self.slug)
        if not match:
            return BASE_URL
        y, m, d, title = match.groups()
        return f"{BASE_URL}/{y}/{m}/{d}/{title}/"


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(v).strip() for v in value if str(v).strip()]


def _as_datetime(value: object, slug: str) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        return dt.datetime.fromisoformat(value.strip())
    match = FILENAME_RE.match(slug)
    if match:
        y, m, d, _ = match.groups()
        return dt.datetime(int(y), int(m), int(d))
    raise ValueError(f"Tarih okunamadı: {slug}")


def extract_internal_links(markdown: str) -> list[str]:
    """Blog içi linkleri hedef yazının slug'ına çevirir (ör. 2009-04-22-koleksiyon-...)."""
    seen: dict[str, None] = {}
    for y, m, d, title in INTERNAL_LINK_RE.findall(markdown):
        seen[f"{y}-{m}-{d}-{title}"] = None
    return list(seen)


def clean_body(markdown: str) -> str:
    """Liquid raw etiketlerini, görselleri ve HTML kalıntılarını temizler.

    Kod bloklarının içine dokunulmaz.
    """
    markdown = LIQUID_RAW_RE.sub("", markdown)
    out: list[str] = []
    in_code = False
    for line in markdown.splitlines():
        if FENCE_RE.match(line):
            in_code = not in_code
            out.append(line.strip())
            continue
        if in_code:
            out.append(line)
            continue
        line = MD_IMAGE_RE.sub("", line)
        line = HTML_IMG_RE.sub("", line)
        line = MD_LINK_RE.sub(r"\1", line)
        line = HTML_TAG_RE.sub("", line)
        out.append(line.rstrip())
    text = "\n".join(out)
    return BLANK_LINES_RE.sub("\n\n", text).strip()


def load_post(path: Path) -> Post:
    raw = path.read_text(encoding="utf-8")
    doc = frontmatter.loads(raw)
    slug = path.stem
    return Post(
        slug=slug,
        title=str(doc.get("title") or slug).strip(),
        date=_as_datetime(doc.get("date"), slug),
        tags=_as_list(doc.get("tags")),
        categories=_as_list(doc.get("categories")),
        body=clean_body(doc.content),
        path=path,
        links=extract_internal_links(doc.content),
    )


def iter_posts(settings: Settings | None = None) -> Iterator[Post]:
    """Kapsamdaki yazıları tarih sırasıyla döner; exclude_categories dışarıda kalır."""
    settings = settings or get_settings()
    excluded = {c.casefold() for c in settings.exclude_categories}
    for path in sorted(settings.posts_dir.glob("*.md")):
        post = load_post(path)
        if excluded & {c.casefold() for c in post.categories}:
            continue
        yield post
