"""data/chunks.jsonl okuma ve embedding girdisi üretimi (Faz 1 ve Faz 2 ortak).

Embedding girdisi her iki fazda birebir aynı olmalı; aksi halde embedding cache'i
ıskalanır ve chunk'lar boşuna yeniden embed edilir.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol


class DocumentFormatter(Protocol):
    def format_document(self, text: str, title: str | None = None) -> str: ...


def read_chunks(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} yok. Önce: uv run python -m scripts.build_chunks")
    chunks: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
            if limit and len(chunks) >= limit:
                break
    return chunks


def read_post_links(path: Path) -> dict[str, list[str]]:
    """data/post_links.jsonl -> {kaynak slug: [hedef slug, ...]} (dosya yoksa boş)."""
    if not path.exists():
        return {}
    links: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            links[row["slug"]] = row["links"]
    return links


def embedding_title(chunk: dict) -> str:
    section = chunk.get("section_path")
    return f"{chunk['title']} > {section}" if section else chunk["title"]


def embedding_input(chunk: dict, formatter: DocumentFormatter) -> str:
    return formatter.format_document(chunk["text"], embedding_title(chunk))


def batched(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
