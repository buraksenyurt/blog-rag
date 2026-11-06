"""Sorgudaki teknoloji adlarından Qdrant filtresi çıkarır.

Serbest tag eşleştirmesi bu blogda riskli ("as", "is", "get", "list" gibi tag'ler var).
Bu yüzden sadece aşağıdaki küratörlü sözlük kullanılır. Tag adları veri setinde
geçtiği yazımla (büyük/küçük harf dahil) yazılmalıdır.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from qdrant_client import models

# anahtar: (sorguda aranacak ifadeler, tags, code_langs)
TECHNOLOGIES: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    "rust": (("rust",), ("rust", "rust-lang"), ("rust",)),
    "go": (("go", "golang"), ("golang",), ("go",)),
    "csharp": (("c#", "csharp"), ("csharp",), ("csharp",)),
    "python": (("python",), ("python",), ("python",)),
    "ruby": (("ruby",), ("ruby-lang",), ("ruby",)),
    "javascript": (("javascript", "js"), ("javascript",), ("javascript",)),
    "typescript": (("typescript", "ts"), ("typescript",), ("typescript",)),
    "java": (("java",), ("java",), ("java",)),
    "scala": (("scala",), ("scala",), ("scala",)),
    "ocaml": (("ocaml",), ("ocaml",), ("ocaml",)),
    "zig": (("zig",), ("zig",), ("zig",)),
    "cpp": (("c++", "cpp"), ("c++",), ("cpp",)),
    "sql": (("sql",), ("sql",), ("sql",)),
    "docker": (("docker",), ("docker",), ("dockerfile",)),
    "kubernetes": (("kubernetes", "k8s"), ("kubernetes", "k8s"), ()),
    "wcf": (("wcf",), ("windows-communication-foundation",), ()),
    "wpf": (("wpf",), ("windows-presentation-foundation",), ()),
    "linq": (("linq",), ("language-integrated-query",), ()),
    "tpl": (("tpl", "task parallel library"), ("task-parallel-library",), ()),
    "entity-framework": (("entity framework", "ef core"), ("entity-framework",), ()),
    "nodejs": (("node.js", "nodejs"), ("node", "nodejs", "node.js"), ()),
    "tokio": (("tokio",), ("tokio",), ()),
    "bevy": (("bevy",), ("bevy",), ()),
    "tauri": (("tauri",), ("tauri",), ()),
    "wasm": (("wasm", "webassembly"), ("wasm", "WebAssembly"), ()),
    "grpc": (("grpc",), ("gRPC",), ("protobuf",)),
    "mongodb": (("mongodb", "mongo"), ("mongodb",), ()),
    "redis": (("redis",), ("redis",), ()),
    "rabbitmq": (("rabbitmq",), ("rabbitmq",), ()),
    "postgresql": (("postgresql", "postgres"), ("postgresql", "PostgreSQL"), ()),
    "sqlite": (("sqlite",), ("sqlite",), ()),
    "elasticsearch": (("elasticsearch",), ("elasticsearch",), ()),
    "blazor": (("blazor",), ("blazor",), ()),
}

APOSTROPHE_SUFFIX_RE = re.compile(r"['’]\w+")


@dataclass
class QueryFilter:
    technologies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    code_langs: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.tags or self.code_langs)

    def to_qdrant(self) -> models.Filter:
        conditions: list[models.Condition] = []
        if self.tags:
            conditions.append(
                models.FieldCondition(key="tags", match=models.MatchAny(any=self.tags))
            )
        if self.code_langs:
            conditions.append(
                models.FieldCondition(key="code_langs", match=models.MatchAny(any=self.code_langs))
            )
        return models.Filter(should=conditions)

    def matches(self, payload: dict) -> bool:
        return bool(
            set(payload.get("tags", [])) & set(self.tags)
            or set(payload.get("code_langs", [])) & set(self.code_langs)
        )


def normalize_query(query: str) -> str:
    """Küçük harfe çevirir ve Türkçe kesme işaretli ekleri atar (Rust'ta -> rust)."""
    return APOSTROPHE_SUFFIX_RE.sub("", query.lower())


def _contains(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(phrase)}(?![\w#+])", text) is not None


def detect_filter(query: str) -> QueryFilter:
    text = normalize_query(query)
    result = QueryFilter()
    for name, (aliases, tags, langs) in TECHNOLOGIES.items():
        if any(_contains(text, alias) for alias in aliases):
            result.technologies.append(name)
            result.tags.extend(t for t in tags if t not in result.tags)
            result.code_langs.extend(lang for lang in langs if lang not in result.code_langs)
    return result
