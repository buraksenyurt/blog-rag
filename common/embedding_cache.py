"""Embedding cache (SQLite). Anahtar: sha256(profil + model + şablonlanmış metin).

Aynı metin aynı profille ikinci kez embed edilmez; Faz 2 de bu cache'i kullanır.
"""

from __future__ import annotations

import hashlib
import sqlite3
from array import array
from collections.abc import Iterable, Sequence
from pathlib import Path

from common.config import EmbeddingProfile


class EmbeddingCache:
    def __init__(self, path: Path, profile: EmbeddingProfile) -> None:
        self.profile = profile
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        )

    def key(self, formatted_input: str) -> str:
        raw = f"{self.profile.name}\x00{self.profile.model}\x00{formatted_input}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_many(self, keys: Sequence[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for start in range(0, len(keys), 500):
            part = list(keys[start : start + 500])
            placeholders = ",".join("?" * len(part))
            rows = self._db.execute(
                f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})", part
            )
            for key, blob in rows:
                vec = array("f")
                vec.frombytes(blob)
                found[key] = vec.tolist()
        return found

    def put_many(self, items: Iterable[tuple[str, Sequence[float]]]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
            ((k, array("f", v).tobytes()) for k, v in items),
        )
        self._db.commit()

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    def close(self) -> None:
        self._db.close()
