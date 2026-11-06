"""data/chunks.jsonl -> embedding (cache'li) -> Qdrant.

Kullanım:
    uv run python -m phase1_vector.ingest                 # aktif profil
    uv run python -m phase1_vector.ingest --limit 200     # hızlı deneme
    uv run python -m phase1_vector.ingest --profile bge-m3 --recreate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models

from common.config import EmbeddingProfile, ensure_writable, get_settings, load_embedding_profile
from common.embedding_cache import EmbeddingCache
from common.embeddings import Embedder
from phase1_vector.store import ensure_collection, existing_ids, point_id

UPSERT_BATCH = 128
WINDOW = 512


@dataclass
class IngestReport:
    collection: str
    chunks: int
    embedded: int
    cached: int
    deleted_stale: int
    seconds: float


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


def embedding_title(chunk: dict) -> str:
    section = chunk.get("section_path")
    return f"{chunk['title']} > {section}" if section else chunk["title"]


def embedding_input(chunk: dict, embedder: Embedder) -> str:
    return embedder.format_document(chunk["text"], embedding_title(chunk))


def batched(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def check_tokens(chunks: list[dict], embedder: Embedder, profile: EmbeddingProfile, n: int) -> None:
    """En büyük n chunk'ın gerçek token sayısını ölçer, model sınırını aşanları raporlar."""
    if n <= 0 or not chunks:
        return
    largest = sorted(chunks, key=lambda c: len(c["text"]), reverse=True)[:n]
    ratios: list[float] = []
    over: list[tuple[str, int]] = []
    for chunk in largest:
        text = embedding_input(chunk, embedder)
        tokens = embedder.count_tokens(text)
        if tokens is None:
            print("! LM Studio usage döndürmedi; token kontrolü atlandı.")
            return
        ratios.append(len(text) / tokens)
        if tokens > profile.max_tokens:
            over.append((chunk["id"], tokens))
    print(
        f"Token kontrolü ({len(largest)} en büyük chunk): "
        f"ortalama {sum(ratios) / len(ratios):.2f} karakter/token, "
        f"en kötü {min(ratios):.2f}"
    )
    for chunk_id, tokens in over:
        print(f"! {chunk_id}: {tokens} token > {profile.max_tokens} (model kırpacak)")


def ingest(
    chunks: list[dict],
    embedder: Embedder,
    cache: EmbeddingCache,
    client: QdrantClient,
    recreate: bool = False,
    delete_stale: bool = True,
) -> IngestReport:
    started = time.perf_counter()
    profile = embedder.profile
    name = ensure_collection(client, profile, recreate=recreate)

    embedded = cached = 0
    wanted: set[str] = set()
    total = len(chunks)
    # Bellek için pencere pencere ilerlenir: cache'e bak -> eksikleri embed et -> upsert.
    for window_start in range(0, total, WINDOW):
        window = chunks[window_start : window_start + WINDOW]
        inputs = [embedding_input(c, embedder) for c in window]
        keys = [cache.key(i) for i in inputs]
        vectors = cache.get_many(keys)
        cached += len(vectors)

        missing = [i for i, k in enumerate(keys) if k not in vectors]
        for batch in batched(missing, embedder.batch_size):
            new_vectors = embedder.embed_raw([inputs[i] for i in batch])
            pairs = list(zip(batch, new_vectors, strict=True))
            cache.put_many((keys[i], v) for i, v in pairs)
            vectors.update({keys[i]: v for i, v in pairs})
        embedded += len(missing)

        for batch in batched(list(range(len(window))), UPSERT_BATCH):
            points = []
            for i in batch:
                pid = point_id(window[i]["id"])
                wanted.add(pid)
                points.append(
                    models.PointStruct(id=pid, vector=vectors[keys[i]], payload=window[i])
                )
            client.upsert(name, points=points, wait=True)

        done = window_start + len(window)
        elapsed = time.perf_counter() - started
        rate = embedded / elapsed if elapsed and embedded else 0
        eta = f", kalan ~{(total - done) / rate:.0f}s" if rate and embedded == done else ""
        print(
            f"  {done}/{total} chunk (yeni embed: {embedded}, cache: {cached}, {elapsed:.0f}s{eta})"
        )

    stale: list[str] = []
    if delete_stale:
        stale = sorted(existing_ids(client, name) - wanted)
        for batch in batched(stale, 1000):
            client.delete(name, points_selector=models.PointIdsList(points=batch), wait=True)

    return IngestReport(
        collection=name,
        chunks=len(chunks),
        embedded=embedded,
        cached=cached,
        deleted_stale=len(stale),
        seconds=round(time.perf_counter() - started, 1),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Qdrant ingest")
    parser.add_argument("--profile", help="embedding_profiles.yaml içindeki profil adı")
    parser.add_argument("--limit", type=int, help="ilk N chunk (deneme amaçlı)")
    parser.add_argument("--recreate", action="store_true", help="collection'ı silip yeniden kur")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--check-tokens",
        type=int,
        default=10,
        metavar="N",
        help="en büyük N chunk'ın gerçek token sayısını ölç (0 = kapalı)",
    )
    args = parser.parse_args()

    settings = get_settings()
    profile = load_embedding_profile(settings, args.profile)
    chunks = read_chunks(settings.data_dir / "chunks.jsonl", args.limit)
    cache_path = ensure_writable(settings.data_dir / "embeddings" / f"{profile.name}.sqlite")

    embedder = Embedder(settings, profile, batch_size=args.batch_size)
    cache = EmbeddingCache(cache_path, profile)
    client = QdrantClient(url=settings.qdrant_url, timeout=60)

    print(f"Profil: {profile.name} ({profile.model}, {profile.dim}) | chunk: {len(chunks)}")
    check_tokens(chunks, embedder, profile, args.check_tokens)
    # --limit ile kısmi çalıştırmada collection'daki diğer kayıtlar silinmesin.
    report = ingest(chunks, embedder, cache, client, args.recreate, delete_stale=not args.limit)
    cache.close()

    print(
        f"\nCollection : {report.collection}\n"
        f"Chunk      : {report.chunks} (yeni embed: {report.embedded}, cache: {report.cached})\n"
        f"Silinen    : {report.deleted_stale} eski kayıt\n"
        f"Süre       : {report.seconds}s\n"
        f"Qdrant'taki kayıt: {client.count(report.collection).count}\n"
        f"Dashboard  : {settings.qdrant_url}/dashboard#/collections/{report.collection}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
