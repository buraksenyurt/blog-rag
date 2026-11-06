"""Ortam doğrulaması: LM Studio, embedding profili, LLM, Qdrant ve kaynak klasör.

Kullanım:
    uv run python -m scripts.smoke_test [--profile bge-m3] [--skip-llm]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable

from openai import OpenAI
from qdrant_client import QdrantClient

from common.config import ensure_writable, get_settings, load_embedding_profile
from common.embeddings import Embedder
from common.llm import LLM

results: list[tuple[str, bool, str]] = []


def check(name: str, fn: Callable[[], str]) -> bool:
    try:
        detail = fn()
        results.append((name, True, detail))
        return True
    except Exception as exc:  # noqa: BLE001 - teşhis amaçlı tüm hataları raporla
        results.append((name, False, f"{type(exc).__name__}: {exc}"))
        return False


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main() -> int:
    parser = argparse.ArgumentParser(description="blog-rag ortam doğrulaması")
    parser.add_argument("--profile", help="embedding_profiles.yaml içindeki profil adı")
    parser.add_argument("--skip-llm", action="store_true", help="LLM çağrısını atla")
    args = parser.parse_args()

    settings = get_settings()
    profile = load_embedding_profile(settings, args.profile)
    print(f"LM Studio : {settings.lmstudio_url}")
    print(f"Embedding : {profile.name} -> {profile.model} ({profile.dim} boyut)")
    print(f"LLM       : {settings.llm_model}")
    print(f"Qdrant    : {settings.qdrant_url}")
    print(f"Posts     : {settings.posts_dir}\n")

    def lmstudio_models() -> str:
        client = OpenAI(base_url=settings.lmstudio_url, api_key=settings.lmstudio_api_key)
        ids = [m.id for m in client.models.list().data]
        missing = [m for m in (profile.model, settings.llm_model) if m not in ids]
        if missing:
            raise RuntimeError(f"Yüklü değil: {missing}. Mevcut modeller: {ids}")
        return f"{len(ids)} model listelendi"

    def embedding_sanity() -> str:
        embedder = Embedder(settings, profile)
        query = embedder.embed_query("Rust dilinde değişken tanımlama örnekleri")
        rust_doc, wcf_doc = embedder.embed_documents(
            [
                "Rust'ta değişkenler varsayılan olarak değiştirilemez; "
                "let mut x = 5; ile değiştirilebilir hale gelir.",
                "WCF servislerinde BasicHttpBinding ve NetTcpBinding farklı "
                "iletişim senaryoları için kullanılır.",
            ],
            titles=["Rust Değişkenler", "WCF Binding Türleri"],
        )
        relevant, irrelevant = cosine(query, rust_doc), cosine(query, wcf_doc)
        if relevant <= irrelevant:
            raise RuntimeError(f"Alakalı skor düşük: rust={relevant:.3f} wcf={irrelevant:.3f}")
        return f"boyut={len(query)} rust={relevant:.3f} > wcf={irrelevant:.3f}"

    def llm_reply() -> str:
        answer = LLM(settings).chat(
            [{"role": "user", "content": "Tek cümleyle: Rust'ta 'let mut' ne işe yarar?"}],
            max_tokens=80,
        )
        return answer.strip().replace("\n", " ")[:120]

    def qdrant_up() -> str:
        client = QdrantClient(url=settings.qdrant_url, timeout=5)
        names = [c.name for c in client.get_collections().collections]
        return f"erişildi, collection'lar: {names or '-'}"

    def posts_readable() -> str:
        if not settings.posts_dir.is_dir():
            raise FileNotFoundError(settings.posts_dir)
        count = len(list(settings.posts_dir.glob("*.md")))
        try:
            ensure_writable(settings.posts_dir / "x.tmp", settings)
        except PermissionError:
            return f"{count} markdown dosyası, yazma koruması aktif"
        raise RuntimeError("Yazma koruması çalışmadı!")

    def data_dir_writable() -> str:
        target = ensure_writable(settings.data_dir, settings)
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    if check("LM Studio modelleri", lmstudio_models):
        check("Embedding (sorgu/doküman)", embedding_sanity)
        if not args.skip_llm:
            check("LLM cevabı", llm_reply)
    check("Qdrant", qdrant_up)
    check("Kaynak klasör (salt okunur)", posts_readable)
    check("data/ klasörü", data_dir_writable)

    print()
    for name, ok, detail in results:
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} kontrol başarılı")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
