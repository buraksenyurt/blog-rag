# AGENTS.md — blog-rag

Bu dosya, bu repoda çalışan AI kodlama ajanları (Claude Code, GitHub Copilot, Codex, Cursor vb.) için yazılmış çalışma talimatlarını içerir. Ayrıntılı plan [PLAN.md](./PLAN.md) dosyasında yer alır. Bu dosya ise kuralları ve çalışma şeklini tanımlar.

## Proje

Burak Selim Şenyurt'un teknik blog yazıları (`buraksenyurt.github.io/_posts`) üzerinde iki farklı RAG deneyi yapılacak:

- **Faz 1:** Klasik vektör RAG (Qdrant)
- **Faz 2:** Graph tabanlı RAG (Neo4j)

Bir soru sorulduğunda, LLM'e gitmeden önce ilgili yazı parçaları bulunacak ve context'e eklenecek.

Örnek soru: *"Rust dilinde değişken kullanımları ile ilgili örnekler yaz."*

Mimari için `docs/images/rag_high_level_view.png` dosyasına bakın.

**Durum:** Faz 1 / Adım 1 (iskelet), Adım 2 (loader + chunker) ve Adım 3 (embedding cache + Qdrant ingest) tamamlandı. Sıradaki iş *Adım 4: retrieval*. Tamamlanan adımlar PLAN.md içinde ✅ ile işaretlenir.

## Değişmez Kurallar

1. **Kaynak blog reposu salt okunurdur.** `../buraksenyurt.github.io` (ya da `POSTS_DIR` ile verilen yol) altında hiçbir dosya oluşturulmayacak, değiştirilmeyecek veya silinmeyecek. Orada git komutu da çalıştırılmayacak. Bu yol sadece okunmak için açılır. Bütün çıktılar bu repo içinde kalmalı (`data/`, `eval/results/`).
2. **Her şey lokal çalışır.** Embedding ve LLM için LM Studio'nun OpenAI uyumlu API'si (`LMSTUDIO_URL`, varsayılan `http://localhost:1234/v1`) kullanılır. Bulut LLM ya da embedding API'si (OpenAI, Anthropic, Cohere vb.) eklenmeyecek.
3. **Veritabanları docker-compose ile çalışır.** Qdrant ve Neo4j `docker-compose.yml` içinde tanımlanır. Bunları host'a kurmaya çalışmayın.
4. **Embedding prefix'leri kodda sabit yazılmaz.** Doküman ve sorgu şablonları `embedding_profiles.yaml` içinde tutulur. Kod sadece `embed_documents()` ve `embed_query()` fonksiyonlarını çağırır. Collection/index adı ve cache anahtarı profil adını içermelidir (`blog_posts__{profil}`).
5. **Kapsam:** `Foto İpucu` kategorisindeki yazılar indexe alınmaz. Bu kural `exclude_categories` ayarıyla uygulanır, koda gömülmez.
6. **PLAN.md'deki kararlar** (dil, veritabanları, modeller, chunking kuralları) kullanıcıya sorulmadan değiştirilmez. Daha iyi bir alternatif görürsen önce öner, onay gelirse PLAN.md'yi de güncelle.
7. **Gizli bilgi:** `.env` commit edilmez. Yeni bir ayar eklendiğinde `.env.example` da güncellenir.

## Teknoloji ve Kod Kuralları

- Python 3.12, paket yönetimi için `uv` (`uv add`, `uv run`). `pip install` kullanılmaz.
- Ayarlar `pydantic-settings` ile `.env` dosyasından okunur. Kod içinde sabit URL ya da model adı olmaz.
- Kod tanımlayıcıları İngilizce olur. Dokümantasyon, kullanıcıya dönen mesajlar ve LLM prompt'ları Türkçe olur.
- Modüller `common/`, `phase1_vector/`, `phase2_graph/` ve `eval/` altında yer alır. Faz modülleri ortak kodu `common/` üzerinden kullanır, birbirini import etmez.
- Ingest işlemleri idempotent olmalı: deterministik ID'ler ve embedding cache kullanılır. Bir ingest'i tekrar çalıştırmak veriyi çoğaltmamalı.
- Chunking sırasında markdown kod blokları asla ortadan bölünmez.

## Komutlar

> Henüz yazılmamış modüller (ingest, ask, eval) ilgili adımlarda eklenecek.

```bash
uv sync                                      # bağımlılıkları kur
copy .env.example .env                       # ayarlar (PowerShell: Copy-Item)
uv run python -m scripts.smoke_test          # LM Studio / Qdrant / kaynak klasör kontrolü
uv run python -m scripts.build_chunks        # data/chunks.jsonl + chunk_stats.json (--show SLUG ile tek yazı)
docker compose up -d qdrant                  # Faz 1 veritabanı
docker compose up -d neo4j                   # Faz 2 veritabanı
uv run python -m phase1_vector.ingest        # indexle (--limit N, --profile X, --recreate)
uv run python -m phase1_vector.ask "soru"    # sor (--debug ile retrieval skorları)
uv run python -m eval.run_eval --phase 1     # değerlendir
uv run ruff check . && uv run pytest         # lint + test
```

## Çalışma Şekli

- Adımları PLAN.md içindeki *Adım Sırası* bölümüne göre, birer birer ilerlet. Bir adım tamamlandığında PLAN.md'deki ilgili checkbox'ı işaretle.
- Her adımda çalıştığını gösteren küçük bir doğrulama olmalı: smoke test, istatistik çıktısı ya da örnek sorgu sonucu.
- LM Studio veya Docker ayakta değilse, bunu taklit eden sahte bir çözüm yazma. Durumu kullanıcıya bildir.
- Commit mesajları Conventional Commits formatında olmalı (`feat(phase1): ...`, `fix(chunker): ...`, `docs(plan): ...`). Push etmeden önce kullanıcıya sor.
- `data/`, docker volume'ları ve embedding cache'i git'e girmez.
