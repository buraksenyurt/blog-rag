# blog-rag — Uygulama Planı

Amaç: `buraksenyurt.github.io/_posts` içeriğini kaynak alan iki RAG sistemi kurmak. Soru LLM'e gitmeden önce ilgili parçalar bulunacak ve context'e eklenecek.
Faz 1 klasik vektör RAG olacak (Qdrant), Faz 2 graph tabanlı RAG olacak (Neo4j).

Örnek soru: *"Rust dilinde değişken kullanımları ile ilgili örnekler yaz."*

---

## 0. Kararlar ve Kısıtlar

| Konu | Karar |
|---|---|
| Dil | Python 3.12, paket yönetimi `uv` |
| Embedding | LM Studio → profil tabanlı. Varsayılan profil `embeddinggemma` (768 boyut). Alternatif olarak `bge-m3` (prefix'siz) seçilebilir |
| LLM | LM Studio → `meta-llama-3-8b-instruct` ile başlanacak. Model config'ten değişebilir olacak, Qwen3-8B / Gemma-3-12B ile kıyaslanacak |
| API | LM Studio'nun OpenAI uyumlu endpoint'i (`http://localhost:1234/v1`), `openai` Python paketi |
| Vektör DB | Qdrant (docker-compose) |
| Graph DB | Neo4j 5.x Community + APOC (docker-compose). Native vector index de kullanılacak |
| Kaynak | `../buraksenyurt.github.io/_posts` **sadece okunacak**. Foto İpucu kategorisi kapsam dışı (868 yazı indexlenecek). Hiçbir script oraya yazmayacak, path config'te read-only olarak tutulacak |
| Çıktılar | Bütün ara çıktılar (`data/`) ve kod `blog-rag` altında duracak |

### Kaynak veri hakkında
- 1030 markdown yazı var (2003–2026). Hepsi Jekyll front matter içeriyor (`title`, `date`, `tags`, `categories`).
- Medyan yazı uzunluğu ~1.200 kelime, en uzun yazı ~20.000 kelime.
- Yazılarda toplam ~5.350 kod bloğu var. 36 yazıda ` ```rust ` bloğu geçiyor.
- En sık geçen tag/kategoriler: `csharp`, `Programlama Dilleri`, `windows-communication-foundation`, `rest-api`, `ruby-lang`…
- Front matter'daki tag/kategori bilgisi Faz 1'de filtre olarak, Faz 2'de doğrudan graph düğümü olarak kullanılacak.

### Embedding profilleri (prefix'ler config'te tutulacak)
Embedding modeli ve modelin beklediği prompt formatı kodun içinde sabit olmayacak. Bunlar `embedding_profiles.yaml` dosyasında tanımlanacak. Kod sadece `embed_documents(texts, title)` ve `embed_query(text)` fonksiyonlarını çağıracak. Hangi prefix'in ekleneceğini aktif profil belirleyecek.

```yaml
active: embeddinggemma
profiles:
  embeddinggemma:
    model: text-embedding-embeddinggemma-300m
    dim: 768
    max_tokens: 2048
    doc_template: "title: {title} | text: {text}"
    query_template: "task: search result | query: {text}"
  bge-m3:                      # prefix istemeyen alternatif
    model: text-embedding-bge-m3
    dim: 1024
    max_tokens: 8192
    doc_template: "{text}"
    query_template: "{text}"
  multilingual-e5-large:
    model: text-embedding-multilingual-e5-large-instruct
    dim: 1024
    max_tokens: 512
    doc_template: "{text}"
    query_template: "Instruct: Given a question, retrieve relevant blog passages\nQuery: {text}"
```

- Asimetrik modellerde (embeddinggemma, e5) prefix kullanmamak sessizce kalite kaybına yol açar. Bu yüzden prefix'i tamamen kaldırmak yerine config'e taşıyoruz. Prefix istemeyen bir model (bge-m3) seçildiğinde şablon sadece `{text}` olur.
- Her model farklı bir vektör uzayı üretir. Bu yüzden collection/index adı profile göre belirlenecek (`blog_posts__embeddinggemma`, `blog_posts__bge-m3`). Embedding cache anahtarına da profil bilgisi eklenecek. Böylece modeller yan yana indexlenip eval ile kıyaslanabilecek.

### Kapsam dışı içerik
- **Foto İpucu** kategorisindeki 162 yazı indexe alınmayacak (hepsinin dosya adında `tek-foto` geçiyor, 58'inde ayrıca `tek-fotoluk-ipucu` tag'i var). Filtre `categories` alanı üzerinden yapılacak ve `config.exclude_categories` listesinde tutulacak.
- Bu yazılar çıkarılınca indexlenecek yazı sayısı **868** oluyor.

---

## 1. Proje Yapısı

```
blog-rag/
├── PLAN.md
├── AGENTS.md                 # AI ajanları için kurallar (CLAUDE.md bunu import eder)
├── CLAUDE.md
├── .claude/settings.json     # Claude Code izinleri (blog reposuna yazma yasak)
├── docker-compose.yml        # qdrant + neo4j (profile'lar ile ayrı ayrı açılabilir)
├── .env.example              # LMSTUDIO_URL, LLM modeli, DB adresleri, POSTS_DIR
├── embedding_profiles.yaml   # embedding modeli + doc/query şablonları
├── pyproject.toml
├── common/
│   ├── config.py             # pydantic-settings
│   ├── loader.py             # _posts okuma + front matter parse (python-frontmatter)
│   ├── chunker.py            # markdown-aware chunking
│   ├── embeddings.py         # profil tabanlı embed_documents / embed_query, batch + cache
│   ├── llm.py                # chat() sarmalayıcısı (OpenAI uyumlu)
│   └── prompts.py            # sistem prompt'u, context şablonu
├── phase1_vector/
│   ├── ingest.py             # load → chunk → embed → Qdrant upsert
│   ├── retrieve.py           # sorgu embed → search (+ filtre, + rerank opsiyonel)
│   └── ask.py                # CLI: soru → context → LLM → cevap + kaynaklar
├── phase2_graph/
│   ├── build_graph.py        # deterministik graph (front matter, linkler, kod dilleri)
│   ├── extract_entities.py   # LLM ile kavram çıkarımı (opsiyonel, kapsamı sınırlı)
│   ├── retrieve.py           # vektör giriş noktası + graph genişletme
│   └── ask.py
├── eval/
│   ├── questions.yaml        # altın soru seti (beklenen yazı slug'ları ile)
│   └── run_eval.py           # Faz 1 ve Faz 2 karşılaştırması
└── data/                     # chunks.jsonl, embedding cache (git'e girmez)
```

---

## 2. Faz 1 — Vektör Tabanlı RAG (Qdrant)

### 1.1 Altyapı
- `docker-compose.yml`: `qdrant/qdrant` image'ı. Portlar 6333 (REST + dashboard) ve 6334 (gRPC). Volume adı `qdrant_storage`.
- LM Studio host üzerinde çalışacak. Python scriptleri de host'ta çalışacak. Container içinden LM Studio'ya erişmek gerekmeyecek.
- Smoke test: `/v1/models` listesi alınacak, ardından tek bir embedding isteği atılıp vektörün 768 boyutlu olduğu doğrulanacak.

### 1.2 Yükleme ve Temizleme (`loader.py`)
- `exclude_categories` listesindeki yazılar (Foto İpucu) bu adımda atlanacak.
- Front matter'dan şunlar okunacak: `title`, `date`, `tags[]`, `categories[]`. Slug dosya adından türetilecek.
- Gövde temizlenecek: Jekyll/Liquid tag'leri (`{% ... %}`), HTML kalıntıları ve görsel satırları atılacak. Görselin alt text'i korunacak.
- Yazılar arası iç linkler Faz 2 için ayrıca kaydedilecek.

### 1.3 Chunking (`chunker.py`) — kalitenin büyük kısmı bu adıma bağlı
- Önce markdown başlıklarına (`##`, `###`) göre bölümlenecek. Her chunk'a `section_path` metadata'sı eklenecek (ör. `Rust > Değişkenler > Shadowing`).
- Bölüm ~500 token'ı aşarsa paragraf sınırlarından bölünecek. Ardışık chunk'lar arasında ~15% overlap olacak.
- **Kod blokları asla ortadan bölünmeyecek.** Bir kod bloğu çok büyükse tek başına chunk olacak, önündeki açıklama paragrafı da yanına eklenecek.
- Her chunk için şu metadata tutulacak: `post_slug, title, date, year, tags, categories, section_path, chunk_index, has_code, code_langs[]`.
- Çıktı `data/chunks.jsonl` dosyasına yazılacak. Gerçekleşen (Adım 2): 868 yazıdan **10.463 chunk** çıktı (p50 ~360, p90 ~535, max 1.500 tahmini token). 4.620 chunk kod içeriyor, 165 chunk dev kod bloğu bölünmesiyle oluştu. Büyük markdown tabloları satır sınırından bölünüyor, başlık satırları her parçada tekrarlanıyor.

### 1.4 Embedding ve Indexleme (`ingest.py`)
- Embedding istekleri 32–64'lük batch'lerle gönderilecek. Cache anahtarı `sha256(profil + şablonlanmış text)` olacak. Böylece tekrar çalıştırmada sadece değişen chunk'lar embed edilecek.
- Qdrant collection adı `blog_posts__{profil}`. Vektör boyutu profilden okunacak, mesafe `Cosine`.
- Payload index'leri: `tags`, `categories`, `year`, `code_langs`, `has_code`.
- Point ID, `slug + chunk_index` değerinden üretilen deterministik bir UUID olacak. Böylece aynı chunk ikinci kez eklenmez (idempotent upsert).

### 1.5 Retrieval (`retrieve.py`)
1. Sorgu `embed_query()` ile embed edilecek (aktif profilin şablonu kullanılır).
2. Qdrant'tan top-k (k=20) aday alınacak.
3. **Hafif filtre çıkarımı:** sorguda geçen dil/teknoloji adları tag sözlüğüyle eşleştirilecek (ör. "Rust" → `tags: rust` veya `code_langs: rust`). Bu filtre önce *should* (boost) olarak denenecek. Sonuç azsa filtresiz aramaya dönülecek.
4. Aynı yazıdan gelen çok fazla chunk olmaması için post başına en fazla 3 chunk alınacak, ardından MMR çeşitlendirmesi yapılacak.
5. Son olarak top-n (n=5–6) chunk context'e verilecek. Context bütçesi Llama 3'ün 8K penceresine göre ~4K token.
6. *(Opsiyonel 1.5b)* Cross-encoder reranker eklenebilir (`bge-reranker-v2-m3`, çok dilli, lokal). Kalite farkı eval ile ölçülecek.

### 1.6 Üretim (`ask.py`)
- Sistem prompt'u: "Yalnızca verilen bağlama dayan, Türkçe cevap ver, kod örneklerini bağlamdan uyarla, sonunda kaynak yazıları listele. Bağlam yetersizse bunu açıkça söyle."
- Her context chunk'ı `[#n] {title} ({date}) — {section_path}` başlığıyla verilecek. Cevaptaki atıflar `[#n]` formatında olacak.
- CLI kullanımı: `uv run python -m phase1_vector.ask "Rust dilinde değişken kullanımları ile ilgili örnekler yaz"` → çıktıda cevap, kullanılan kaynaklar (başlık + blog URL'i) ve `--debug` ile retrieval skorları görünecek.
- Model `--model` parametresi ile değiştirilebilecek (Llama 3 8B, Qwen3-8B, Gemma-3-12B kıyası için).

### 1.7 Faz 1 Bitti Kriterleri
- [ ] `docker compose up qdrant` ile Qdrant ayağa kalkıyor, ingest bütün yazıları hatasız indexliyor.
- [ ] Örnek Rust sorusunda ilk 5 sonucun çoğu Rust yazılarından geliyor ve cevapta `let`, `mut`, shadowing, `const` örnekleri bulunuyor.
- [ ] Eval setinde Recall@5 ve MRR ölçülmüş, sonuçlar kaydedilmiş.

---

## 3. Faz 2 — Graph Tabanlı RAG (Neo4j)

### 2.1 Altyapı
- `docker-compose.yml` içine `neo4j:5-community` eklenecek. APOC plugin'i açık olacak. Portlar 7474 (Browser) ve 7687 (Bolt). Kimlik bilgileri `.env`'den okunacak.
- Python tarafında `neo4j` driver kullanılacak. Hazır GraphRAG retriever'ları için opsiyonel olarak `neo4j-graphrag` paketi denenebilir.

### 2.2 Graph Modeli

```
(:Post {slug, title, date, year, url})
(:Chunk {id, text, section_path, chunk_index, embedding})
(:Tag {name})   (:Category {name})   (:Language {name})   (:Concept {name})

(Post)-[:HAS_CHUNK]->(Chunk)
(Chunk)-[:NEXT]->(Chunk)                 // yazı içindeki sıra
(Post)-[:TAGGED]->(Tag)
(Post)-[:IN_CATEGORY]->(Category)
(Chunk)-[:HAS_CODE_IN]->(Language)       // kod bloğu dilleri
(Post)-[:LINKS_TO]->(Post)               // yazılar arası iç linkler
(Chunk)-[:MENTIONS]->(Concept)           // LLM çıkarımı (2.4)
(Concept)-[:RELATED_TO]->(Concept)
```

### 2.3 Deterministik Graph (`build_graph.py`) — LLM kullanılmayacak
- Faz 1'deki `chunks.jsonl` ve embedding cache'i tekrar kullanılacak. Hiçbir şey yeniden embed edilmeyecek.
- Neo4j vector index tanımı: `CREATE VECTOR INDEX chunk_emb FOR (c:Chunk) ON c.embedding` (768 boyut, cosine).
- Tag normalizasyonu yapılacak (`csharp`/`c#`, `.net`/`dotnet` gibi). Gerekirse `(:Tag)-[:ALIAS_OF]->(:Tag)` ilişkisi kurulacak.
- Bu adım tek başına güçlü bir graph çıkarıyor çünkü front matter zaten kaliteli.

### 2.4 LLM ile Kavram Çıkarımı (`extract_entities.py`) — kapsamı sınırlı tutulacak
- 8B bir modelle ~10,5 bin chunk işlemek yavaş olur, sonuçlar da gürültülü çıkar. Bu yüzden önce bir pilot yapılacak: yalnızca `rust` + 2–3 tag alt kümesi (~100–150 yazı).
- JSON şemalı prompt kullanılacak: `{concepts: [{name, type}], relations: [{source, target, type}]}`. Bunun için LM Studio'nun structured output (`response_format: json_schema`) desteği kullanılacak.
- Kavram isimleri normalize edilecek (küçük harf, TR/EN eşleştirmesi, örn. "değişken gölgeleme" ↔ "shadowing"). Bunun için embedding benzerliğiyle birleştirme yapılacak.
- Sonuçlar yeterince iyiyse kapsam genişletilecek. Değilse sadece deterministik graph ile devam edilecek.

### 2.5 Graph Retrieval (`retrieve.py`)
1. **Giriş noktası:** Neo4j vector index ile top-k chunk bulunacak (Faz 1 ile aynı mantık).
2. **Genişletme (Cypher):**
   - Bulunan chunk'ların `NEXT` komşuları alınacak (bağlam bütünlüğü için).
   - Aynı `Tag`/`Language`/`Concept`'e bağlı diğer yazılardan en yüksek skorlu chunk'lar eklenecek.
   - `LINKS_TO` ile bağlantılı yazılar eklenecek (ör. bir serinin önceki bölümleri).
3. **Yapısal sorular** için graph doğrudan sorgulanacak. Örnekler: "Rust ile ilgili kaç yazı var, hangi yıllarda?", "Tokio geçen yazılar hangileri?" Bunlar için iki yol denenecek: (a) hazır Cypher şablonları, (b) opsiyonel text2cypher (8B modelle güvenilirliği test edilecek).
4. Skorlama: `vektör skoru + graph bonusu` (paylaşılan tag/kavram sayısı, link mesafesi). Sonra Faz 1'deki ile aynı top-n seçimi yapılacak.
5. LLM'e giden context'e ilişki bilgisi de eklenecek. Örnek: "Bu chunk *Rust Değişkenler* yazısının devamı", "Bu yazı X'e link veriyor".

### 2.6 Faz 2 Bitti Kriterleri
- [ ] Graph Neo4j Browser'da gezilebiliyor (Post–Tag–Chunk ilişkileri görünüyor).
- [ ] Örnek Rust sorusunda Faz 1'e göre daha bütünlüklü bir cevap çıkıyor: komşu chunk'lar, seri bütünlüğü sağlanıyor.
- [ ] Yapısal sorular ("kaç yazı", "hangi yazılar") Faz 1'in cevaplayamadığı şekilde doğru cevaplanıyor.
- [ ] Eval karşılaştırma tablosu hazır: Faz 1 ve Faz 2 için Recall@5, MRR, cevap kalitesi ve gecikme süresi.

---

## 4. Değerlendirme (her iki faz için ortak)
- `eval/questions.yaml` dosyasında ~25–30 soru olacak. Her soru için beklenen yazı slug'ları yazılacak. Üç soru tipi bulunacak:
  - **Olgusal/örnek:** "Rust'ta değişken tanımlama örnekleri", "WCF'te binding türleri"
  - **Çok yazılı/ilişkisel:** "Rust ownership ve borrowing'i birlikte anlatan yazılar"
  - **Yapısal:** "2009'da en çok hangi konuda yazılmış?" (Faz 2 bu tipte avantajlı olmalı)
- Retrieval metrikleri: Recall@k, MRR. Cevap kalitesi elle 1–5 puan ile değerlendirilecek, istenirse büyük bir modelle LLM-as-judge da yapılabilir.
- Bütün sonuçlar `eval/results/{faz}-{model}-{tarih}.json` olarak kaydedilecek.

---

## 5. Adım Sırası (iş listesi)

**Faz 1**
1. ✅ İskelet: `pyproject.toml`, `.env.example`, `docker-compose.yml` (qdrant), LM Studio smoke test
2. ✅ `loader.py` + `chunker.py` → `data/chunks.jsonl` + chunk istatistikleri (dağılım, en uzun chunk'lar)
3. `lmstudio.py` + `ingest.py` → Qdrant'ın dolması
4. `retrieve.py` → örnek soru ile retrieval debug edilecek (henüz LLM yok)
5. `ask.py` → uçtan uca cevap
6. Eval seti + ilk ölçüm. Opsiyonel olarak reranker, LLM kıyası ve embedding profili kıyası (embeddinggemma / bge-m3).

**Faz 2**
7. Neo4j compose'a eklenecek + `build_graph.py` (deterministik graph + vector index)
8. Graph retrieval + `ask.py`
9. Pilot entity extraction (Rust alt kümesi)
10. Faz 1 ve Faz 2 karşılaştırmalı eval + bulguların yazılması

---

## 6. Açık Konular / Riskler
- **Llama 3 8B Türkçe kalitesi:** model kıyası 1.6'da yapılacak. Context penceresi 8K olduğu için context bütçesi sıkı tutulmalı.
- **Eski yazılar (2003–2012):** ağırlıklı olarak .NET/WCF içeriği var. Zaman filtresi veya tarih ağırlığı gerekebilir, eval sonuçlarına göre karar verilecek.
- **Embedding süresi:** ~10,5 bin chunk işlenecek. Lokal GPU'da birkaç dakika ile onlarca dakika arasında sürebilir. Cache sayesinde bu maliyet bir kez ödenecek.
