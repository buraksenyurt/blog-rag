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
| LLM | LM Studio → `meta-llama-3-8b-instruct` ile başlanacak. Model config'ten değişebilir olacak. Donanım sınırı nedeniyle küçük modellerle kıyaslanacak: `gemma-3-4b-it-qat` (Q4_0) ve `qwen3-4b-2507` (Instruct, Q4_K_M) |
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
- Çıktı `data/chunks.jsonl` dosyasına yazılacak. Gerçekleşen: 868 yazıdan **10.515 chunk** (p50 ~360, p90 ~540 tahmini token). Kırpılma testi kodlu Türkçe metinde oranın ~2,4 karakter/token'a kadar düştüğünü gösterdi. 1.500 tokenlik üst sınırda (~4.800 karakter) en büyük 30 chunk'tan 5'i kırpılıyordu. Bu yüzden üst sınır **1.250 tahmini token (~4.000 karakter)** yapıldı. Bu değişiklikte sadece 382 chunk yeniden embed edildi. Büyük markdown tabloları satır sınırından bölünüyor, başlık satırları her parçada tekrarlanıyor.

### 1.4 Embedding ve Indexleme (`ingest.py`)
- Embedding istekleri 32–64'lük batch'lerle gönderilecek. Cache anahtarı `sha256(profil + şablonlanmış text)` olacak. Böylece tekrar çalıştırmada sadece değişen chunk'lar embed edilecek.
- Qdrant collection adı `blog_posts__{profil}`. Vektör boyutu profilden okunacak, mesafe `Cosine`.
- Payload index'leri: `tags`, `categories`, `year`, `code_langs`, `has_code`.
- Point ID, `slug + chunk_index` değerinden üretilen deterministik bir UUID olacak. Böylece aynı chunk ikinci kez eklenmez (idempotent upsert).
- Uygulama (Adım 3): LM Studio embedding yanıtında `usage` döndürmüyor. Bu yüzden token kontrolü kırpılma testiyle yapılıyor: metnin sonuna uzun bir ek yapılıp vektör değişiyor mu diye bakılıyor (`ingest --check-only`). İlk tam ingest 10.463 chunk için ~65 dakika sürdü.
- Uygulama (Adım 3): cache `data/embeddings/{profil}.sqlite` dosyasında (float32). Embedding girdisinin başlığı `{yazı başlığı} > {section_path}` şeklinde. Ingest 512'lik pencerelerle ilerliyor. Artık chunks.jsonl'da olmayan eski kayıtlar collection'dan siliniyor. Ingest'ten önce en büyük 10 chunk'ın gerçek token sayısı ölçülüyor ve model sınırını aşanlar raporlanıyor.

### 1.5 Retrieval (`retrieve.py`)
1. Sorgu `embed_query()` ile embed edilecek (aktif profilin şablonu kullanılır).
2. Qdrant'tan top-k (k=20) aday alınacak.
3. **Hafif filtre çıkarımı:** sorguda geçen dil/teknoloji adları tag sözlüğüyle eşleştirilecek (ör. "Rust" → `tags: rust` veya `code_langs: rust`). Bu filtre önce *should* (boost) olarak denenecek. Sonuç azsa filtresiz aramaya dönülecek.
4. Aynı yazıdan gelen çok fazla chunk olmaması için post başına en fazla 3 chunk alınacak, ardından MMR çeşitlendirmesi yapılacak.
5. Son olarak top-n (n=5–6) chunk context'e verilecek. Context bütçesi Llama 3'ün 8K penceresine göre ~4K token.
6. Uygulama (Adım 4): Filtre çıkarımı serbest tag eşleştirmesiyle değil, küratörlü bir teknoloji sözlüğüyle yapılıyor (`phase1_vector/query_filters.py`), çünkü blogda `as`, `is`, `get`, `list` gibi tag'ler var. Filtreli ve filtresiz iki arama yapılıp sonuçlar birleştiriliyor, filtreye uyan adaylar +0.05 boost alıyor. Böylece filtre az sonuç verirse filtresiz sonuçlar zaten listede oluyor. Seçim MMR ile yapılıyor (λ=0.7, post başına en fazla 3 chunk, ~3.500 token bütçe). İlk denemede seçilen 6 chunk'tan 2'si zayıftı: biri yazının kapanış paragrafıydı, diğeri koda giriş cümlesiydi ve asıl kod sonraki chunk'taydı. Bunun üzerine iki ekleme yapıldı: (a) sorgu kod örneği istiyorsa (örnek/kod/göster/yaz) kod içeren adaylara +0.02 boost, (b) 150 tokendan kısa ve kodsuz seçilen chunk'lara aynı yazıdaki sonraki chunk ekleniyor. İkisi de kapatılabilir (`--no-code-boost`, `--no-expand`), etkileri eval'de ölçülecek.
7. *(Opsiyonel 1.5b)* Cross-encoder reranker eklenebilir (`bge-reranker-v2-m3`, çok dilli, lokal). Kalite farkı eval ile ölçülecek.

### 1.6 Üretim (`ask.py`)
- Sistem prompt'u: "Yalnızca verilen bağlama dayan, Türkçe cevap ver, kod örneklerini bağlamdan uyarla, sonunda kaynak yazıları listele. Bağlam yetersizse bunu açıkça söyle."
- Her context chunk'ı `[#n] {title} ({date}) — {section_path}` başlığıyla verilecek. Cevaptaki atıflar `[#n]` formatında olacak.
- CLI kullanımı: `uv run python -m phase1_vector.ask "Rust dilinde değişken kullanımları ile ilgili örnekler yaz"` → çıktıda cevap, kullanılan kaynaklar (başlık + blog URL'i) ve `--debug` ile retrieval skorları görünecek.
- Model `--model` parametresi ile değiştirilebilecek (Llama 3 8B, Gemma 3 4B QAT, Qwen3 4B Instruct 2507 kıyası için).

- Uygulama (Adım 5): Prompt'lar `common/prompts.py` içinde. Cevap konsola canlı akıyor, kaynak listesinde atıf yapılan parçalar ✓ ile işaretleniyor. Atıf yoksa ya da geçersiz numara varsa uyarı veriliyor. Her soru `data/runs/ask-{tarih}.jsonl` dosyasına kaydediliyor (eval için). `--no-rag` bağlamsız cevap veriyor, kıyas için kullanılıyor.

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
- Uygulama (Adım 7): Graph `neo4j:5.26-community` (LTS) ile kuruluyor. Vektörler Faz 1'in embedding cache'inden okunuyor: 10.515 chunk'ın hepsi cache'te, yeniden embed gerekmiyor, LM Studio'ya da ihtiyaç yok.
  - Tag adları küçük harfe normalize ediliyor. Farklı yazılışları `Tag.variants` alanında tutuluyor (`PostgreSQL`/`postgresql` gibi).
  - Plana ek olarak `(:Series)` düğümü var. Aynı başlık önekini paylaşan en az 3 yazı tarih sırasıyla bir seri sayılıyor ve sıra `IN_SERIES.position` ilişkisinde tutuluyor. 226 yazı bir seriye bağlandı. Örnekler: Ruby Kod Parçacıkları (35), GoLang (20), Rust Pratikleri (15). Bazı "seriler" aslında konu öneki (`WCF`, `GoLang`). Ad bu yüzden bilerek geniş tutuldu.
  - Blog içi linklerden 114 `LINKS_TO` ilişkisi kuruldu. Kapsam dışı yazılara giden 3 link atlandı.
  - Kurulum idempotent çalışıyor: her şey MERGE ile yazılıyor, yeniden kurulumda eski ilişkiler temizleniyor, chunks.jsonl'da olmayan düğümler siliniyor. Graph'ın hangi embedding profiliyle kurulduğu `(:GraphMeta)` düğümünde tutuluyor. Farklı bir profille kurmak `--recreate` gerektiriyor.
  - Kurulum sonunda doğrulama sorguları çalışıyor: Rust yazı sayısı, Go yazılarının yıllara göre dağılımı, Rust Pratikleri serisi ve vector index'in kendi vektörüyle sınanması.

### 2.4 LLM ile Kavram Çıkarımı (`extract_entities.py`) — kapsamı sınırlı tutulacak
- 8B bir modelle ~10,5 bin chunk işlemek yavaş olur, sonuçlar da gürültülü çıkar. Bu yüzden önce bir pilot yapılacak: yalnızca `rust` + 2–3 tag alt kümesi (~100–150 yazı).
- JSON şemalı prompt kullanılacak: `{concepts: [{name, type}], relations: [{source, target, type}]}`. Bunun için LM Studio'nun structured output (`response_format: json_schema`) desteği kullanılacak.
- Kavram isimleri normalize edilecek (küçük harf, TR/EN eşleştirmesi, örn. "değişken gölgeleme" ↔ "shadowing"). Bunun için embedding benzerliğiyle birleştirme yapılacak.
- Sonuçlar yeterince iyiyse kapsam genişletilecek. Değilse sadece deterministik graph ile devam edilecek.

- Uygulama (Adım 9): `phase2_graph/concepts.py` ve `extract_entities.py`.
  - Birim yazı başına bir LLM çağrısı. Girdi başlık, tag'ler, bölüm başlıkları ve ~12.000 karakterlik özet (kod blokları ilk 8 satırla sınırlı). Chunk başına çağrı bu makinede saatler sürerdi, bu yüzden yazı seviyesinde çalışılıyor.
  - Çıktı JSON şemasıyla zorlanıyor (LM Studio structured output). Kavram adları küçük harfli İngilizce terimler, Türkçe karşılıkları `aliases` alanında. En fazla 15 kavram çıkarılıyor. Tür ve ilişki çeşidi sabit listelerden seçiliyor.
  - Sonuçlar `data/concepts/{model}/` altında yazı başına cache'leniyor. Yarıda kalan çalıştırma kaldığı yerden devam ediyor.
  - Birleştirme sadece metin üzerinden yapılıyor: ad ile takma adın eşleşmesi ve tekil/çoğul farkı. embeddinggemma skorları dar aralıkta olduğu için embedding benzerliğiyle otomatik birleştirme yapılmıyor (yanlış birleştirme riski).
  - Graph'a eklenenler: `(:Concept {name, type, aliases, df})`, `(Post)-[:ABOUT]->(Concept)`, `(Chunk)-[:MENTIONS]->(Concept)` ve `(Concept)-[:RELATED_TO {kind, count}]->(Concept)`. `MENTIONS` kavram adı ya da takma adının pilot yazıların chunk metninde kelime sınırıyla geçmesine göre kuruluyor, LLM kullanılmıyor.
  - Retrieval'da `concept_expand` seçeneği var: sorguda adı ya da takma adı geçen ve en fazla 8 yazıda görülen kavramların geçtiği chunk'lar aday havuzuna +0.03 ile ekleniyor. `rust` gibi çok yazıda geçen merkez kavramlar kullanılmıyor. Eval'deki karşılığı `concepts` varyantı.
  - Dry-run (3 yazı, gemma-3-4b-it-qat, yazı başına ~2 dakika): Kavramlar isabetli çıktı (ownership, borrowing, lifetime, borrow-checker, tokio, warp, serde…). Takma adla birleştirme çalıştı (lifetimes → lifetime, borrow checker → borrow-checker). Görülen sorunlar ve düzeltmeler:
    - (1) Modelin önerdiği takma adların yarısına yakını yazının metninde geçmiyordu ("possession", "memory allocation", "compiler check"). Takma adlar artık kavramda tutuluyor ama MENTIONS sadece kanıtlı terimlerle (`mention_terms`) kuruluyor. Kanıtlı terim, kavramın adı ya da önerildiği yazının metninde gerçekten geçen takma ad demek.
    - (2) `shadowing` uzun yazının 11.400. karakterindeydi, 7.000 karakterlik bütçeye girmiyordu. Bütçe 12.000 karaktere (~4,5 bin token), kavram sınırı 15'e çıkarıldı.
    - (3) Bir kavram Türkçe adla döndü ("asenkron programlama"). Prompt'a İngilizce ad kuralı açıkça eklendi.
    - (4) Eşleştirme deseninde kelime sınırı sadece ilk ve son alternatife uygulanıyordu, gruplanarak düzeltildi.
    - Prompt değiştiği için cache `v2` klasörüne yazılıyor.
  - Dry-run v2 (3 yazı): Uzun Rust yazısının 15 kavramı neredeyse birebir tag listesinden geldi, `shadowing` yine çıkmadı. Kanıtlı takma adların bir kısmı da yanlış eşleşiyordu ("kopyalama" → borrowing, "package" → crate, "reference counting" iki ayrı kavramda). Düzeltmeler (v3):
    - (5) Tag'ler özete verilmiyor. Tag'ler graph'ta zaten var, kavram katmanı metinden yeni bilgi çıkarmalı.
    - (6) MENTIONS yalnızca kavram adlarıyla kuruluyor: kanonik ad ve onunla birleşen adlar, metinde geçiyorsa. Takma adlar sadece sorgudaki Türkçe terimi kavrama bağlamak için kullanılıyor. Yazar İngilizce terimleri çoğunlukla parantez içinde verdiği için ("Sahiplenme (Ownership)") ad bazlı eşleştirme yeterli.
    - (7) Seçimin konusu olan tag'lerle aynı adlı kavramlar (`rust`) atılıyor. Aksi halde 119 chunk'a bağlanan bir merkez düğüm oluşuyordu.
  - Dry-run v3 (3 yazı): Kavramlar artık tag kopyası değil (async-std, message passing, higher order function, heap/stack memory, webassembly). `shadowing` yine seçilmedi: model 15 kavramlık sınırda daha genel kavramları öne alıyor. Yeni bulgular:
    - (8) Model kod ifadelerini kavram sandı (`println!` 70 chunk'ta, `main` 66 chunk'ta, `String::from`). `::`, `!`, parantez, `<>` içeren adlar ve `main` gibi genel adlar atılıyor. Bu filtre kavram sayısını 31'den 27'ye, MENTIONS'ı 395'ten 227'ye indirdi. Cache bozulmadığı için yeniden çıkarım gerekmedi.
    - (9) Eval'deki Go sorusu "channel" kelimesiyle Rust'ın channel kavramına eşleniyordu. Sorguda teknoloji varsa kavram adayları artık sadece o teknolojinin tag'ine sahip yazılardan geliyor.
  - Tam çalıştırma v3 (46 yazı, gemma-3-4b-it-qat):
    - 45 yazı başarılı. `tauri-hello-world` yazısında çıktı max_tokens'ta kesildi ve JSON yarım kaldı.
    - Asıl sorun prompt'tan sızma: en çok yazıda görünen 8 kavram (ownership 36, lifetime 31, smart pointer 30, borrowing 29, trait object 28, channel 26, tokio 23, shadowing 18) prompt'taki örnek listesinin birebir aynısıydı. Kavram atamalarının %38'i (626'da 239) yazının metninde ne adıyla ne Türkçe karşılığıyla geçiyordu. `shadowing`'in 18 atamasının 18'i de kanıtsızdı.
    - `var` kavramı Türkçe "var" kelimesiyle eşleşip 221 chunk'a bağlanmıştı.
    - Düzeltmeler (v4):
      - (10) Kanıt kontrolü: kavram, adı ya da bir karşılığı yazının metninde geçmiyorsa o yazıya bağlanmıyor. Kontrol deterministik olduğu için v3 cache'ine de uygulanabiliyor. Sonuç: 224 kavram, 367 ABOUT, 3.208 MENTIONS. En yaygın kavramlar artık ownership (20), borrowing (11), smart pointer (11), lifetime (9), cargo (8).
      - (11) Prompt'taki Rust örnekleri kaldırıldı. Yerine alan dışı bir biçim örneği kondu ("çöp toplayıcı" → garbage collection). Ayrıca "yazıda geçmeyen kavram ekleme" kuralı eklendi.
      - (12) Türkçe yaygın kelimelerle çakışan adlar (var, yok, bir, ile…) ve `&`/`*` içeren kod ifadeleri atılıyor.
      - (13) JSON şemasına üst sınırlar eklendi (15 kavram, 4 takma ad, 20 ilişki). max_tokens 2.048'e çıkarıldı. JSON yarım kalırsa 3.072 ile bir kez daha deneniyor.
    - v3 cache'i `--cache-version 3` ile kullanılabiliyor. Temiz bir pilot için v4 ile yeniden çıkarım öneriliyor (~1,5–2 saat).
  - Bilinen sınırlar: Özete tag'ler de girdiği için model kavramların bir kısmını tag'lerden tekrar edebilir. Çok uzun yazılarda 12.000 karakter sınırı sonraki bölümleri dışarıda bırakıyor (bölüm başlıkları yine de listede).
- Giriş araması (2026-11-25 kararı): Faz 2'de varsayılan olarak tam cosine kullanılıyor (`exact_search=True`). Ölçüm (20 Kasım): `exact` varyantı Faz 1 ile birebir aynı sonucu verdi (Hit@ctx 0,956, zor sorularda 0,846). Quantization kapatılmış HNSW index'i ise hâlâ geride kaldı (0,911 / 0,769). 10 bin chunk için tam cosine yeterince hızlı. Böylece iki faz arasındaki fark yalnızca graph katmanından geliyor. Index karşılaştırma için `vector` varyantında tutuluyor.

### 2.5 Graph Retrieval (`retrieve.py`)
1. **Giriş noktası:** Neo4j vector index ile top-k chunk bulunacak (Faz 1 ile aynı mantık).
2. **Genişletme (Cypher):**
   - Bulunan chunk'ların `NEXT` komşuları alınacak (bağlam bütünlüğü için).
   - Aynı `Tag`/`Language`/`Concept`'e bağlı diğer yazılardan en yüksek skorlu chunk'lar eklenecek.
   - `LINKS_TO` ile bağlantılı yazılar eklenecek (ör. bir serinin önceki bölümleri).
3. **Yapısal sorular** için graph doğrudan sorgulanacak. Örnekler: "Rust ile ilgili kaç yazı var, hangi yıllarda?", "Tokio geçen yazılar hangileri?" Bunlar için iki yol denenecek: (a) hazır Cypher şablonları, (b) opsiyonel text2cypher (8B modelle güvenilirliği test edilecek).
4. Skorlama: `vektör skoru + graph bonusu` (paylaşılan tag/kavram sayısı, link mesafesi). Sonra Faz 1'deki ile aynı top-n seçimi yapılacak.
5. LLM'e giden context'e ilişki bilgisi de eklenecek. Örnek: "Bu chunk *Rust Değişkenler* yazısının devamı", "Bu yazı X'e link veriyor".

- Uygulama (Adım 8): `phase2_graph/retrieve.py`, `structural.py`, `ask.py`.
  - Adaylar Faz 1'in karşılığı olacak şekilde üretiliyor. Vector index'ten top-k alınıyor. Teknoloji filtresi varsa ilgili tag ve kod dili altındaki chunk'lar arasında tam cosine araması yapılıyor (`vector.similarity.cosine`).
  - Neo4j cosine skorunu (1+cos)/2 aralığında döndürüyor. Skorlar cos'a çevriliyor, böylece Faz 1 ile aynı ölçekte kalıyor ve aynı boost'lar geçerli oluyor.
  - Graph genişletmesi: en iyi 3 aday yazının `LINKS_TO` ve seri komşularından en alakalı chunk'lar aday havuzuna ekleniyor. En fazla 4 yazı ekleniyor ve her biri +0.02 alıyor. Bu adayların graph'tan gelme nedeni tabloda görünüyor.
  - Kısa chunk genişletmesi `NEXT` ilişkisiyle yapılıyor. Seriye ait chunk'ların bağlam başlığına "X serisi, n. yazı" notu ekleniyor.
  - Yapısal sorular kural tabanlı tespit ediliyor: sayma ("kaç yazı"), yıllara göre dağılım, listeleme ve seri (sorunun "seri" kelimesini içermesi gerekiyor). Cevap Cypher şablonlarıyla hesaplanıyor ve LLM'e "GRAPH BİLGİSİ" bloğu olarak veriliyor. Sistem prompt'una bu sayıların kesin olduğunu söyleyen bir kural eklendi.
  - Ortak kod `common/` altına taşındı: `retrieval.py` (aday yapısı, boost'lar, MMR, kısa chunk genişletmesi), `answering.py` (soru-cevap akışı) ve `query_filters.py`. Böylece iki faz arasındaki fark yalnızca aday üretimi, graph genişletmesi ve yapısal sorgulardan geliyor.
  - `eval.run_eval --phase 2` dört varyantı kıyaslıyor: `vector`, `exact` (index'siz tam cosine), `graph` (genişletme) ve `full` (genişletme + yapısal). Yapısal sorular için beklenen cevaptaki bütün sayıların graph bilgisinde geçip geçmediği kontrol ediliyor.
  - İlk Faz 2 ölçümü (2026-11-20, 48 soru):
    - `full` üç yapısal sorunun üçünü de doğru cevapladı. Seri sorusunda recall 0,13'ten 1,00'e çıktı.
    - Graph genişletmesi zor sorularda Recall@ctx'i 0,513'ten 0,551'e, Recall@20'yi 0,596'dan 0,635'e çıkardı. Normal sorularda etkisi küçük (+0,008).
    - Beklenmedik bulgu: Faz 2'nin `vector` varyantı aynı embedding'lerle Faz 1'den kötü çıktı (zor sorularda Hit@ctx 0,692'ye karşı 0,846, Recall@20 0,596'ya karşı 0,712).
    - Şüphe: Neo4j vector index'i varsayılan olarak quantization kullanıyor (kendi vektörüyle arama skoru 0,9988). embeddinggemma skorları dar bir aralıkta toplandığı için küçük sapmalar sıralamayı değiştiriyor olabilir.
    - Bunu sınamak için iki şey eklendi: index'siz `exact` varyantı ve index'i quantization kapalı yeniden kuran `build_graph --rebuild-index` seçeneği. Index artık varsayılan olarak quantization kapalı kuruluyor.

### 2.6 Faz 2 Bitti Kriterleri
- [ ] Graph Neo4j Browser'da gezilebiliyor (Post–Tag–Chunk ilişkileri görünüyor).
- [ ] Örnek Rust sorusunda Faz 1'e göre daha bütünlüklü bir cevap çıkıyor: komşu chunk'lar, seri bütünlüğü sağlanıyor.
- [ ] Yapısal sorular ("kaç yazı", "hangi yazılar") Faz 1'in cevaplayamadığı şekilde doğru cevaplanıyor.
- [ ] Eval karşılaştırma tablosu hazır: Faz 1 ve Faz 2 için Recall@5, MRR, cevap kalitesi ve gecikme süresi.

---

## 3b. Faz 3 — Ders Demo Arabirimi (Streamlit)

Amaç: Derste projeksiyonda soru girip aynı soruya **vektör RAG** (Faz 1) ve **graph RAG**
(Faz 2) ile cevap almak, ikisini yan yana karşılaştırmak ve RAG'in iç işleyişini
(getirilen chunk'lar, skorlar, graph bilgisi, LLM'e giden prompt) öğrencilere göstermek.

### 3b.1 Kararlar (2026-11-26)

| Konu | Karar | Gerekçe |
|---|---|---|
| Kullanım | Eğitmen makinesinde, projeksiyonda demo. Tek kullanıcı. | Bu makinede bir cevap ~70 sn (Gemma 3 4B) ve LM Studio aynı anda tek cevap üretiyor. Çok kullanıcılı erişim kuyruk ve uzun bekleme demek. |
| Teknoloji | Streamlit (Python) | Mevcut `common/` kodunu doğrudan kullanıyor. Stream, tablo ve genişletilebilir paneller hazır. Ayrı bir API katmanı gerekmiyor. |
| Mod | Seçmeli: Vektör / Graph. İsteğe bağlı "Karşılaştır": iki cevap sırayla üretilir, yan yana gösterilir. | Karşılaştırma iki kat sürüyor (~2,5 dk), bu yüzden varsayılan değil. |
| Görünürlük | Kaynak yazılar (linkli), getirilen chunk'lar ve skorlar, graph bilgisi, LLM'e giden tam prompt, süreler | RAG'in iç işleyişini derste göstermek. Prompt anatomisi görselindeki parçalar arayüzde görünür olmalı. |

Kapsam dışı (şimdilik): çok kullanıcılı erişim, kimlik doğrulama, internete açma.
Mimari bunları sonradan eklemeye izin verecek şekilde kurulacak: arayüz mantığı
`ui/service.py` içinde, Streamlit'ten bağımsız tutulacak.

### 3b.2 Yapı

```
ui/
├── app.py          # Streamlit sayfası (ince katman: düzen, widget'lar, gösterim)
├── service.py      # Streamlit'ten bağımsız: kaynak bağlantıları, servis durumu, soru çalıştırma
└── components.py   # tekrar kullanılan gösterimler: cevap kartı, kaynaklar, chunk tablosu, prompt
.streamlit/config.toml   # headless, telemetri kapalı, geniş düzen
```

### 3b.3 Ekran düzeni

- **Kenar çubuğu:**
  - Servis durumu: LM Studio, Qdrant ve Neo4j ayakta mı, embedding modeli ve LLM yüklü mü. Kapalı olan servisin modu devre dışı kalır ve sebebi yazılır.
  - LLM modeli: LM Studio'daki yüklü modellerden seçilir. LLM seçimi park edildiği için derste modeller karşılaştırılabilir.
  - Retrieval ayarları: chunk sayısı (`n`), token bütçesi, sıcaklık. Graph için genişletme, yapısal sorgular ve kavram katmanı (varsayılan kapalı).
  - Örnek sorular: eval setinden normal, zor ve yapısal birer örnek, tek tıkla soru kutusuna gelir.
- **Ana alan:**
  - Soru kutusu, mod seçimi (Vektör / Graph / Karşılaştır), "Sor" düğmesi.
  - Cevap: canlı akış (stream). Bitince `[#n]` atıflarının karşılığı olan kaynak yazılar linkleriyle listelenir (atıf yapılanlar işaretli).
  - Genişletilebilir paneller:
    1. Getirilen chunk'lar: sıra, final skor, ham skor, F/k/G boost işaretleri, graph'tan gelme nedeni, bölüm, LLM'e gidip gitmediği.
    2. Graph bilgisi: yapısal sorgu sonucu (varsa).
    3. LLM'e giden prompt: sistem prompt'u ve kullanıcı mesajı (bağlam + soru), aynen.
    4. Süreler: retrieval, ilk token, toplam üretim, bağlam token'ı.
- **Karşılaştır modu:**
  - İki sütun: önce vektör, sonra graph cevabı üretilir. Her sütun kendi panellerini gösterir.
  - Üstte retrieval farkı özetlenir: iki bağlamda ortak olan yazılar, yalnızca birinde olanlar, graph bilgisi var mı.
- **Soru geçmişi:** Oturum içindeki sorular ve cevaplar listelenir. Her soru `data/runs/` altına `phase: "ui-vector"` / `"ui-graph"` olarak kaydedilir.

### 3b.4 Demo kolaylıkları

- **Cevap cache'i (isteğe bağlı, varsayılan açık):** Aynı soru + mod + model + ayarlar için daha önce üretilmiş cevap `data/ui_cache/` altından anında gösterilir, "yeniden üret" düğmesiyle tazelenir. Ders öncesi örnek soruların cevapları hazırlanıp derste beklemeden gösterilebilir.
- **Ön ısıtma:** Uygulama açılışta embedding modelini ve veritabanı bağlantılarını hazırlar (`st.cache_resource`), ilk sorudaki gecikme azalır.
- Projeksiyon için geniş düzen ve okunaklı kod blokları.

### 3b.5 Gerekli küçük düzenlemeler

- `common/answering.answer_question` önceden yapılmış bir retrieval sonucunu kabul edecek. Arayüz retrieval'ı bir kez yapıp hem tabloyu gösterecek hem cevabı üretecek. Şu an `--debug` retrieval'ı iki kez çalıştırıyor.
- `AskResult` LLM'e giden mesajları (`messages`) da taşıyacak. Prompt paneli bunu gösterecek.
- Retrieval tablosu için `print_result` mantığı yapısal veriye (satır listesi) ayrılacak. Konsol ve arayüz aynı veriyi kullanacak.

### 3b.6 Faz 3 Bitti Kriterleri

- [ ] `uv run streamlit run ui/app.py` ile arayüz açılıyor. Servis durumu doğru görünüyor, kapalı servisin modu devre dışı kalıyor.
- [ ] Vektör ve graph modlarında cevap canlı akıyor. Kaynaklar, chunk tablosu, graph bilgisi, prompt ve süreler görünüyor.
- [ ] Karşılaştır modunda iki cevap yan yana, retrieval farkı özetli.
- [ ] Örnek sorular ve cevap cache'i çalışıyor. Cache'teki bir cevap anında geliyor.
- [ ] `ui/service.py` testli. Arayüz için Streamlit `AppTest` ile sahte servislerle bir duman testi var.
- [ ] README'de arayüz bölümü var.

---

## 4. Değerlendirme (her iki faz için ortak)
- `eval/questions.yaml` dosyasında ~25–30 soru olacak. Her soru için beklenen yazı slug'ları yazılacak. Üç soru tipi bulunacak:
  - **Olgusal/örnek:** "Rust'ta değişken tanımlama örnekleri", "WCF'te binding türleri"
  - **Çok yazılı/ilişkisel:** "Rust ownership ve borrowing'i birlikte anlatan yazılar"
  - **Yapısal:** "2009'da en çok hangi konuda yazılmış?" (Faz 2 bu tipte avantajlı olmalı)
- Retrieval metrikleri: Recall@k, MRR. Cevap kalitesi elle 1–5 puan ile değerlendirilecek, istenirse büyük bir modelle LLM-as-judge da yapılabilir.
- Bütün sonuçlar `eval/results/{faz}-{model}-{tarih}.json` olarak kaydedilecek.
- Uygulama (Adım 6): `eval/questions.yaml` dosyasında 35 soru var: 29 example, 3 relational, 3 structural. Beklenen yazılar korpusta aranarak elle eşlendi, tamamı doğrulandı. `eval/run_eval.py` retrieval varyantlarını (`plain`, `filter`, `filter+code`, `full`) aynı sorgu vektörüyle kıyaslıyor. Metrikler post seviyesinde: Hit@ctx, Recall@ctx, MRR@20, Recall@20. `--with-llm` cevapları da üretiyor ve atıf oranı, Türkçe oranı ve süreyi ölçüyor. Manuel 1-5 puanlama için ayrıca bir markdown dosyası yazıyor.
- İlk ölçüm (2026-11-13, embeddinggemma, 32 soru): `plain` Hit@ctx 0,969 · Recall@ctx 0,891 · MRR 0,892 · Recall@20 0,977. `filter` sonucu birebir aynı çıktı. Kod boost'u MRR'yi 0,888'e düşürdü. Soruların neredeyse hepsi teknoloji adını içerdiği için set tavana yakındı ve varyantları ayırt edemedi. Tek ıskalanan `parallel-for` sorusu bir etiketleme sorunuydu: beklenen listeye `forparallelism` ve `parallel-programming-reduction` yazıları eklendi.
- Küçük LLM kıyası (2026-11-20, Faz 1 retrieval, 5 soru, 4 chunk, 2.000 token bütçe):
  - Ortalama üretim süresi: Llama 3 8B 149 sn, `gemma-3-4b-it-qat` 68 sn. İkisinde de ilk token ~5 sn, yani süreyi bağlam işleme değil token üretimi belirliyor.
  - Atıf yapan cevap: Llama 1/5, Gemma 5/5. Türkçe oranı: Llama 1,00, Gemma 0,97.
  - İki model de Rust sorularında kavram hataları yaptı. Llama, shadowing sorusunda yine kendine sorular uydurup cevapladı ve kaynakları kurala aykırı biçimde listeledi. Gemma Go (defer/panic/recover) ve e-posta testi sorularında doğru ve kaynaklı cevap verdi.
  - Yanlışlıkla `qwen3-4b-thinking-2507` indirildi. Bu model bütün çıktısını düşünme bölümüne yazdığı için 5 cevabın hepsi boş döndü (ortalama 272 sn). Ölçümün `qwen3-4b-instruct-2507` ile tekrarlanması gerekiyor. Boş cevaplar artık eval'de hata olarak işaretleniyor.
- Set genişletildi (48 soru): teknoloji ya da kavram adı geçmeyen, günlük dille sorulmuş 13 **zor soru** (`difficulty: hard`) eklendi. Örnek: "Bir değişkeni fonksiyona verdikten sonra neden tekrar kullanamıyorum?" Eval özeti normal ve zor soruları ayrı raporluyor.
- LLM'li eval'de cevaplar stream ediliyor. Zaman aşımı `LLM_TIMEOUT` ile ayarlanabiliyor (varsayılan 900 sn) ve zaman aşımına düşen üretim tekrar denenmiyor. Hata veren soru kaydedilip geçiliyor, sonuçlar her sorudan sonra dosyaya yazılıyor. Her cevap için ilk token süresi (prefill) ayrıca ölçülüyor. `--n` ve `--budget` ile bağlam küçültülebiliyor.

---

## 5. Adım Sırası (iş listesi)

**Faz 1**
1. ✅ İskelet: `pyproject.toml`, `.env.example`, `docker-compose.yml` (qdrant), LM Studio smoke test
2. ✅ `loader.py` + `chunker.py` → `data/chunks.jsonl` + chunk istatistikleri (dağılım, en uzun chunk'lar)
3. ✅ `embeddings.py` + `embedding_cache.py` + `ingest.py` → Qdrant'ın dolması
4. ✅ `retrieve.py` → örnek soru ile retrieval debug edilecek (henüz LLM yok)
5. ✅ `ask.py` → uçtan uca cevap
6. ✅ Eval seti + ölçüm aracı, ilk retrieval ölçümü yapıldı. Opsiyonel olarak reranker, LLM kıyası ve embedding profili kıyası (embeddinggemma / bge-m3).

**Faz 2**
7. ✅ Neo4j compose'a eklenecek + `build_graph.py` (deterministik graph + vector index)
8. ✅ Graph retrieval + `ask.py`
9. ✅ Pilot entity extraction (Rust alt kümesi)
10. ✅ Faz 1 ve Faz 2 karşılaştırmalı eval + bulguların yazılması → [docs/FINDINGS.md](./docs/FINDINGS.md), tablolar `eval.report` ile

**Faz 3 (demo arabirimi)**
11. ✅ Hazırlık: `answer_question` dışarıdan retrieval sonucu alabilecek, `AskResult.messages`, retrieval tablosu için yapısal veri, `ui/service.py` (bağlantılar, servis durumu, soru çalıştırma) + testler
12. ✅ Streamlit arayüzü: kenar çubuğu (durum, model, ayarlar, örnek sorular), soru kutusu, tek mod cevap akışı, kaynaklar ve paneller (chunk'lar, graph bilgisi, prompt, süreler)
13. Karşılaştır modu: iki sütun, sıralı üretim, retrieval farkı özeti
14. Demo kolaylıkları: cevap cache'i ve yeniden üret, soru geçmişi, çalıştırma kaydı, ön ısıtma
15. Doğrulama ve doküman: `AppTest` duman testi, README arayüz bölümü, ders akışı için kısa kullanım notu

---

## 6. Açık Konular / Riskler
- **`pytz` bağımlılığı:** `neo4j` Python sürücüsü (6.4.1) `pytz`'yi zorunlu bağımlılık olarak getiriyor. `pytz` PyPI'de deprecated olarak işaretli (2038 sonrası tarihlerde hata veriyor, yerine standart kütüphanedeki `zoneinfo` öneriliyor). Kodumuz `pytz`'yi kullanmıyor ve `import neo4j` onu belleğe yüklemiyor. Sürücü onu sadece saat dilimli tarih-saat değerlerini Python'a çevirirken kullanıyor, bizim sorgularımız ise tarihleri metin ya da sayı olarak döndürüyor. Karar: resmi sürücüyle devam ediliyor, `pytz` dışlanmıyor. Sürücü güncellemelerinde tekrar bakılacak (sürücü `zoneinfo`'ya geçerse bağımlılık kendiliğinden düşer).
- **Donanım:** Geliştirme makinesinde ayrı bir GPU yok. Intel Iris Xe tümleşik grafik, i7-1255U (15 W, 10 çekirdek) ve 32 GB RAM var. Llama 3 8B ile ~3.000 tokenlik bağlamlı bir cevap ~6 dakika sürüyor, embedding hızı ~2,6 chunk/sn. Model seçiminde ve bağlam boyutunda bu sınır gözetilmeli: 8B'den büyük modeller pratik değil.
- **Llama 3 8B Türkçe kalitesi:** model kıyası 1.6'da yapılacak. Context penceresi 8K olduğu için context bütçesi sıkı tutulmalı.
- **Eski yazılar (2003–2012):** ağırlıklı olarak .NET/WCF içeriği var. Zaman filtresi veya tarih ağırlığı gerekebilir, eval sonuçlarına göre karar verilecek.
- **Embedding süresi:** ~10,5 bin chunk işlenecek. Lokal GPU'da birkaç dakika ile onlarca dakika arasında sürebilir. Cache sayesinde bu maliyet bir kez ödenecek.
