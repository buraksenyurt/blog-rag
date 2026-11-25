# blog-rag — Bulgular

Bu doküman iki RAG kurgusunun (Faz 1: Qdrant ile vektör RAG, Faz 2: Neo4j ile graph RAG)
aynı veri ve aynı soru setiyle karşılaştırılmasının sonuçlarını özetler. Ham tablolar
[`results_tables.md`](./results_tables.md) dosyasındadır ve `uv run python -m eval.report --out docs/results_tables.md`
ile yeniden üretilebilir. Kararların gerekçeleri ve ara ölçümler [`PLAN.md`](../PLAN.md) içindedir.

## Kurulum özeti

| | |
|---|---|
| Kaynak | `buraksenyurt.github.io/_posts`, 868 yazı (Foto İpucu hariç), 10.515 chunk |
| Embedding | LM Studio, `text-embedding-embeddinggemma-300m` (768 boyut), cache'li |
| LLM | LM Studio; Llama 3 8B, Gemma 3 4B QAT ölçüldü (seçim park edildi) |
| Faz 1 | Qdrant 1.20; teknoloji filtresi + kod niyeti boost'u + MMR + kısa chunk genişletmesi |
| Faz 2 | Neo4j 5.26; Post/Chunk/Tag/Category/Language/Series graph'ı, tam cosine giriş araması, link/seri genişletmesi, yapısal Cypher şablonları, kavram katmanı (pilot) |
| Eval | 48 soru: 32 normal, 13 zor (teknoloji adı geçmeyen), 3 yapısal |
| Donanım | Ayrı GPU yok (Iris Xe), i7-1255U, 32 GB RAM |

## Retrieval sonuçları (48 soru)

| Varyant | Hit@ctx | Recall@ctx | MRR@20 | Recall@20 | Zor: Recall@ctx | Zor: Recall@20 | Yapısal |
|---|---:|---:|---:|---:|---:|---:|---:|
| Faz 1 (Qdrant, full) | 0,956 | 0,819 | 0,844 | 0,900 | 0,590 | 0,712 | 0/3 |
| Faz 2 · vector (Neo4j HNSW index) | 0,889 | 0,785 | 0,817 | 0,878 | 0,513 | 0,635 | 0/3 |
| Faz 2 · exact (tam cosine) | 0,956 | 0,819 | 0,847 | 0,900 | 0,590 | 0,712 | 0/3 |
| Faz 2 · graph (+ link/seri genişletmesi) | 0,956 | **0,830** | **0,861** | **0,911** | **0,628** | **0,750** | 0/3 |
| Faz 2 · full (+ yapısal sorgular) | 0,956 | **0,830** | **0,861** | **0,911** | **0,628** | **0,750** | **3/3** |
| Faz 2 · concepts (+ kavram katmanı) | 0,956 | 0,807 | 0,861 | 0,911 | 0,590 | 0,750 | 3/3 |

## Bulgular

1. **Graph'ın en net katkısı yapısal sorular.** "Blogda Rust ile ilgili kaç yazı var?",
   "Go yazıları hangi yıllarda?", "Rust Pratikleri serisinde hangi yazılar var?" sorularını
   vektör araması hiç cevaplayamıyor (seri sorusunda recall 0,20). Graph üçünü de Cypher ile
   kesin olarak cevaplıyor (seri sorusunda recall 1,00). Bu sonuç LLM'e "GRAPH BİLGİSİ"
   olarak veriliyor.

2. **Graph genişletmesi ölçülebilir ama küçük bir iyileşme sağlıyor, en çok zor sorularda.**
   Zor sorularda Recall@ctx 0,590 → 0,628, MRR 0,679 → 0,723, Recall@20 0,712 → 0,750.
   Normal sorularda etkisi neredeyse yok. Soru bazında 3 kazanç, 2 kayıp var:
   - Kazanç: `rust-trait-objects` (seri komşusu "State Tasarım Kalıbı" yazısı geldi),
     `hard-lock-free-counter` (TPL serisinden "Interlocked" yazısı geldi), `go-no-inheritance`.
   - Kayıp: `rust-variables`, `wcf-uritemplate`. Geniş sorularda seri komşuları
     (Rust Pratikleri: State, Lifetimes; WCF WebHttp serisi) bağlamdaki asıl yazıları dışarı itiyor.
   Seri genişletmesi, dar konulu sorularda faydalı, geniş konulu sorularda gürültülü.

3. **Neo4j vector index'i bu veri için uygun değil.** Aynı vektörlerle Neo4j'nin HNSW index'i
   Faz 1'in gerisinde kaldı (zor sorularda Hit@ctx 0,692 / 0,846). Sebepler:
   - Varsayılan quantization, embeddinggemma'nın dar skor aralığında (0,73–0,77) sıralamayı
     bozuyor. Kapatmak farkı kapattı ama tamamen değil.
   - Kalan fark yaklaşık aramadan (HNSW) geliyor.
   10 bin chunk için tam cosine hem kesin hem yeterince hızlı. Faz 2'nin giriş araması bu
   yüzden tam cosine yapıldı. `exact` varyantı Faz 1 ile birebir aynı sonucu veriyor.
   Böylece iki faz arasındaki fark yalnızca graph'tan geliyor.

4. **LLM ile kavram çıkarımı (pilot, 46 Rust yazısı) retrieval'a katkı sağlamadı.**
   `concepts` varyantı Recall@ctx'i 0,830'dan 0,807'ye düşürdü. Kavram üzerinden gelen
   adaylar bağlamdaki doğru yazıları dışarı itti. Pilotun kendisi de öğretici oldu:
   - Prompt'taki örnek kavramlar sızdı: atamaların %38'i yazıda geçmeyen kavramlardı
     (`shadowing` 18 yazıda, 18'i de uydurma). Kanıt kontrolü (kavram yazıda geçmiyorsa
     bağlama) zorunlu oldu.
   - Model kod ifadelerini (`println!`, `String::from`) ve Türkçe kelimeyle çakışan adları
     (`var`) kavram sandı.
   - Model takma adları metinde olmayan ya da yanlış terimlerle üretebiliyor
     ("kopyalama" → borrowing). Chunk eşleştirmesi yalnızca kavram adlarıyla yapılıyor.
   Bu donanımda yazı başına ~2 dakika süren çıkarım için kazanç, maliyeti karşılamıyor.
   Kavram katmanı varsayılan olarak kapalı. Keşif amaçlı Cypher sorguları için graph'ta duruyor.

5. **Teknoloji filtresi ve kod boost'u ölçülebilir fayda sağlamadı.** Soruların çoğu
   teknoloji adını içerdiği için embedding zaten doğru yazılara gidiyor. Zor sorularda ise
   filtre hiç devreye girmiyor.

6. **İki zor soru her iki fazda da ıskalanıyor.** `hard-cancel-parallel-work` sorusunda
   iptal yazıları yerine diğer Parallel.For yazıları geliyor. `hard-compile-time-marker`
   (PhantomData) sorusunda "struct" kelimesi 2003 tarihli C# struct yazısını öne çekiyor.
   Bunlar anlamsal arama açıkları. Graph değil, hibrit arama (BM25 + vektör) ya da
   reranker gerektiriyor.

7. **LLM tarafında darboğaz donanım.** Ayrı GPU olmayan bu makinede:
   - Llama 3 8B ortalama 149 sn'de cevap veriyor, atıf kuralına 1/5 uyuyor ve zaman zaman
     kendine soru uydurup cevaplıyor.
   - Gemma 3 4B QAT ortalama 68 sn'de cevap veriyor ve 5/5 atıf yapıyor.
   - İki model de Rust kavramlarında hata yapıyor.
   - Thinking modelleri (qwen3-4b-thinking) bütün çıktıyı düşünme bölümüne yazıp boş cevap
     döndürüyor.
   - İlk token ~5 sn: süreyi bağlam işleme değil token üretimi belirliyor.

## Önerilen sonraki adımlar

- **Hibrit arama** (Qdrant sparse vektör ya da Neo4j full-text index + vektör). Zor
  sorulardaki anlamsal açık için en olası kazanç burada.
- **Seri genişletmesini koşula bağlamak.** Örneğin sadece aday yazılar aynı seride
  yoğunlaşıyorsa ya da soru "serisi" kelimesini içeriyorsa uygulanabilir. Böylece geniş
  sorulardaki kayıp önlenir.
- **LLM seçimi.** Qwen3 4B Instruct 2507 ölçülüp Gemma 3 4B ile karşılaştırılmalı.
  Cevap seviyesinde Faz 1 / Faz 2 kıyası (özellikle yapısal sorular) seçilen modelle
  `eval.run_eval --with-llm` ile yapılmalı.
- **Kavram çıkarımı v4** (yeni prompt) gece çalıştırılıp kanıtsız atama oranı ölçülebilir.
  Retrieval'a katkı vermediği sürece varsayılan kapalı kalmalı.
