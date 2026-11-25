# Ölçüm tabloları

_`uv run python -m eval.report` ile üretildi._

Kaynak dosyalar: `phase1-retrieval-embeddinggemma-20261113-1712.json`, `phase2-retrieval-embeddinggemma-20261125-2347.json`

## Retrieval (48 soru)

**Tümü**

| Varyant | Hit@ctx | Recall@ctx | MRR@20 | Recall@20 |
|---|---:|---:|---:|---:|
| Faz 1 · full | 0.956 | 0.819 | 0.844 | 0.900 |
| Faz 2 · vector | 0.889 | 0.785 | 0.817 | 0.878 |
| Faz 2 · exact | 0.956 | 0.819 | 0.847 | 0.900 |
| Faz 2 · graph | 0.956 | 0.830 | 0.861 | 0.911 |
| Faz 2 · full | 0.956 | 0.830 | 0.861 | 0.911 |
| Faz 2 · concepts | 0.956 | 0.807 | 0.861 | 0.911 |

**Normal sorular**

| Varyant | Hit@ctx | Recall@ctx | MRR@20 | Recall@20 |
|---|---:|---:|---:|---:|
| Faz 1 · full | 1.000 | 0.911 | 0.915 | 0.977 |
| Faz 2 · vector | 0.969 | 0.896 | 0.915 | 0.977 |
| Faz 2 · exact | 1.000 | 0.911 | 0.915 | 0.977 |
| Faz 2 · graph | 1.000 | 0.911 | 0.917 | 0.977 |
| Faz 2 · full | 1.000 | 0.911 | 0.917 | 0.977 |
| Faz 2 · concepts | 1.000 | 0.896 | 0.917 | 0.977 |

**Zor sorular**

| Varyant | Hit@ctx | Recall@ctx | MRR@20 | Recall@20 |
|---|---:|---:|---:|---:|
| Faz 1 · full | 0.846 | 0.590 | 0.669 | 0.712 |
| Faz 2 · vector | 0.692 | 0.513 | 0.577 | 0.635 |
| Faz 2 · exact | 0.846 | 0.590 | 0.679 | 0.712 |
| Faz 2 · graph | 0.846 | 0.628 | 0.723 | 0.750 |
| Faz 2 · full | 0.846 | 0.628 | 0.723 | 0.750 |
| Faz 2 · concepts | 0.846 | 0.590 | 0.723 | 0.750 |

**Yapısal sorular (beklenen sayılar graph bilgisinde)**

| Varyant | Doğru |
|---|---:|
| Faz 1 · full | – |
| Faz 2 · vector | 0/3 |
| Faz 2 · exact | 0/3 |
| Faz 2 · graph | 0/3 |
| Faz 2 · full | 3/3 |
| Faz 2 · concepts | 3/3 |

## Soru bazında fark (Faz 1 · full → Faz 2 · full, Recall@ctx)

| Soru | Faz 1 | Faz 2 | Fark |
|---|---:|---:|---:|
| rust-trait-objects | 0.50 | 1.00 | +0.50 |
| hard-lock-free-counter | 0.50 | 1.00 | +0.50 |
| go-no-inheritance | 0.50 | 1.00 | +0.50 |
| wcf-uritemplate | 1.00 | 0.50 | -0.50 |
| rust-variables | 1.00 | 0.50 | -0.50 |

## LLM cevapları

| Dosya | Model | Soru | Cevaplanan | Atıf | Türkçe | Ort. süre | İlk token |
|---|---|---:|---:|---:|---:|---:|---:|
| phase1·20261120-1725 | gemma-3-4b-it-qat | 5 | 5 | 1.0 | 0.97 | 68.3 sn | 5.6 sn |
| phase1·20261113-1713 | meta-llama-3-8b-instruct | 3 | 3 | 0.33 | 0.98 | 240.7 sn | 51.3 sn |
| phase1·20261120-1803 | meta-llama-3-8b-instruct | 5 | 5 | 0.2 | 1.0 | 148.8 sn | 4.5 sn |
| phase1·20261120-1731 | qwen/qwen3-4b-thinking-2507 | 5 | 5 | 0.0 | None | 271.7 sn | None sn |
