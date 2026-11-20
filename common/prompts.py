"""LLM prompt'ları ve bağlam (context) biçimlendirme. Faz 1 ve Faz 2 ortak kullanır."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

SYSTEM_PROMPT = """Sen Burak Selim Şenyurt'un teknik blog yazılarına dayanarak soruları \
cevaplayan bir asistansın.

Kurallar:
1. Yalnızca BAĞLAM bölümündeki bilgileri kullan. Bağlamda olmayan bilgi, API veya kod uydurma.
2. Bağlam soruyu cevaplamaya yetmiyorsa bunu açıkça söyle ve bağlamda ne bulduğunu kısaca belirt.
3. Cevabı her zaman Türkçe yaz. Kod, anahtar kelime ve teknik terimler orijinal halleriyle \
kalabilir.
4. Kod örneği istenirse bağlamdaki kodları temel al. Gerekirse kısaltıp sadeleştir ve kod \
bloklarını dil adıyla işaretle (```rust gibi).
5. Bağlamdan aldığın her bilgi için cümlenin sonunda [#n] biçiminde kaynak numarası ver.
6. Cevabın sonuna kaynak listesi yazma, kaynaklar ayrıca gösterilecek.
7. BAĞLAM'da GRAPH BİLGİSİ bölümü varsa oradaki sayılar ve listeler blog veritabanından \
alınmış kesin sonuçlardır. Sayma ve listeleme sorularında bunları aynen kullan."""

USER_TEMPLATE = """BAĞLAM:
{context}

SORU: {question}

(Yalnızca yukarıdaki bağlamı kullanarak Türkçe cevap ver ve [#n] ile kaynak göster.)"""

NO_RAG_SYSTEM_PROMPT = """Sen yardımcı bir yazılım asistanısın. Cevabı her zaman Türkçe yaz. \
Kod ve teknik terimler orijinal halleriyle kalabilir."""

CITATION_RE = re.compile(r"\[#(\d+)\]")


class ContextItem(Protocol):
    payload: dict

    @property
    def text(self) -> str: ...


def format_context(items: Sequence[ContextItem]) -> str:
    blocks = []
    for number, item in enumerate(items, start=1):
        p = item.payload
        section = p.get("section_path") or "Giriş"
        note = f" ({p['context_note']})" if p.get("context_note") else ""
        blocks.append(f"[#{number}] {p['title']} ({p['date']}) — {section}{note}\n{item.text}")
    return "\n\n---\n\n".join(blocks)


def format_facts(facts: Sequence[str]) -> str:
    lines = "\n".join(f"- {fact}" for fact in facts)
    return f"GRAPH BİLGİSİ (blog veritabanından kesin sonuçlar):\n{lines}"


def build_messages(
    question: str, items: Sequence[ContextItem], facts: Sequence[str] | None = None
) -> list[dict[str, str]]:
    context = format_context(items)
    if facts:
        context = f"{format_facts(facts)}\n\n---\n\n{context}" if items else format_facts(facts)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]


def build_no_rag_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": NO_RAG_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def extract_citations(answer: str, n_items: int) -> tuple[list[int], list[int]]:
    """Cevaptaki [#n] atıflarını (geçerli, geçersiz) olarak döner."""
    numbers = sorted({int(n) for n in CITATION_RE.findall(answer)})
    valid = [n for n in numbers if 1 <= n <= n_items]
    invalid = [n for n in numbers if n not in valid]
    return valid, invalid
