"""Faz 1 değerlendirme: retrieval varyantlarını altın soru setiyle kıyaslar.

Kullanım:
    uv run python -m eval.run_eval                          # retrieval kıyası (LLM yok, hızlı)
    uv run python -m eval.run_eval --variants plain,full
    uv run python -m eval.run_eval --with-llm --limit 5     # ilk 5 soruya LLM cevabı da üret
    uv run python -m eval.run_eval --with-llm --ids rust-variables,go-defer-panic --model qwen3-8b

Metrikler (post seviyesinde, structural sorular hariç):
    Hit@ctx    LLM'e giden bağlamda en az bir beklenen yazı var mı
    Recall@ctx beklenen yazıların ne kadarı bağlamda
    MRR@k      k adayın post sıralamasında ilk beklenen yazının 1/sırası
    Recall@k   beklenen yazıların ne kadarı k aday içinde
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import yaml
from qdrant_client import QdrantClient

from common.config import ROOT, ensure_writable, get_settings, load_embedding_profile
from common.embeddings import Embedder
from common.llm import LLM
from eval.metrics import hit, recall, reciprocal_rank, turkish_ratio, unique_posts
from phase1_vector.ask import answer_question
from phase1_vector.retrieve import RetrievalConfig, RetrievalResult, Retriever

QUESTIONS_PATH = ROOT / "eval" / "questions.yaml"
RESULTS_DIR = ROOT / "eval" / "results"
RETRIEVAL_TYPES = {"example", "relational"}

VARIANTS: dict[str, dict] = {
    "plain": {"use_filter": False, "code_boost": 0.0, "expand_short": False},
    "filter": {"use_filter": True, "code_boost": 0.0, "expand_short": False},
    "filter+code": {"use_filter": True, "expand_short": False},
    "full": {},
}
METRICS = ("hit_ctx", "recall_ctx", "mrr", "recall_k")


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]


def score(question: dict, result: RetrievalResult) -> dict:
    relevant = question.get("relevant") or []
    context_posts = unique_posts(h.post_slug for h in result.hits)
    candidate_posts = unique_posts(h.post_slug for h in result.candidates)
    return {
        "hit_ctx": hit(relevant, context_posts),
        "recall_ctx": recall(relevant, context_posts),
        "mrr": reciprocal_rank(relevant, candidate_posts),
        "recall_k": recall(relevant, candidate_posts),
        "context_posts": context_posts,
        "top_candidates": candidate_posts[:5],
        "context_tokens": result.context_tokens,
    }


def aggregate(rows: list[dict]) -> dict[str, float]:
    return {m: round(mean(r[m] for r in rows), 3) if rows else 0.0 for m in METRICS}


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", text.lower()).strip("-")


def run_retrieval(questions: list[dict], retriever: Retriever, variants: list[str]) -> dict:
    per_question: list[dict] = []
    for q in questions:
        vector = retriever.embedder.embed_query(q["question"])
        entry = {
            "id": q["id"],
            "type": q["type"],
            "relevant": q.get("relevant") or [],
            "variants": {},
        }
        for name in variants:
            entry["variants"][name] = score(
                q, retriever.retrieve(q["question"], vector, **VARIANTS[name])
            )
        per_question.append(entry)
        full = entry["variants"][variants[-1]]
        print(
            f"  {q['id']:<28} hit={full['hit_ctx']:.0f} recall={full['recall_ctx']:.2f} "
            f"mrr={full['mrr']:.2f}"
        )

    summary: dict[str, dict] = {}
    for name in variants:
        rows = [e["variants"][name] for e in per_question if e["type"] in RETRIEVAL_TYPES]
        summary[name] = aggregate(rows)
        for qtype in sorted(RETRIEVAL_TYPES):
            typed = [e["variants"][name] for e in per_question if e["type"] == qtype]
            summary[name][f"hit_ctx[{qtype}]"] = aggregate(typed)["hit_ctx"]
    return {"summary": summary, "questions": per_question}


def print_summary(report: dict, variants: list[str], questions: list[dict]) -> None:
    n = sum(1 for q in questions if q["type"] in RETRIEVAL_TYPES)
    print(f"\nRetrieval özeti ({n} soru, structural hariç)\n")
    print(f"{'varyant':<14}" + "".join(f"{m:>12}" for m in METRICS))
    for name in variants:
        s = report["summary"][name]
        print(f"{name:<14}" + "".join(f"{s[m]:>12.3f}" for m in METRICS))

    last = variants[-1]
    misses = [
        e
        for e in report["questions"]
        if e["type"] in RETRIEVAL_TYPES and e["variants"][last]["hit_ctx"] == 0
    ]
    if misses:
        print(f"\nBağlama hiç beklenen yazı giremeyen sorular ({last}):")
        for e in misses:
            top = ", ".join(s[:45] for s in e["variants"][last]["top_candidates"][:3])
            print(f"  - {e['id']}: ilk adaylar -> {top}")


def run_llm(questions: list[dict], retriever: Retriever, llm: LLM) -> list[dict]:
    rows: list[dict] = []
    for i, q in enumerate(questions, start=1):
        print(f"  [{i}/{len(questions)}] {q['id']} ...", flush=True)
        result = answer_question(q["question"], llm, retriever)
        relevant = set(q.get("relevant") or [])
        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "expected": q.get("expected"),
                "answer": result.answer,
                "sources": [
                    {
                        "n": n,
                        "id": h.payload["id"],
                        "title": h.payload["title"],
                        "url": h.payload["url"],
                        "relevant": h.post_slug in relevant,
                        "cited": n in result.cited,
                    }
                    for n, h in enumerate(result.hits, start=1)
                ],
                "cited": result.cited,
                "invalid_citations": result.invalid_citations,
                "turkish_ratio": turkish_ratio(result.answer),
                "generation_seconds": result.generation_seconds,
            }
        )
    return rows


def write_answers_markdown(rows: list[dict], path: Path, model: str, profile: str) -> None:
    lines = [
        f"# Faz 1 cevapları — {model} / {profile}",
        "",
        "Her cevabı 1-5 arası puanlayın (5 = doğru, bağlama dayalı, Türkçe, işe yarar kod).",
        "",
    ]
    for row in rows:
        lines += [f"## {row['id']} ({row['type']})", "", f"**Soru:** {row['question']}", ""]
        if row["expected"]:
            lines += [f"**Beklenen:** {row['expected']}", ""]
        lines += [row["answer"].strip(), "", "**Kaynaklar:**", ""]
        for s in row["sources"]:
            flags = ("✓atıf " if s["cited"] else "") + ("★beklenen" if s["relevant"] else "")
            lines.append(f"- [#{s['n']}] {s['title']} — `{s['id']}` {flags}".rstrip())
        lines += [
            "",
            f"Atıf: {row['cited'] or 'yok'} · Türkçe oranı: {row['turkish_ratio']} · "
            f"Süre: {row['generation_seconds']}s",
            "",
            "**Puan (1-5):** ",
            "",
            "**Not:** ",
            "",
            "---",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Faz 1 değerlendirme")
    parser.add_argument("--profile")
    parser.add_argument(
        "--variants", default=",".join(VARIANTS), help=f"virgülle ayrılmış: {', '.join(VARIANTS)}"
    )
    parser.add_argument("--ids", help="sadece bu soru id'leri (virgülle)")
    parser.add_argument("--with-llm", action="store_true", help="LLM cevaplarını da üret")
    parser.add_argument("--model", help="LLM modeli (varsayılan .env LLM_MODEL)")
    parser.add_argument("--limit", type=int, help="LLM ile cevaplanacak soru sayısı")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        parser.error(f"bilinmeyen varyant: {unknown}")

    settings = get_settings()
    profile = load_embedding_profile(settings, args.profile)
    questions = load_questions()
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        questions = [q for q in questions if q["id"] in wanted]

    retriever = Retriever(
        Embedder(settings, profile), QdrantClient(url=settings.qdrant_url, timeout=30)
    )
    out_dir = ensure_writable(RESULTS_DIR, settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = stamp()

    print(f"Retrieval değerlendirmesi: {len(questions)} soru, varyantlar: {variants}\n")
    report = run_retrieval(questions, retriever, variants)
    print_summary(report, variants, questions)
    retrieval_path = out_dir / f"phase1-retrieval-{profile.name}-{run_stamp}.json"
    retrieval_path.write_text(
        json.dumps(
            {
                "phase": 1,
                "embedding_profile": profile.name,
                "created": run_stamp,
                "retrieval_defaults": asdict(RetrievalConfig()),
                "variants": {v: VARIANTS[v] for v in variants},
                **report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {retrieval_path}")

    if args.with_llm:
        llm = LLM(settings, model=args.model)
        selected = questions[: args.limit] if args.limit else questions
        print(f"\nLLM cevapları: {llm.model}, {len(selected)} soru (varyant: full)\n")
        rows = run_llm(selected, retriever, llm)
        base = out_dir / f"phase1-answers-{slugify(llm.model)}-{run_stamp}"
        base.with_suffix(".json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_answers_markdown(rows, base.with_suffix(".md"), llm.model, profile.name)
        cited = mean(1.0 if r["cited"] else 0.0 for r in rows)
        tr = [r["turkish_ratio"] for r in rows if r["turkish_ratio"] is not None]
        print(
            f"\nAtıf yapan cevap oranı: {cited:.2f} | ortalama Türkçe oranı: "
            f"{mean(tr) if tr else 0:.2f} | ortalama süre: "
            f"{mean(r['generation_seconds'] for r in rows):.0f}s"
        )
        print(f"-> {base.with_suffix('.md')}  (puanlama için)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
