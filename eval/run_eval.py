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
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import yaml
from openai import APIError
from qdrant_client import QdrantClient

from common.answering import answer_question, retrieval_config
from common.config import ROOT, ensure_writable, get_settings, load_embedding_profile
from common.embeddings import Embedder
from common.llm import LLM
from common.retrieval import RetrievalResult
from eval.metrics import hit, recall, reciprocal_rank, turkish_ratio, unique_posts
from phase1_vector.retrieve import Retriever
from phase2_graph.retrieve import GraphRetrievalConfig, GraphRetriever
from phase2_graph.store import connect, make_runner

QUESTIONS_PATH = ROOT / "eval" / "questions.yaml"
RESULTS_DIR = ROOT / "eval" / "results"
RETRIEVAL_TYPES = {"example", "relational"}

VARIANTS: dict[str, dict] = {
    "plain": {"use_filter": False, "code_boost": 0.0, "expand_short": False},
    "filter": {"use_filter": True, "code_boost": 0.0, "expand_short": False},
    "filter+code": {"use_filter": True, "expand_short": False},
    "full": {},
}
# Faz 2: vector index / tam cosine (index'siz, kıyas için) / + graph genişletme / + yapısal
GRAPH_VARIANTS: dict[str, dict] = {
    "vector": {"graph_expand": False, "structural": False},
    "exact": {"exact_search": True, "graph_expand": False, "structural": False},
    "graph": {"graph_expand": True, "structural": False},
    "full": {},
}
PHASE_VARIANTS = {1: VARIANTS, 2: GRAPH_VARIANTS}
NUMBER_RE = re.compile(r"\d+")
METRICS = ("hit_ctx", "recall_ctx", "mrr", "recall_k")


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]


def structural_ok(question: dict, facts: list[str]) -> bool | None:
    """Beklenen cevaptaki bütün sayılar graph bilgisinde geçiyor mu? (structural sorular)"""
    expected = question.get("expected")
    if question["type"] != "structural" or not expected:
        return None
    wanted = set(NUMBER_RE.findall(expected))
    found = set(NUMBER_RE.findall(" ".join(facts)))
    return bool(facts) and wanted <= found


def score(question: dict, result: RetrievalResult) -> dict:
    relevant = question.get("relevant") or []
    facts = list(getattr(result, "facts", []) or [])
    fact_posts = list(getattr(result, "fact_posts", []) or [])
    context_posts = unique_posts([*fact_posts, *(h.post_slug for h in result.hits)])
    candidate_posts = unique_posts([*fact_posts, *(h.post_slug for h in result.candidates)])
    return {
        "hit_ctx": hit(relevant, context_posts),
        "recall_ctx": recall(relevant, context_posts),
        "mrr": reciprocal_rank(relevant, candidate_posts),
        "recall_k": recall(relevant, candidate_posts),
        "context_posts": context_posts,
        "top_candidates": candidate_posts[:5],
        "context_tokens": result.context_tokens,
        "facts": facts,
        "structural_ok": structural_ok(question, facts),
    }


def aggregate(rows: list[dict]) -> dict[str, float]:
    return {m: round(mean(r[m] for r in rows), 3) if rows else 0.0 for m in METRICS}


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", text.lower()).strip("-")


def run_retrieval(
    questions: list[dict],
    retriever: Retriever,
    variants: list[str],
    variant_defs: dict[str, dict] | None = None,
) -> dict:
    variant_defs = variant_defs or VARIANTS
    per_question: list[dict] = []
    for q in questions:
        vector = retriever.embedder.embed_query(q["question"])
        entry = {
            "id": q["id"],
            "type": q["type"],
            "difficulty": q.get("difficulty", "normal"),
            "relevant": q.get("relevant") or [],
            "variants": {},
        }
        for name in variants:
            entry["variants"][name] = score(
                q, retriever.retrieve(q["question"], vector, **variant_defs[name])
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
        for level in ("normal", "hard"):
            leveled = [
                e["variants"][name]
                for e in per_question
                if e["type"] in RETRIEVAL_TYPES and e["difficulty"] == level
            ]
            summary[name][level] = aggregate(leveled)
        structural = [
            e["variants"][name]["structural_ok"]
            for e in per_question
            if e["variants"][name]["structural_ok"] is not None
        ]
        summary[name]["structural_ok"] = (
            round(mean(1.0 if ok else 0.0 for ok in structural), 3) if structural else None
        )
    return {"summary": summary, "questions": per_question}


def print_summary(report: dict, variants: list[str], questions: list[dict]) -> None:
    n = sum(1 for q in questions if q["type"] in RETRIEVAL_TYPES)
    print(f"\nRetrieval özeti ({n} soru, structural hariç)\n")
    print(f"{'varyant':<14}" + "".join(f"{m:>12}" for m in METRICS))
    for name in variants:
        s = report["summary"][name]
        print(f"{name:<14}" + "".join(f"{s[m]:>12.3f}" for m in METRICS))

    hard = sum(
        1 for q in questions if q["type"] in RETRIEVAL_TYPES and q.get("difficulty") == "hard"
    )
    if hard:
        for level, label in (("normal", "Normal sorular"), ("hard", "Zor sorular")):
            count = n - hard if level == "normal" else hard
            print(f"\n{label} ({count})")
            for name in variants:
                s = report["summary"][name][level]
                print(f"{name:<14}" + "".join(f"{s[m]:>12.3f}" for m in METRICS))

    structural = [e for e in report["questions"] if e["type"] == "structural"]
    if structural:
        print(f"\nYapısal sorular ({len(structural)})")
        for name in variants:
            marks = []
            for e in structural:
                v = e["variants"][name]
                ok = v["structural_ok"]
                label = "doğru" if ok else ("yanlış/yok" if ok is False else "-")
                marks.append(f"{e['id']}: {label} (recall {v['recall_ctx']:.2f})")
            print(f"{name:<14}" + " | ".join(marks))

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


def run_llm(
    questions: list[dict],
    retriever: Retriever,
    llm: LLM,
    save: Callable[[list[dict]], None] | None = None,
) -> list[dict]:
    """Soruları LLM ile cevaplar. Cevaplar stream edilir (zaman aşımı token'lar arası
    beklemeye uygulanır), hata veren soru kaydedilip geçilir ve her sorudan sonra
    `save` çağrılır; yarıda kesilen çalıştırmada da elde edilen cevaplar kaybolmaz."""
    rows: list[dict] = []
    for i, q in enumerate(questions, start=1):
        print(f"  [{i}/{len(questions)}] {q['id']} ...", end="", flush=True)
        relevant = set(q.get("relevant") or [])
        row = {
            "id": q["id"],
            "type": q["type"],
            "difficulty": q.get("difficulty", "normal"),
            "question": q["question"],
            "expected": q.get("expected"),
            "answer": "",
            "sources": [],
            "cited": [],
            "invalid_citations": [],
            "turkish_ratio": None,
            "generation_seconds": None,
            "first_token_seconds": None,
            "facts": [],
            "error": None,
        }
        try:
            result = answer_question(q["question"], llm, retriever, on_token=lambda _t: None)
        except APIError as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f" HATA ({type(exc).__name__})", flush=True)
        else:
            row.update(
                answer=result.answer,
                sources=[
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
                cited=result.cited,
                invalid_citations=result.invalid_citations,
                turkish_ratio=turkish_ratio(result.answer),
                generation_seconds=result.generation_seconds,
                first_token_seconds=result.first_token_seconds,
                facts=result.facts,
            )
            print(
                f" {result.generation_seconds:.0f}s (ilk token {result.first_token_seconds}s), "
                f"atıf: {result.cited or 'yok'}",
                flush=True,
            )
        rows.append(row)
        if save is not None:
            save(rows)
    return rows


def summarize_answers(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    tr = [r["turkish_ratio"] for r in ok if r["turkish_ratio"] is not None]
    first = [r["first_token_seconds"] for r in ok if r["first_token_seconds"] is not None]
    return {
        "answered": len(ok),
        "errors": len(rows) - len(ok),
        "citation_rate": round(mean(1.0 if r["cited"] else 0.0 for r in ok), 2) if ok else 0.0,
        "turkish_ratio": round(mean(tr), 2) if tr else None,
        "generation_seconds": round(mean(r["generation_seconds"] for r in ok), 1) if ok else None,
        "first_token_seconds": round(mean(first), 1) if first else None,
    }


def write_answers_markdown(
    rows: list[dict], path: Path, model: str, profile: str, title: str = "Faz 1 cevapları"
) -> None:
    lines = [
        f"# {title} — {model} / {profile}",
        "",
        "Her cevabı 1-5 arası puanlayın (5 = doğru, bağlama dayalı, Türkçe, işe yarar kod).",
        "",
    ]
    for row in rows:
        title = f"## {row['id']} ({row['type']}" + (
            ", zor)" if row.get("difficulty") == "hard" else ")"
        )
        lines += [title, "", f"**Soru:** {row['question']}", ""]
        if row["expected"]:
            lines += [f"**Beklenen:** {row['expected']}", ""]
        if row["error"]:
            lines += [f"**HATA:** {row['error']}", "", "---", ""]
            continue
        if row.get("facts"):
            lines += ["**Graph bilgisi:**", "", *(f"- {f}" for f in row["facts"]), ""]
        lines += [row["answer"].strip(), "", "**Kaynaklar:**", ""]
        for s in row["sources"]:
            flags = ("✓atıf " if s["cited"] else "") + ("★beklenen" if s["relevant"] else "")
            lines.append(f"- [#{s['n']}] {s['title']} — `{s['id']}` {flags}".rstrip())
        lines += [
            "",
            f"Atıf: {row['cited'] or 'yok'} · Türkçe oranı: {row['turkish_ratio']} · "
            f"Süre: {row['generation_seconds']}s (ilk token {row['first_token_seconds']}s)",
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
    parser = argparse.ArgumentParser(description="Faz 1 / Faz 2 değerlendirme")
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    parser.add_argument("--profile")
    parser.add_argument(
        "--variants",
        help="virgülle ayrılmış. Faz 1: "
        + ", ".join(VARIANTS)
        + " | Faz 2: "
        + ", ".join(GRAPH_VARIANTS),
    )
    parser.add_argument("--ids", help="sadece bu soru id'leri (virgülle)")
    parser.add_argument("--with-llm", action="store_true", help="LLM cevaplarını da üret")
    parser.add_argument("--model", help="LLM modeli (varsayılan .env LLM_MODEL)")
    parser.add_argument("--limit", type=int, help="LLM ile cevaplanacak soru sayısı")
    parser.add_argument("--n", type=int, help="LLM'e gidecek chunk sayısı (varsayılan 6)")
    parser.add_argument("--budget", type=int, help="bağlam token bütçesi (varsayılan 3500)")
    args = parser.parse_args()

    variant_defs = PHASE_VARIANTS[args.phase]
    variants = [
        v.strip() for v in (args.variants or ",".join(variant_defs)).split(",") if v.strip()
    ]
    unknown = set(variants) - set(variant_defs)
    if unknown:
        parser.error(f"bilinmeyen varyant: {unknown}")

    settings = get_settings()
    profile = load_embedding_profile(settings, args.profile)
    questions = load_questions()
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        questions = [q for q in questions if q["id"] in wanted]

    driver = None
    embedder = Embedder(settings, profile)
    if args.phase == 1:
        retriever = Retriever(
            embedder,
            QdrantClient(url=settings.qdrant_url, timeout=30),
            retrieval_config(args.n, args.budget),
        )
    else:
        driver = connect(settings)
        retriever = GraphRetriever(
            embedder,
            make_runner(driver, settings.neo4j_database),
            retrieval_config(args.n, args.budget, base=GraphRetrievalConfig()),
        )
    try:
        return evaluate(args, settings, profile, questions, retriever, variants, variant_defs)
    finally:
        if driver is not None:
            driver.close()


def evaluate(args, settings, profile, questions, retriever, variants, variant_defs) -> int:
    phase = args.phase
    out_dir = ensure_writable(RESULTS_DIR, settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = stamp()

    print(f"Retrieval değerlendirmesi: {len(questions)} soru, varyantlar: {variants}\n")
    report = run_retrieval(questions, retriever, variants, variant_defs)
    print_summary(report, variants, questions)
    retrieval_path = out_dir / f"phase{phase}-retrieval-{profile.name}-{run_stamp}.json"
    retrieval_path.write_text(
        json.dumps(
            {
                "phase": phase,
                "embedding_profile": profile.name,
                "created": run_stamp,
                "retrieval_config": asdict(retriever.config),
                "variants": {v: variant_defs[v] for v in variants},
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
        base = out_dir / f"phase{phase}-answers-{slugify(llm.model)}-{run_stamp}"

        def save(rows: list[dict]) -> None:
            payload = {
                "model": llm.model,
                "embedding_profile": profile.name,
                "retrieval_config": asdict(retriever.config),
                "summary": summarize_answers(rows),
                "answers": rows,
            }
            base.with_suffix(".json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            write_answers_markdown(
                rows, base.with_suffix(".md"), llm.model, profile.name, f"Faz {phase} cevapları"
            )

        rows = run_llm(selected, retriever, llm, save=save)
        summary = summarize_answers(rows)
        print(
            f"\nCevaplanan: {summary['answered']}/{len(rows)} | atıf yapan: "
            f"{summary['citation_rate']:.2f} | Türkçe oranı: {summary['turkish_ratio']} | "
            f"ort. süre: {summary['generation_seconds']}s "
            f"(ilk token {summary['first_token_seconds']}s)"
        )
        print(f"-> {base.with_suffix('.md')}  (puanlama için)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
