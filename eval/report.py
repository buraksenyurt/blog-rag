"""eval/results altındaki ölçümlerden karşılaştırma tablolarını (markdown) üretir.

Kullanım:
    uv run python -m eval.report                       # ekrana bas
    uv run python -m eval.report --out docs/results_tables.md

Seçim kuralı: her faz için soru sayısı en yüksek, sonra varyant sayısı en yüksek, sonra en
yeni retrieval ölçümü alınır. Cevap ölçümlerinin hepsi listelenir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common.config import ROOT

RESULTS_DIR = ROOT / "eval" / "results"
METRICS = (("hit_ctx", "Hit@ctx"), ("recall_ctx", "Recall@ctx"), ("mrr", "MRR@20"),
           ("recall_k", "Recall@20"))  # fmt: skip


def load_results(results_dir: Path = RESULTS_DIR) -> tuple[list[dict], list[dict]]:
    retrievals, answers = [], []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):  # ilk sürümün liste formatı (özet alanı yok)
            continue
        data["_file"] = path.name
        (answers if "answers" in data else retrievals).append(data)
    return retrievals, answers


def pick_retrieval(retrievals: list[dict], phase: int) -> dict | None:
    candidates = [r for r in retrievals if r.get("phase") == phase]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (len(r["questions"]), len(r["summary"]), r["_file"]))


def fmt(value) -> str:
    return "–" if value is None else f"{value:.3f}"


def structural_count(report: dict, variant: str) -> str:
    rows = [
        q["variants"][variant].get("structural_ok")
        for q in report["questions"]
        if q["type"] == "structural" and variant in q["variants"]
    ]
    rows = [r for r in rows if r is not None]
    return f"{sum(1 for r in rows if r)}/{len(rows)}" if rows else "–"


def retrieval_table(phase1: dict | None, phase2: dict | None) -> list[str]:
    columns = []
    if phase1:
        columns.append(
            (
                "Faz 1 · " + next(reversed(phase1["summary"])),
                phase1,
                next(reversed(phase1["summary"])),
            )
        )
    if phase2:
        columns += [(f"Faz 2 · {v}", phase2, v) for v in phase2["summary"]]
    lines = []
    for scope, title in (("all", "Tümü"), ("normal", "Normal sorular"), ("hard", "Zor sorular")):
        lines += [f"**{title}**", "", "| Varyant | " + " | ".join(m[1] for m in METRICS) + " |",
                  "|---|" + "---:|" * len(METRICS)]  # fmt: skip
        for label, report, variant in columns:
            summary = report["summary"][variant]
            values = summary if scope == "all" else summary.get(scope, {})
            lines.append(
                f"| {label} | " + " | ".join(fmt(values.get(m)) for m, _ in METRICS) + " |"
            )
        lines.append("")
    lines += ["**Yapısal sorular (beklenen sayılar graph bilgisinde)**", "", "| Varyant | Doğru |",
              "|---|---:|"]  # fmt: skip
    for label, report, variant in columns:
        lines.append(f"| {label} | {structural_count(report, variant)} |")
    return lines


def question_diff(phase1: dict, phase2: dict, v1: str, v2: str) -> list[str]:
    """Faz 1 ile Faz 2 arasında Recall@ctx'i değişen sorular."""
    by_id = {q["id"]: q for q in phase1["questions"]}
    rows = []
    for q in phase2["questions"]:
        if q["type"] == "structural" or q["id"] not in by_id:
            continue
        a = by_id[q["id"]]["variants"].get(v1, {}).get("recall_ctx")
        b = q["variants"].get(v2, {}).get("recall_ctx")
        if a is not None and b is not None and abs(a - b) > 1e-9:
            rows.append((b - a, q["id"], a, b))
    if not rows:
        return ["Faz 1 ile Faz 2 arasında Recall@ctx'i değişen soru yok."]
    lines = ["| Soru | Faz 1 | Faz 2 | Fark |", "|---|---:|---:|---:|"]
    for delta, qid, a, b in sorted(rows, reverse=True):
        lines.append(f"| {qid} | {a:.2f} | {b:.2f} | {delta:+.2f} |")
    return lines


def answers_table(answers: list[dict]) -> list[str]:
    lines = ["| Dosya | Model | Soru | Cevaplanan | Atıf | Türkçe | Ort. süre | İlk token |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for a in sorted(answers, key=lambda x: x["_file"]):
        s = a.get("summary") or {}
        rows = a["answers"]
        lines.append(
            f"| {a['_file'].split('-answers-')[0]}·{a['_file'][-18:-5]} | {a.get('model')} | "
            f"{len(rows)} | {s.get('answered', '–')} | {s.get('citation_rate', '–')} | "
            f"{s.get('turkish_ratio', '–')} | {s.get('generation_seconds', '–')} sn | "
            f"{s.get('first_token_seconds', '–')} sn |"
        )
    return lines


def build_report(results_dir: Path = RESULTS_DIR) -> str:
    retrievals, answers = load_results(results_dir)
    phase1, phase2 = pick_retrieval(retrievals, 1), pick_retrieval(retrievals, 2)
    lines = ["# Ölçüm tabloları", "", "_`uv run python -m eval.report` ile üretildi._", ""]
    sources = [r["_file"] for r in (phase1, phase2) if r]
    lines += ["Kaynak dosyalar: " + ", ".join(f"`{s}`" for s in sources), ""]
    if phase1 or phase2:
        n = len((phase2 or phase1)["questions"])
        lines += [f"## Retrieval ({n} soru)", "", *retrieval_table(phase1, phase2), ""]
    if phase1 and phase2:
        v1 = next(reversed(phase1["summary"]))
        v2 = "full" if "full" in phase2["summary"] else next(reversed(phase2["summary"]))
        lines += [f"## Soru bazında fark (Faz 1 · {v1} → Faz 2 · {v2}, Recall@ctx)", "",
                  *question_diff(phase1, phase2, v1, v2), ""]  # fmt: skip
    if answers:
        lines += ["## LLM cevapları", "", *answers_table(answers), ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ölçüm tablolarını üret")
    parser.add_argument("--out", type=Path, help="markdown dosyasına yaz")
    args = parser.parse_args()
    report = build_report()
    if args.out:
        out = args.out if args.out.is_absolute() else ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"-> {out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
