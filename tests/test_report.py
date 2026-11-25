import json
from pathlib import Path

from eval.report import build_report, pick_retrieval


def write(path: Path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def summary(hit: float) -> dict:
    base = {"hit_ctx": hit, "recall_ctx": hit, "mrr": hit, "recall_k": hit}
    return {**base, "normal": base, "hard": base}


def question(qid: str, qtype: str, **variants) -> dict:
    return {"id": qid, "type": qtype, "variants": variants}


def test_build_report(tmp_path: Path) -> None:
    write(tmp_path / "old-list-format.json", [{"id": "x"}])  # eski format atlanır
    write(
        tmp_path / "phase1-retrieval-p-20261101-1000.json",
        {
            "phase": 1,
            "summary": {"full": summary(0.5)},
            "questions": [question("q1", "example", full={"recall_ctx": 0.5})],
        },
    )
    write(
        tmp_path / "phase2-retrieval-p-20261102-1000.json",
        {
            "phase": 2,
            "summary": {"exact": summary(0.5), "full": summary(1.0)},
            "questions": [
                question("q1", "example", exact={"recall_ctx": 0.5}, full={"recall_ctx": 1.0}),
                question(
                    "s1", "structural", exact={"structural_ok": False}, full={"structural_ok": True}
                ),
            ],
        },
    )
    write(
        tmp_path / "phase1-answers-m-20261103-1000.json",
        {"model": "m", "summary": {"answered": 1, "citation_rate": 1.0}, "answers": [{}]},
    )
    report = build_report(tmp_path)
    assert "| Faz 1 · full | 0.500 |" in report and "| Faz 2 · full | 1.000 |" in report
    assert "| Faz 2 · full | 1/1 |" in report and "| Faz 2 · exact | 0/1 |" in report
    assert "| q1 | 0.50 | 1.00 | +0.50 |" in report
    assert "| m | 1 | 1 | 1.0 |" in report


def test_pick_retrieval_prefers_more_questions_then_variants() -> None:
    small = {"phase": 2, "questions": [1], "summary": {"a": 1, "b": 2}, "_file": "z"}
    large = {"phase": 2, "questions": [1, 2], "summary": {"a": 1}, "_file": "a"}
    assert pick_retrieval([small, large], 2) is large
    assert pick_retrieval([small], 1) is None
