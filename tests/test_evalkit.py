import json
from pathlib import Path

import pytest

from evalkit.cli import main
from evalkit.compare import compare
from evalkit.io import load_pages, load_records
from evalkit.labels import check_labels
from evalkit.ledger import file_hash, record_heldout
from evalkit.report import summarize
from evalkit.scoring import score
from evalkit.stats import mcnemar_exact, wilson

EX = Path(__file__).parent.parent / "examples"
Q = {"id": "q", "question": "?", "answerable": True, "doc_id": "d", "pages": [3], "key_facts": [["three months", "3 months"]]}


# ── scoring ──────────────────────────────────────────────────────────────────
def test_alternative_wordings_and_normalisation():
    assert score(Q, {"answer": "It needs 3 months."})["correct"]
    assert not score(Q, {"answer": "It needs 6 months."})["correct"]
    q = dict(Q, key_facts=["1,000 shares", "March, June, September and December"])
    assert score(q, {"answer": "1000 shares; in March, June, September, and December"})["correct"]


def test_forbidden_fact_with_exception():
    q = dict(Q, key_facts=["WMR"], forbidden_facts=[{"any": ["except latin american"], "unless": ["2000"]}])
    assert not score(q, {"answer": "WMR, except Latin American countries."})["correct"]
    assert score(q, {"answer": "WMR, except Latin American countries until 2000."})["correct"]


def test_refusals_and_grounding():
    unans = {"id": "u", "question": "?", "answerable": False}
    assert score(unans, {"answer": "NOT_FOUND"})["correct"]
    assert not score(unans, {"answer": "It is 20%."})["correct"]
    assert not score(Q, {"answer": "3 months", "refused": True})["correct"]  # explicit flag wins
    row = score(Q, {"answer": "3 months", "citations": [["d", 3], ["d", 3], ["d", 9]]})
    assert row["grounded"] and row["cited_ok"] == [True, False]  # duplicate citation counted once
    assert row["retrieval_hit"] is None  # unknown, not False, when nothing was reported


# ── statistics ───────────────────────────────────────────────────────────────
def test_wilson_known_values():
    lo, hi = wilson(77, 94)  # held-out result of methodology-rag
    assert round(100 * lo) == 73 and round(100 * hi) == 88
    assert wilson(14, 14)[1] == 1.0 and round(wilson(14, 14)[0], 3) == 0.785


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)  # 6 flips, all one way
    assert mcnemar_exact(1, 1) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    # 2 vs 10 flips: 2 * (C(12,0) + C(12,1) + C(12,2)) / 2^12 = 2 * 79 / 4096
    assert mcnemar_exact(2, 10) == pytest.approx(158 / 4096)


def test_compare_flips():
    a = [{"id": "1", "correct": True}, {"id": "2", "correct": False}, {"id": "3", "correct": False}]
    b = [{"id": "1", "correct": True}, {"id": "2", "correct": True}, {"id": "3", "correct": True}, {"id": "x", "correct": True}]
    r = compare(a, b)
    assert r["n_common"] == 3 and r["only_b_right"] == ["2", "3"] and r["only_a_right"] == []
    assert r["delta_points"] == 66.7 and not r["significant"]  # two flips are never significant


# ── labels and ledger ────────────────────────────────────────────────────────
def test_label_check_catches_wrong_page_and_overlap():
    pages = {("d", 3): "It must have traded for three months.", ("d", 4): "Something else."}
    assert check_labels([Q], pages) == []
    bad = dict(Q, id="b", pages=[4])
    assert any("not on" in p for p in check_labels([bad], pages))
    assert any("shared" in p for p in check_labels([Q], pages, [[dict(Q, id="other")]]))


def test_ledger_warns_when_heldout_reused_on_another_system(tmp_path):
    golden = tmp_path / "heldout.jsonl"
    golden.write_text('{"id": "q"}\n', encoding="utf-8")
    ledger = tmp_path / "ledger.jsonl"
    assert record_heldout(ledger, golden, "sysA", 80.0) == []
    assert any("another system" in w for w in record_heldout(ledger, golden, "sysB", 82.0))
    assert any("re-run" in w for w in record_heldout(ledger, golden, "sysA", 80.0))


def test_file_hash_ignores_line_endings(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_bytes(b'{"x": 1}\n{"y": 2}\n')
    b.write_bytes(b'{"x": 1}\r\n{"y": 2}\r\n')
    assert file_hash(a) == file_hash(b)


# ── end to end on the example files ──────────────────────────────────────────
def test_cli_on_examples(tmp_path, capsys):
    out = tmp_path / "report.json"
    assert main(["score", str(EX / "questions.jsonl"), str(EX / "predictions.jsonl"), "-o", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in report["rows"]}
    assert by_id["q1"]["correct"] and by_id["q2"]["correct"] and by_id["q4"]["correct"]
    assert not by_id["q3"]["correct"] and by_id["q3"]["contradicted"]  # outdated claim is caught
    assert report["summary"]["accuracy"]["k"] == 3
    assert main(["check", str(EX / "questions.jsonl"), "--pages", str(EX / "pages.jsonl")]) == 0
    assert main(["gate", str(out), "--min", "accuracy=70"]) == 0
    assert main(["gate", str(out), "--min", "accuracy=80"]) == 1


def test_missing_prediction_counts_as_wrong(tmp_path):
    preds = tmp_path / "p.jsonl"
    preds.write_text('{"id": "q1", "answer": "10%"}\n', encoding="utf-8")
    out = tmp_path / "r.json"
    main(["score", str(EX / "questions.jsonl"), str(preds), "-o", str(out)])
    rows = json.loads(out.read_text(encoding="utf-8"))["rows"]
    assert sum(r.get("missing", False) for r in rows) == 3 and sum(r["correct"] for r in rows) == 1


def test_loaders_accept_jsonl_list_and_rows(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"rows": [{"id": 1}], "summary": {}}), encoding="utf-8")
    assert load_records(p) == [{"id": 1}]
    p.write_text(json.dumps([{"id": 2}]), encoding="utf-8")
    assert load_records(p) == [{"id": 2}]
    assert len(load_pages(EX / "pages.jsonl")) == 3
