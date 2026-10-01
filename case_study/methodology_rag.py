"""Case study: re-analyse the saved runs of methodology-rag with evalkit.

Expects the methodology-rag repo next to this one (../methodology-rag) with its parsed page cache
(data/index/pages, built locally by `mrag ingest`; the PDFs aren't redistributed).

  python case_study/methodology_rag.py > case_study/RESULTS.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evalkit import compare, score, summarize  # noqa: E402
from evalkit.io import load_pages, load_records  # noqa: E402
from evalkit.labels import check_labels  # noqa: E402

RAG = Path(__file__).resolve().parents[2] / "methodology-rag"
RUNS = RAG / "evals" / "results"
DEV = load_records(RAG / "evals" / "golden.jsonl")
HELDOUT = load_records(RAG / "evals" / "heldout.jsonl")


def scored(run: str, questions=DEV) -> list[dict]:
    preds = {p["id"]: p for p in load_records(RUNS / f"{run}.json")}
    return [score(q, preds[q["id"]], default_tag="lexical") for q in questions if q["id"] in preds]


def pct(s: dict | None) -> str:
    return "–" if s is None else f"{s['rate']}% ({s['ci95'][0]:.0f}–{s['ci95'][1]:.0f})"


# decisions taken while building methodology-rag: (question asked, metric, run A, run B)
DECISIONS = [
    ("Answer model: qwen2.5:3b -> qwen3:4b-instruct", "correct",
     "20260930-175400_dev_hybrid+rewrite_bge-m3_qwen2.5-3b_prompt-v1_k6",
     "20260930-182919_dev_hybrid+rewrite_bge-m3_qwen3-4b-instruct_prompt-v1_k6"),
    ("Query rewriting + citation check (qwen2.5)", "correct",
     "20260928-234203_hybrid_qwen2.5-3b", "20260928-235852_hybrid_rewrite_prompt-v1_qwen2.5-3b"),
    ("Prompt v1 -> v2 (stricter citations)", "correct",
     "20260928-235852_hybrid_rewrite_prompt-v1_qwen2.5-3b", "20260928-235305_hybrid_rewrite_qwen2.5-3b"),
    ("Prompt v1 -> v3 (answer-first, example format)", "correct",
     "20260928-235852_hybrid_rewrite_prompt-v1_qwen2.5-3b", "20260929-000353_hybrid_rewrite_prompt-v3_qwen2.5-3b"),
    ("Dense retrieval: nomic-embed-text -> bge-m3", "retrieval_hit",
     "20260929-223241_dev_dense_nomic-embed-text_retrieval_k6", "20260929-223317_dev_dense_bge-m3_retrieval_k6"),
    ("Hybrid retrieval: + LLM query rewrite (bge-m3)", "retrieval_hit",
     "20260929-212853_dev_hybrid_bge-m3_retrieval_k6", "20260929-213007_dev_hybrid+rewrite_bge-m3_retrieval_k6"),
    ("Reranker: none -> MiniLM-L-6 (bge-m3, rewrite)", "retrieval_hit",
     "20260929-213007_dev_hybrid+rewrite_bge-m3_retrieval_k6",
     "20260929-213620_dev_hybrid+rewrite_rerank-ms-marco-MiniLM-L-6-v2_bge-m3_retrieval_k6"),
    ("Reranker: none -> bge-reranker-base (bge-m3, rewrite)", "retrieval_hit",
     "20260929-213007_dev_hybrid+rewrite_bge-m3_retrieval_k6",
     "20260929-220511_dev_hybrid+rewrite_rerank-bge-reranker-base_bge-m3_retrieval_k6"),
    ("Reranker: MiniLM-L-6 -> bge-reranker-base", "retrieval_hit",
     "20260929-213620_dev_hybrid+rewrite_rerank-ms-marco-MiniLM-L-6-v2_bge-m3_retrieval_k6",
     "20260929-220511_dev_hybrid+rewrite_rerank-bge-reranker-base_bge-m3_retrieval_k6"),
]

print("# Case study: methodology-rag\n")
print("All runs re-scored by evalkit against methodology-rag's current, audited labels.\n")

pages = load_pages(RAG / "data" / "index" / "pages")
for name, qs, other in [("dev", DEV, HELDOUT), ("held-out", HELDOUT, DEV)]:
    problems = check_labels(qs, pages, [other])
    print(f"- Label check, {name} set ({len(qs)} questions): {'OK' if not problems else problems}")

final = summarize(scored("20261001-220109_heldout_hybrid+rewrite_bge-m3_qwen3-4b-instruct_prompt-v1_k6", HELDOUT))
dev = summarize(scored("20260930-182919_dev_hybrid+rewrite_bge-m3_qwen3-4b-instruct_prompt-v1_k6"))
print(f"- Final system, held-out: accuracy {pct(final['accuracy'])}, grounded {pct(final['grounded_accuracy'])}, "
      f"unanswerable refused {pct(final['refusal_accuracy'])}")
print(f"- Same system, dev: accuracy {pct(dev['accuracy'])}\n")

print("## Which decisions were real improvements?\n")
print("Paired on the same dev questions; exact McNemar test. *n* = questions where the two versions disagree.\n")
print("| Change | Metric | A | B | Δ points | A-only / B-only right | p | Verdict |")
print("|---|---|---|---|---|---|---|---|")
for title, metric, a, b in DECISIONS:
    r = compare(scored(a), scored(b), metric)
    verdict = "**significant**" if r["significant"] else "not significant"
    label = "accuracy" if metric == "correct" else "recall@6"
    print(f"| {title} | {label} | {r['a']}% | {r['b']}% | {r['delta_points']:+} | "
          f"{len(r['only_a_right'])} / {len(r['only_b_right'])} | {r['p_value']} | {verdict} |")

print("""
**Reading this.** Only the answer-model swap is a statistically significant improvement. The two
"better" prompts lost 8-9 points, but at p = 0.05-0.06 even that is borderline on this many questions.
Retrieval changes of 3-7 points are not distinguishable from chance with ~118 questions: that does not
mean they had no effect, only that this test set is too small to show it. Detecting a 5-point gain
reliably needs a few hundred questions, or a paired design with more disagreements.
""")
