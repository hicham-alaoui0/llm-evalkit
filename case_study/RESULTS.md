# Case study: methodology-rag

All runs re-scored by evalkit against methodology-rag's current, audited labels.

- Label check, dev set (126 questions): OK
- Label check, held-out set (94 questions): OK
- Final system, held-out: accuracy 81.9% (73–88), grounded 66.0% (56–75), unanswerable refused 100.0% (78–100)
- Same system, dev: accuracy 81.7% (74–88)

## Which decisions were real improvements?

Paired on the same dev questions; exact McNemar test. *n* = questions where the two versions disagree.

| Change | Metric | A | B | Δ points | A-only / B-only right | p | Verdict |
|---|---|---|---|---|---|---|---|
| Answer model: qwen2.5:3b -> qwen3:4b-instruct | accuracy | 69.8% | 81.7% | +11.9 | 4 / 19 | 0.0026 | **significant** |
| Query rewriting + citation check (qwen2.5) | accuracy | 70.6% | 69.8% | -0.8 | 9 / 8 | 1.0 | not significant |
| Prompt v1 -> v2 (stricter citations) | accuracy | 69.8% | 61.9% | -7.9 | 16 / 6 | 0.0525 | not significant |
| Prompt v1 -> v3 (answer-first, example format) | accuracy | 69.8% | 61.1% | -8.7 | 20 / 9 | 0.0614 | not significant |
| Dense retrieval: nomic-embed-text -> bge-m3 | recall@6 | 75.4% | 78.8% | +3.4 | 6 / 10 | 0.4545 | not significant |
| Hybrid retrieval: + LLM query rewrite (bge-m3) | recall@6 | 83.9% | 80.5% | -3.4 | 6 / 2 | 0.2891 | not significant |
| Reranker: none -> MiniLM-L-6 (bge-m3, rewrite) | recall@6 | 80.5% | 85.6% | +5.1 | 5 / 11 | 0.2101 | not significant |
| Reranker: none -> bge-reranker-base (bge-m3, rewrite) | recall@6 | 80.5% | 87.3% | +6.8 | 7 / 15 | 0.1338 | not significant |
| Reranker: MiniLM-L-6 -> bge-reranker-base | recall@6 | 85.6% | 87.3% | +1.7 | 6 / 8 | 0.7905 | not significant |

**Reading this.** Only the answer-model swap is a statistically significant improvement. The two
"better" prompts lost 8-9 points, but at p = 0.05-0.06 even that is borderline on this many questions.
Retrieval changes of 3-7 points are not distinguishable from chance with ~118 questions: that does not
mean they had no effect, only that this test set is too small to show it. Detecting a 5-point gain
reliably needs a few hundred questions, or a paired design with more disagreements.

