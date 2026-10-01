# llm-evalkit

Deterministic evaluation for RAG and LLM question answering. Give it labelled questions and your
system's answers; it tells you how often the system is right, how sure you can be of that number,
and whether version B is actually better than version A or just luckier.

No LLM judge, no dependencies beyond the Python standard library.

Real output, on [methodology-rag](https://github.com/hicham-alaoui0/methodology-rag)'s saved runs:

```
$ evalkit score heldout.jsonl final_run.json
| Metric | Score | 95% interval |
|---|---|---|
| Accuracy | **81.9%** (77/94) | 73–88% |
| Grounded accuracy | **66.0%** (62/94) | 56–75% |
| Unanswerable refused | **100.0%** (14/14) | 78–100% |
...

$ evalkit compare qwen2.5_report.json qwen3_report.json
correct on 126 shared questions: A 69.8%  ->  B 81.7%  (+11.9 points)
only A right: 4  p012 p028 p029 p035
only B right: 19  p003 p008 p015 p022 p025 q009 q013 ...
exact McNemar p = 0.0026: a real difference at the 5% level
```

## Why

I built this out of [methodology-rag](https://github.com/hicham-alaoui0/methodology-rag), a local RAG
assistant over index-provider rulebooks, after the eval harness kept catching problems that a single
accuracy number hides:

- **Test wording can flatter a system.** Keyword search found the right page for 98% of questions written
  close to the documents, and 42% of the same facts paraphrased.
- **Scorers have bugs too.** A correct answer failed because of an Oxford comma; another passed while
  stating a rule that stopped applying in 2000.
- **Most "improvements" were noise.** Re-analysed with this toolkit, only 1 of 9 design decisions in that
  project (swapping the answer model) is a statistically significant gain. See
  [case_study/RESULTS.md](case_study/RESULTS.md).
- **Held-out sets leak.** Once a held-out result influences a change, it has become a dev set.

## What it does

| Command | |
|---|---|
| `evalkit score Q P` | Score predictions: accuracy, grounded accuracy (right answer **and** a citation to a gold page), refusals on unanswerable questions, false refusals, retrieval recall, citation precision; each with a 95% Wilson interval, overall and per tag |
| `evalkit compare A B` | Paired comparison of two runs on the same questions: which questions flipped, and an exact McNemar test |
| `evalkit check Q --pages` | Check labels against the source text: every expected fact must appear on its gold page; no duplicates; no gold pages shared with another set |
| `evalkit gate REPORT` | Exit 1 below a threshold, on the rate or on the lower bound of its interval (for CI) |
| `evalkit freeze PATHS` | Fingerprint the files that define a system version (line-ending independent) |
| `score --heldout` | Log the held-out run in a ledger and warn if the same held-out set was already used on another system version |

## Install

```bash
pip install git+https://github.com/hicham-alaoui0/llm-evalkit
evalkit score examples/questions.jsonl examples/predictions.jsonl
```

## Formats

**Questions** (JSONL, one per line):

```json
{"id": "q2", "question": "How long must a small IPO trade before an MSCI index review?",
 "answerable": true, "doc_id": "msci", "pages": [21],
 "key_facts": [["three months", "3 months"]], "style": "paraphrase"}
{"id": "q4", "question": "What is the maximum weight in the Nasdaq-100?", "answerable": false}
```

- `key_facts`: every fact must appear in a correct answer. A fact is a string, or a list of accepted
  wordings with the source's own wording first (that's the one `check` looks for on the page).
- `forbidden_facts` (optional): wordings that make an answer wrong even when all key facts are present,
  e.g. an outdated rule from the same page. `{"any": [...], "unless": [...]}` adds an exception.
- `answerable: false`: the correct answer is a refusal.
- `style` / `tags`: anything you want results broken down by.

Matching ignores case, thousands separators, `10 %` vs `10%`, "per cent", curly quotes and Oxford commas.

**Predictions** (JSONL, a JSON list, or a JSON object with a `rows` list):

```json
{"id": "q2", "answer": "At least 3 months [2].", "citations": [["msci", 21]],
 "retrieved": [["msci", 21], ["msci", 22]], "latency_s": 2.1}
```

`refused` is optional; without it, an answer containing `NOT_FOUND` (configurable) counts as a refusal.
A question with no prediction counts as wrong.

## Limits

Fact matching is transparent and reproducible, but it checks that the right facts are **present**, not
that nothing wrong was **added**. `forbidden_facts` covers known failure patterns, not all of them, so
read a sample of answers by hand before trusting a number. Intervals and p-values assume questions are
independent; near-duplicate questions make them look tighter than they are.

## Development

```bash
pip install -e .[dev]
pytest
```

MIT licensed.
