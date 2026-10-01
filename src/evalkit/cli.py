import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .compare import compare
from .io import load_pages, load_records, write_json
from .labels import check_labels
from .ledger import manifest, record_heldout
from .report import summarize, to_markdown
from .scoring import score


def cmd_score(args) -> int:
    questions = load_records(args.questions)
    preds = {p["id"]: p for p in load_records(args.predictions)}
    rows = []
    for q in questions:
        pred = preds.get(q["id"])
        row = score(q, pred or {"answer": "", "refused": False}, args.refusal, args.default_tag)
        if pred is None:
            row["missing"] = True  # counted as wrong: a system that skips questions shouldn't score higher
        rows.append(row)
    missing = sum(r.get("missing", False) for r in rows)
    summary = summarize(rows)
    report = {"questions": str(args.questions), "predictions": str(args.predictions), "summary": summary, "rows": rows}

    if args.heldout:
        system = manifest(args.system)["fingerprint"] if args.system else "unspecified"
        acc = summary["accuracy"]["rate"] if summary["accuracy"] else None
        for w in record_heldout(Path(args.ledger), Path(args.questions), system, acc):
            print(f"WARNING: {w}", file=sys.stderr)
        report["heldout"] = {"system_fingerprint": system, "ledger": args.ledger}

    print(to_markdown(summary, args.title))
    if missing:
        print(f"\n{missing} question(s) had no prediction and were scored as wrong.")
    if args.output:
        write_json(report, args.output)
    if args.markdown:
        Path(args.markdown).write_text(to_markdown(summary, args.title) + "\n", encoding="utf-8")
    return 0


def cmd_compare(args) -> int:
    a, b = load_records(args.a), load_records(args.b)
    r = compare(a, b, args.metric)
    print(f"{args.metric} on {r['n_common']} shared questions: A {r['a']}%  ->  B {r['b']}%  ({r['delta_points']:+} points)")
    print(f"only A right: {len(r['only_a_right'])}  {' '.join(r['only_a_right'])}")
    print(f"only B right: {len(r['only_b_right'])}  {' '.join(r['only_b_right'])}")
    verdict = "a real difference" if r["significant"] else "not distinguishable from noise"
    print(f"exact McNemar p = {r['p_value']}: {verdict} at the 5% level")
    if args.output:
        write_json(r, args.output)
    return 0


def cmd_check(args) -> int:
    questions = load_records(args.questions)
    pages = load_pages(args.pages)
    others = [load_records(p) for p in args.against]
    problems = check_labels(questions, pages, others)
    answerable = sum(q["answerable"] for q in questions)
    print(f"{len(questions)} questions ({answerable} answerable) checked against {len(pages)} pages")
    for p in problems:
        print(f"  - {p}")
    print("labels OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def cmd_gate(args) -> int:
    summary = json.loads(Path(args.report).read_text(encoding="utf-8"))["summary"]
    failed = []
    for spec, use_low in [(s, False) for s in args.min] + [(s, True) for s in args.min_ci_low]:
        metric, _, threshold = spec.partition("=")
        m = summary.get(metric)
        if m is None:
            failed.append(f"{metric}: not in report")
            continue
        value = m["ci95"][0] if use_low else m["rate"]
        ok = value >= float(threshold)
        print(f"{'ok  ' if ok else 'FAIL'} {metric}{' (95% CI lower bound)' if use_low else ''} = {value} (min {threshold})")
        if not ok:
            failed.append(metric)
    return 1 if failed else 0


def cmd_freeze(args) -> int:
    m = manifest(args.paths)
    write_json(m, args.output)
    print(f"fingerprint {m['fingerprint']} over {len(m['files'])} files -> {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evalkit", allow_abbrev=False,
                                description="Deterministic evaluation for RAG / LLM question answering")
    p.add_argument("--version", action="version", version=f"evalkit {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score predictions against labelled questions")
    s.add_argument("questions")
    s.add_argument("predictions", help="JSONL, JSON list, or a JSON object with a 'rows' list")
    s.add_argument("-o", "--output", help="write the full report (summary + per-question rows) as JSON")
    s.add_argument("--markdown", help="also write the summary table as markdown")
    s.add_argument("--title", default="Evaluation")
    s.add_argument("--refusal", default="NOT_FOUND", help="text that marks a refusal when 'refused' is absent")
    s.add_argument("--default-tag", default="untagged", help="tag for answerable questions without one")
    s.add_argument("--heldout", action="store_true", help="record this run in the held-out ledger")
    s.add_argument("--ledger", default="evals/heldout_ledger.jsonl")
    s.add_argument("--system", nargs="*", default=[], help="files/dirs that define the system version (fingerprinted)")
    s.set_defaults(fn=cmd_score)

    c = sub.add_parser("compare", help="paired comparison of two scored reports (exact McNemar test)")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--metric", default="correct", choices=["correct", "grounded", "retrieval_hit"])
    c.add_argument("-o", "--output")
    c.set_defaults(fn=cmd_compare)

    k = sub.add_parser("check", help="check labels against the source page texts")
    k.add_argument("questions")
    k.add_argument("--pages", required=True, help="JSONL of {doc_id, page, text} or a dir of <doc>.pages.json")
    k.add_argument("--against", nargs="*", default=[], help="other question sets that must not share gold pages")
    k.set_defaults(fn=cmd_check)

    g = sub.add_parser("gate", help="exit 1 if a report misses a threshold (for CI)")
    g.add_argument("report")
    g.add_argument("--min", nargs="*", default=[], metavar="METRIC=PCT", help="e.g. accuracy=80")
    g.add_argument("--min-ci-low", nargs="*", default=[], metavar="METRIC=PCT",
                   help="threshold on the 95%% CI lower bound, e.g. accuracy=70")
    g.set_defaults(fn=cmd_gate)

    f = sub.add_parser("freeze", help="fingerprint the files that define a system version")
    f.add_argument("paths", nargs="+")
    f.add_argument("-o", "--output", default="freeze.json")
    f.set_defaults(fn=cmd_freeze)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
