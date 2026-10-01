"""Aggregate scored rows into rates with 95% intervals, overall and per tag."""

import statistics

from .stats import wilson


def rate(rows: list[dict], key: str) -> dict | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    k, n = sum(map(bool, vals)), len(vals)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(100 * k / n, 1), "ci95": [round(100 * lo, 1), round(100 * hi, 1)]}


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    cited = [{"ok": ok} for r in answerable for ok in r["cited_ok"]]
    lat = sorted(r["latency_s"] for r in rows if r.get("latency_s") is not None)
    summary = {
        "n": len(rows),
        "accuracy": rate(rows, "correct"),
        "grounded_accuracy": rate(rows, "grounded"),
        "answerable_accuracy": rate(answerable, "correct"),
        "refusal_accuracy": rate(unanswerable, "correct"),
        "false_refusal_rate": rate(answerable, "refused"),
        "retrieval_recall": rate(answerable, "retrieval_hit"),
        "citation_precision": rate(cited, "ok"),
        "contradicted": sum(r["contradicted"] for r in answerable),
        "latency_s": {"p50": round(statistics.median(lat), 2), "p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2)}
        if lat else None,
        "by_tag": {},
    }
    for tag in sorted({t for r in rows for t in r["tags"]}):
        sub = [r for r in rows if tag in r["tags"]]
        summary["by_tag"][tag] = {"accuracy": rate(sub, "correct"), "grounded_accuracy": rate(sub, "grounded"),
                                  "retrieval_recall": rate([r for r in sub if r["answerable"]], "retrieval_hit")}
    return summary


def _fmt(r: dict | None) -> str:
    return "–" if r is None else f"**{r['rate']}%** ({r['k']}/{r['n']}) | {r['ci95'][0]:.0f}–{r['ci95'][1]:.0f}%"


def to_markdown(summary: dict, title: str = "Evaluation") -> str:
    lines = [f"### {title} ({summary['n']} questions)", "", "| Metric | Score | 95% interval |", "|---|---|---|"]
    for label, key in [("Accuracy", "accuracy"), ("Grounded accuracy", "grounded_accuracy"),
                       ("Answerable accuracy", "answerable_accuracy"), ("Unanswerable refused", "refusal_accuracy"),
                       ("False refusals", "false_refusal_rate"), ("Retrieval recall", "retrieval_recall"),
                       ("Citation precision", "citation_precision")]:
        if summary[key] is not None:
            lines.append(f"| {label} | {_fmt(summary[key])} |")
    for tag, m in summary["by_tag"].items():
        lines.append(f"| Accuracy · {tag} | {_fmt(m['accuracy'])} |")
    if summary["latency_s"]:
        lines += ["", f"Latency p50 {summary['latency_s']['p50']} s, p95 {summary['latency_s']['p95']} s."]
    if summary["contradicted"]:
        lines.append(f"{summary['contradicted']} answer(s) contained a forbidden fact.")
    return "\n".join(lines)
