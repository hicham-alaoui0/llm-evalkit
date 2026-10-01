"""Check labels against the source text before trusting any score computed with them."""

from collections import Counter

from .scoring import alternatives
from .text import normalize


def check_labels(questions: list[dict], pages: dict[tuple[str, int], str], others: list[list[dict]] = ()) -> list[str]:
    """Return a list of problems (empty = labels are consistent with the source).

    pages: {(doc_id, page_number): page text}. The first wording of every key fact (the source's
    own wording) must appear on at least one of the question's gold pages. Also flags duplicate ids,
    duplicate questions, and gold pages shared with another question set (e.g. dev vs held-out).
    """
    problems = []
    ids = Counter(q["id"] for q in questions)
    problems += [f"duplicate id {i}" for i, c in ids.items() if c > 1]
    texts = Counter(normalize(q["question"]) for q in questions)
    problems += [f"duplicate question: {t[:60]!r}" for t, c in texts.items() if c > 1]
    norm = {k: normalize(v) for k, v in pages.items()}

    for q in questions:
        if not q["answerable"]:
            continue
        missing_pages = [p for p in q["pages"] if (q["doc_id"], p) not in norm]
        if missing_pages:
            problems.append(f"{q['id']}: no text for {q['doc_id']} page(s) {missing_pages}")
            continue
        text = " ".join(norm[(q["doc_id"], p)] for p in q["pages"])
        for fact in q["key_facts"]:
            if normalize(alternatives(fact)[0]) not in text:
                problems.append(f"{q['id']}: {alternatives(fact)[0]!r} is not on {q['doc_id']} p.{q['pages']}")

    mine = {(q["doc_id"], p) for q in questions if q["answerable"] for p in q["pages"]}
    for other in others:
        shared = mine & {(q["doc_id"], p) for q in other if q["answerable"] for p in q["pages"]}
        problems += [f"gold page shared with another set: {d} p.{p}" for d, p in sorted(shared)]
    return problems
