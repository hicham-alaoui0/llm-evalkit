"""Deterministic scoring of one answer against one labelled question.

A question (JSON object):
  id            unique id
  question      text
  answerable    false for questions the corpus does not answer (correct answer: a refusal)
  doc_id, pages gold source (pages a correct citation may point to)
  key_facts     facts a correct answer must contain; each is a string or a list of accepted
                wordings, the first being the source's own wording: ["three months", "3 months"]
  forbidden_facts  optional; statements that make an answer wrong even if every key fact is there.
                Each is a wording, a list of wordings, or {"any": [...], "unless": [...]}.

A prediction (JSON object):
  id, answer    the system's answer text
  refused       optional bool; if absent, the answer is a refusal when it contains the refusal marker
  citations     optional [[doc_id, page], ...] cited by the answer
  retrieved     optional [[doc_id, page], ...] passed to the model (for retrieval recall)
  latency_s     optional
"""

from .text import normalize

Fact = str | list[str]


def alternatives(fact: Fact) -> list[str]:
    return [fact] if isinstance(fact, str) else list(fact)


def fact_found(answer: str, fact: Fact) -> bool:
    a = normalize(answer)
    return any(normalize(alt) in a for alt in alternatives(fact))


def forbidden_found(answer: str, rule: Fact | dict) -> bool:
    if isinstance(rule, dict):
        return fact_found(answer, rule["any"]) and not fact_found(answer, rule.get("unless", []))
    return fact_found(answer, rule)


def is_refusal(pred: dict, marker: str) -> bool:
    if "refused" in pred:
        return bool(pred["refused"])
    return marker.upper() in (pred.get("answer") or "").upper()


def score(question: dict, pred: dict, refusal_marker: str = "NOT_FOUND", default_tag: str = "untagged") -> dict:
    """Return the prediction with verdict fields added. Missing optional inputs give None, not False."""
    answer = pred.get("answer") or ""
    refused = is_refusal(pred, refusal_marker)
    gold = {(question.get("doc_id"), p) for p in question.get("pages", [])}
    # dedupe citations: several chunks can share a page
    cites = list(dict.fromkeys(tuple(c) for c in pred.get("citations") or []))
    row = {
        "id": question["id"],
        "answerable": question["answerable"],
        "tags": _tags(question, default_tag),
        "refused": refused,
        "latency_s": pred.get("latency_s"),
    }
    if not question["answerable"]:
        row.update(correct=refused, grounded=refused, facts=[], contradicted=False, retrieval_hit=None, cited_ok=[])
        return row

    facts = [{"fact": alternatives(f)[0], "found": fact_found(answer, f)} for f in question["key_facts"]]
    contradicted = any(forbidden_found(answer, r) for r in question.get("forbidden_facts", []))
    correct = not refused and all(f["found"] for f in facts) and not contradicted
    cited_ok = [c in gold for c in cites]
    retrieved = pred.get("retrieved")
    row.update(
        correct=correct,
        grounded=correct and any(cited_ok),
        facts=facts,
        contradicted=contradicted,
        cited_ok=cited_ok,
        retrieval_hit=None if retrieved is None else any(tuple(r) in gold for r in retrieved),
    )
    return row


def _tags(q: dict, default_tag: str) -> list[str]:
    tags = list(q.get("tags", []))
    if q.get("style"):
        tags.append(q["style"])
    if not q["answerable"]:
        tags.append("unanswerable")
    elif not tags:
        tags.append(default_tag)
    return tags
