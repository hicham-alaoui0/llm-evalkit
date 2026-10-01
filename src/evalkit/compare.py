"""Paired comparison of two scored runs on the same questions."""

from .stats import mcnemar_exact


def compare(a: list[dict], b: list[dict], key: str = "correct") -> dict:
    """Which questions flipped between run A and run B, and is the difference more than noise?

    Only questions present in both runs are compared. A difference is "significant" here at p < 0.05
    (exact McNemar); with ~100 questions, a few points of difference usually is not.
    """
    ra, rb = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    common = [i for i in ra if i in rb and ra[i].get(key) is not None and rb[i].get(key) is not None]
    a_only = sorted(i for i in common if ra[i][key] and not rb[i][key])
    b_only = sorted(i for i in common if rb[i][key] and not ra[i][key])
    n = len(common)
    acc_a = sum(bool(ra[i][key]) for i in common)
    acc_b = sum(bool(rb[i][key]) for i in common)
    p = mcnemar_exact(len(a_only), len(b_only))
    return {
        "metric": key,
        "n_common": n,
        "a": round(100 * acc_a / n, 1) if n else None,
        "b": round(100 * acc_b / n, 1) if n else None,
        "delta_points": round(100 * (acc_b - acc_a) / n, 1) if n else None,
        "only_a_right": a_only,
        "only_b_right": b_only,
        "p_value": round(p, 4),
        "significant": p < 0.05,
    }
