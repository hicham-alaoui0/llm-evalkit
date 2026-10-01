"""Small-sample statistics for eval results: Wilson intervals and the exact McNemar test."""

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n, in [0, 1].
    Better than p ± 1.96·sqrt(p(1-p)/n) for small n and for rates near 0% or 100%."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value for a paired comparison of two systems on the same questions.

    a_only: questions system A got right and B got wrong; b_only: the reverse. Questions both got
    right (or both wrong) carry no information about which system is better and are ignored.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)
