"""Statistical significance tests: bootstrap CI and McNemar's test."""

from __future__ import annotations

import math
import random
from typing import Callable


def bootstrap_ci(
    values: list[float],
    stat_fn: Callable[[list[float]], float] | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for a statistic.

    Args:
        values: Sample values.
        stat_fn: Function that maps a list of floats to a scalar statistic.
                 Defaults to the mean.
        n_boot: Number of bootstrap resamples.
        alpha: Significance level (e.g. 0.05 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (lower_bound, upper_bound) of the confidence interval.
    """
    if stat_fn is None:
        stat_fn = lambda x: sum(x) / len(x)  # noqa: E731
    rng = random.Random(seed)
    boot_stats: list[float] = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_stats.append(stat_fn(sample))
    boot_stats.sort()
    lo = boot_stats[int(n_boot * alpha / 2)]
    hi = boot_stats[int(n_boot * (1 - alpha / 2))]
    return lo, hi


def mcnemar_test(correct_a: list[int], correct_b: list[int]) -> float:
    """Return approximate p-value for McNemar's test.

    Uses the continuity-corrected chi-squared statistic with 1 degree of
    freedom.  The returned p-value is an approximation based on the
    chi-squared CDF evaluated at the test statistic.

    Args:
        correct_a: Binary correctness vector for model A (0/1 per example).
        correct_b: Binary correctness vector for model B (0/1 per example).

    Returns:
        Approximate p-value (float).  Values < 0.05 suggest a significant
        difference between the two models.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("correct_a and correct_b must have the same length.")
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a == 1 and bb == 0)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if a == 0 and bb == 1)
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # Approximate p-value from chi-squared(1) CDF
    return math.exp(-chi2 / 2)
